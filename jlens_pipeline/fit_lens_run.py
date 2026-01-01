"""Production Jacobian-lens fit: J_l = E[d h_final / d h_l] for one model.

Fits the classic "layer l -> final layer" lens (source layers 0..L-2, target L-1),
accumulates a running mean over wikitext-103 prompts, checkpoints atomically so a
restart resumes rather than redoing work, stops early on a convergence criterion,
then writes the fp16 J file plus per-layer effective dimension.

Safety rules baked in:
  * bf16 only - never load a quantized checkpoint across multiple devices in one
    process (FP8/MXFP4 kernels read the current CUDA device and silently produce
    garbage when driven from another device in the same process).
  * A coherence gate runs before fitting: next-token cross-entropy on real text
    must be sane, or we abort rather than burn hours fitting a broken model.

Requires the upstream jacobian-lens package, https://github.com/anthropics/jacobian-lens
(Apache-2.0). Either install it, or set JLENS_UPSTREAM to a checkout.
"""
import argparse, glob, json, math, os, sys, time
import torch

if os.environ.get("JLENS_UPSTREAM"):
    sys.path.insert(0, os.environ["JLENS_UPSTREAM"])
import jlens
from jlens import fitting
from jlens.lens import JacobianLens

# Fit prompts come from WikiText-103 (raw), the corpus the upstream lens uses in
# jlens.examples.load_wikitext_prompts. Source of record, a public dataset repo:
#
#     https://huggingface.co/datasets/Salesforce/wikitext
#     config wikitext-103-raw-v1, split train, 2 parquet shards, ~314 MB
#
# Upstream streams it with `datasets`; we read those parquet shards directly so the
# fit venv needs only pyarrow and huggingface_hub. Selection below is our own: we
# skip " = = Title = = " section headers, truncate to max_chars and shuffle with a
# fixed seed, where upstream simply takes the first n records over min_chars.
#
# The revision is pinned so a refit draws the same prompts. It is the repo head as
# of 2024-01-04 and is the snapshot the shipped lenses were fitted on.
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- prompts
def wikitext_train_files():
    """Sorted paths of the WikiText-103-raw train shards.

    Resolution order: $WIKITEXT_DIR, for a copy already on disk, then the
    HuggingFace cache, which downloads on a miss and honours HF_HOME and
    HF_HUB_OFFLINE=1.
    """
    local = os.environ.get("WIKITEXT_DIR")
    if local:
        files = (sorted(glob.glob(os.path.join(local, WIKITEXT_CONFIG, "train-*.parquet")))
                 or sorted(glob.glob(os.path.join(local, "train-*.parquet"))))
        if not files:
            raise FileNotFoundError(
                f"WIKITEXT_DIR={local} holds no train-*.parquet; expected the "
                f"{WIKITEXT_CONFIG} shards of {WIKITEXT_REPO}")
        return files
    from huggingface_hub import snapshot_download
    snap = snapshot_download(WIKITEXT_REPO, repo_type="dataset",
                             revision=WIKITEXT_REVISION,
                             allow_patterns=f"{WIKITEXT_CONFIG}/train-*.parquet")
    files = sorted(glob.glob(os.path.join(snap, WIKITEXT_CONFIG, "train-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no {WIKITEXT_CONFIG} train shards under {snap}; re-download "
            f"{WIKITEXT_REPO} or point WIKITEXT_DIR at a local copy")
    return files


def load_prompts(n, max_chars=2000, min_chars=600, seed=42):
    """Documents from wikitext-103 train; see the notes above on provenance."""
    import pyarrow.parquet as pq
    files = wikitext_train_files()
    log(f"wikitext: {len(files)} train shards from {os.path.dirname(files[0])}")
    texts, seen = [], 0
    for f in files:
        tbl = pq.read_table(f, columns=["text"])
        for v in tbl.column("text").to_pylist():
            seen += 1
            t = (v or "").strip()
            # wikitext section headers look like " = = Title = = "; skip them
            if len(t) < min_chars or t.startswith("="):
                continue
            texts.append(t[:max_chars])
            if len(texts) >= n * 3:
                break
        if len(texts) >= n * 3:
            break
    import random
    random.Random(seed).shuffle(texts)
    log(f"prompts: {len(texts[:n])} selected from {seen} wikitext rows")
    return texts[:n]


# ---------------------------------------------------------------- model
def load_model(model_id, ngpu, cpu_first=False, max_mem_gib=230):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    t0 = time.time()
    if cpu_first:
        # transformers materializes shards in a ThreadPoolExecutor; a worker thread
        # whose current CUDA device differs from the target device can trip an
        # illegal memory access (seen with the MXFP4 dequant path on gpt-oss-120b).
        # Loading to CPU keeps CUDA out of materialization entirely, then we move
        # the finished model to one device in a single-threaded copy.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=None)
        log(f"materialized on CPU in {time.time()-t0:.0f}s; moving to cuda:0")
        model.to("cuda:0")
        device_map = "cpu->cuda:0"
    else:
        # Single visible GPU -> pin everything to it, so no cross-device dispatch.
        device_map = {"": 0} if ngpu == 1 else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map,
            max_memory=None if ngpu == 1 else {i: f"{max_mem_gib}GiB" for i in range(ngpu)},
        )
    model.eval()
    model.requires_grad_(False)          # activation grads only; never weight grads
    log(f"loaded in {time.time()-t0:.0f}s  device_map={device_map}")
    return model, tok


@torch.no_grad()
def coherence_gate(model, tok, text, max_nats):
    """Next-token cross-entropy on real text. Catches silently-broken loads."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
    ids = ids.to(next(model.parameters()).device)
    out = model(ids, labels=ids)
    ce = float(out.loss)
    log(f"coherence gate: next-token CE = {ce:.3f} nats (ppl {math.exp(min(ce,20)):.1f})")
    if not math.isfinite(ce) or ce > max_nats:
        raise RuntimeError(
            f"coherence gate FAILED: CE={ce:.3f} > {max_nats} nats. Model looks broken "
            f"(quantized-kernel/device mismatch?). Refusing to fit.")
    return ce


# ---------------------------------------------------------------- metrics
def eff_dim_table(J, device):
    """Participation ratio PR = ||J||_F^4 / ||J^T J||_F^2, plus sigma_1/stable rank."""
    rows = []
    for l in sorted(J):
        A = J[l].to(device, torch.float32)
        fro2 = float((A * A).sum())
        G = A.T @ A
        pr = fro2 ** 2 / float((G * G).sum())
        v = torch.randn(G.shape[0], device=device)
        v /= v.norm()
        for _ in range(60):
            v = G @ v
            v /= v.norm().clamp_min(1e-30)
        lam = float(v @ (G @ v))
        rows.append(dict(layer=l, fro_norm=fro2 ** 0.5, eff_dim_pr=pr,
                         sigma_1=lam ** 0.5, stable_rank=fro2 / lam))
        del A, G, v
        torch.cuda.empty_cache()
    mx = max(r["fro_norm"] for r in rows)
    n = len(rows)
    for r in rows:
        r["fro_rel_max"] = r["fro_norm"] / mx
        r["depth_frac"] = r["layer"] / max(n - 1, 1)
    return rows


def write_csv(rows, path):
    cols = ["layer", "depth_frac", "fro_norm", "fro_rel_max",
            "eff_dim_pr", "sigma_1", "stable_rank"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")


# ---------------------------------------------------------------- fit loop
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--outdir", default=os.environ.get("LENS_DIR", "lenses"),
                   help="where the lens, effdim table and meta are written "
                        "(default: $LENS_DIR, else ./lenses)")
    p.add_argument("--dim-batch", type=int, default=64)
    p.add_argument("--max-prompts", type=int, default=300)
    p.add_argument("--min-prompts", type=int, default=100)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--stop-delta", type=float, default=0.002)
    p.add_argument("--stop-window", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--max-nats", type=float, default=5.0)
    p.add_argument("--max-mem-gib", type=int, default=230,
                   help="per-GPU weight budget for device_map=auto; must exceed "
                        "total_weights/ngpu or accelerate silently offloads to CPU")
    p.add_argument("--cpu-first", action="store_true",
                   help="materialize on CPU then move to cuda:0 (avoids threaded "
                        "cross-device materialization; needed for gpt-oss MXFP4)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ck = os.path.join(args.outdir, f"{args.slug}_checkpoint.pt")
    ngpu = torch.cuda.device_count()
    log(f"model={args.model} slug={args.slug} gpus={ngpu} dim_batch={args.dim_batch}")

    prompts = load_prompts(args.max_prompts)
    model, tok = load_model(args.model, ngpu, cpu_first=args.cpu_first, max_mem_gib=args.max_mem_gib)
    ce = coherence_gate(model, tok, prompts[0], args.max_nats)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    L, d = tcfg.num_hidden_layers, tcfg.hidden_size
    layers = list(range(L - 1))                    # sources 0..L-2, target L-1
    log(f"L={L} d={d} -> fitting {len(layers)} source layers, target={L-1}")

    lens_model = jlens.from_hf(model, tok)
    dev = "cuda:0"

    # resume
    Jsum = {l: torch.zeros(d, d, dtype=torch.float32) for l in layers}
    n_done, times = 0, []
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        Jsum, n_done = st["jacobian_sum"], st["n_done"]
        log(f"resumed from checkpoint at n_done={n_done}")

    def save_ckpt():
        tmp = ck + ".tmp"
        torch.save({"jacobian_sum": Jsum, "n_done": n_done, "next_idx": n_done,
                    "source_layers": layers, "d_model": d}, tmp)
        os.replace(tmp, ck)

    def running_mean():
        return {l: Jsum[l] / max(n_done, 1) for l in layers}

    prev = None
    below = 0
    db = args.dim_batch
    t_start = time.time()
    while n_done < args.max_prompts:
        text = prompts[n_done % len(prompts)]
        t0 = time.time()
        try:
            Jp, seq_len, n_valid = fitting.jacobian_for_prompt(
                lens_model, text, source_layers=layers,
                dim_batch=db, max_seq_len=args.max_seq_len)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if db <= 4:
                raise
            db = max(4, db // 2)
            log(f"OOM -> retrying with dim_batch={db}")
            continue
        for l in layers:
            Jsum[l] += Jp[l]
        n_done += 1
        times.append(time.time() - t0)

        cur = running_mean()
        if prev is not None:
            rel = sum(float((cur[l] - prev[l]).norm() / prev[l].norm().clamp_min(1e-12))
                      for l in layers) / len(layers)
            below = below + 1 if rel < args.stop_delta else 0
        else:
            rel = float("nan")
        prev = cur
        if n_done % 5 == 0 or n_done <= 3:
            eta = (args.max_prompts - n_done) * (sum(times[-5:]) / len(times[-5:])) / 3600
            log(f"  n={n_done:4d} {times[-1]:6.1f}s seq={seq_len} valid={n_valid} "
                f"rel_change={rel:.5f} below={below}/{args.stop_window} eta<={eta:.1f}h")
        if n_done % args.checkpoint_every == 0:
            save_ckpt(); log(f"  checkpointed at n={n_done}")
        if n_done >= args.min_prompts and below >= args.stop_window:
            log(f"converged: rel_change < {args.stop_delta} for {below} consecutive prompts")
            break

    save_ckpt()
    J = running_mean()
    lens = JacobianLens(J, n_prompts=n_done, d_model=d)
    jpath = os.path.join(args.outdir, f"{args.slug}_jacobian_lens.pt")
    lens.save(jpath, dtype=torch.float16)

    rows = eff_dim_table(J, dev)
    write_csv(rows, os.path.join(args.outdir, f"{args.slug}_effdim.csv"))

    ident = sum(float((J[l] - torch.eye(d)).norm()) for l in layers) / len(layers) / d ** 0.5
    meta = dict(model=args.model, slug=args.slug, n_layers=L, d_model=d,
                source_layers=[layers[0], layers[-1]], target_layer=L - 1,
                n_prompts=n_done, dim_batch=db, max_seq_len=args.max_seq_len,
                coherence_ce_nats=ce, identity_distance=ident,
                final_rel_change=rel, converged=bool(below >= args.stop_window),
                n_gpus=ngpu, sec_per_prompt=sum(times) / max(len(times), 1),
                total_h=(time.time() - t_start) / 3600,
                eff_dim_first=rows[0]["eff_dim_pr"], eff_dim_last=rows[-1]["eff_dim_pr"])
    json.dump(meta, open(os.path.join(args.outdir, f"{args.slug}_meta.json"), "w"), indent=2)
    log(f"DONE {args.slug}: n_prompts={n_done} {meta['sec_per_prompt']:.1f}s/prompt "
        f"total={meta['total_h']:.2f}h -> {jpath}")
    log(f"  eff_dim layer {rows[0]['layer']}={rows[0]['eff_dim_pr']:.1f} "
        f"-> layer {rows[-1]['layer']}={rows[-1]['eff_dim_pr']:.1f}")


if __name__ == "__main__":
    main()
