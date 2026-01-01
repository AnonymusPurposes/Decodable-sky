#!/usr/bin/env python
"""
Low-frequency (l<=2 spherical-harmonic) sky-subspace ablation on the location-prompt grid
(25 templates x 188 objects). Side-by-side variant of ablate_sky.py (l=1 only).

Layers: every hidden_states level L whose sky_r2 (top-16 PCA, per-prompt `r2`
column of <model>_sky_r2_16.csv) is >= FRAC * max (default FRAC = 0.8).

Per selected layer, from the extraction npz (last-token, standardized-PCA scores):
    z   = top-16 PCA scores (n_samples, 16),  s = (x, y, z) sky unit vector
    Y1  = l=1 real harmonics  (x, y, z)                                   (n, 3)
    Y2  = l=2 real harmonics  (xy, yz, xz, x^2 - y^2, 3z^2 - 1)           (n, 5)
    A8  = OLS fit of the harmonics into top-16 PCA space: z ~ [Y1,Y2] @ A8.T + c   (16x8)
    Q8  = orthonormal basis of span(A8)   -- the low-frequency sky subspace       (16, 8)
    Q3  = orthonormal basis of span(A3), A3 from z ~ Y1 alone (= ablate_sky.py `sky`);
          span(A3) is contained in span(A8)
    C8  = orthogonal complement of span(A8) inside the top-16 space               (16, 8)
    Q5p = the part of span(A8) orthogonal to span(A3) -- the PURE quadrupole      (16, 5)
    Q5r = orthonormal basis of span(A5), A5 from z ~ Y2 alone                     (16, 5)
    U   = Q.T @ components[L, :16]            (d, hidden), orthonormal rows
The column span of A8 is invariant to any invertible re-basing of the l<=2 functions, so the
harmonic normalization does not matter. Removing span(A8) zeroes the cross-covariance
between the remaining top-16 scores and every l=1 and l=2 function of the sky position.
span(A5) is NOT orthogonal to span(A3) (overlap 0.24-0.72 by layer/model), so removing it also
destroys part of the l=1 read-out: `l2pure` is the clean "l=2 only" condition and `l2raw` is
the literal-fit variant.

Conditions (projection applied at ALL token positions, prompt + generated):
    none     no hook (baseline)
    sky3     remove span(A3)   (l=1 only; reference, same as ablate_sky.py `sky`)
    harm8    remove span(A8)   (l=1 + l=2)
    comp8    remove C8         (control: the complementary 8 dims of the top-16 space)
    l2pure   remove Q5p        (l=2 only, orthogonal to the l=1 span)
    l2raw    remove Q5r        (l=2 fit span as-is; overlaps the l=1 span)

The projection is AFFINE, onto the subspace passing through the data centroid
(not the origin of hidden space). In the standardized coordinates the PCA was fit in,
the fit data's centroid is  c = feat_mean + feat_std * pca_mean  and
    x  = (h - feat_mean)/feat_std - pca_mean        # = (h - c)/feat_std, centroid -> 0
    h' = h - feat_std * ((x @ U.T) @ U)
so P(c) = c, P(P(h)) = P(h), and on the fit data the removed scores are set to their
mean. The preflight verifies these numerically on the model's own hidden states.

Hook site: hidden_states[L] is the output of decoder block L-1, except the last
level, which HF returns after the final norm -- the preflight checks each
selected layer against output_hidden_states and picks the matching module.

Output (<out>/):
    completions_<model>.csv   condition, pr_id, obj_id, object, type, prompt, completion
    diagnostics_<model>.json  layer set, per-layer subspace stats, preflight, timings
"""
import argparse
import csv
import json
import os
import re
import time

# Weights are read through the usual HuggingFace cache; set HF_HOME to relocate it.

import numpy as np

MODELS = {   # key -> (HF repo, prompt format)
    "qwen32b":            ("Qwen/Qwen3-32B", "raw"),
    "qwen235b":           ("Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", "raw"),
    "gptoss120b":         ("openai/gpt-oss-120b", "raw"),
    "gptoss120b_harmony": ("openai/gpt-oss-120b", "harmony"),
    "llama33_70b":        ("unsloth/Llama-3.3-70B-Instruct", "raw"),
    "mixtral8x22b":       ("mistralai/Mixtral-8x22B-Instruct-v0.1", "raw"),
    "mistrallarge123b":   ("mistralai/Mistral-Large-Instruct-2411", "raw"),
    "glm45air":           ("zai-org/GLM-4.5-Air", "raw"),
}
K = 16
# extract_pca_gptoss_harmony.py: empty analysis channel, then the phrase seeds `final`
HARMONY_SEED_MID = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
HARMONY_DATE = "2026-08-11"   # system-prompt date of the PCA128_location_fixed harmony extraction


def load_prompts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().strip('"')
            if s:
                out.append(s)
    return out


def select_layers(r2_csv, frac):
    with open(r2_csv, newline="") as f:
        rows = [(int(r["layer"]), float(r["r2"])) for r in csv.DictReader(f)]
    rmax = max(r for _, r in rows)
    return [L for L, r in rows if r >= frac * rmax], rmax


def ols(X, Y):
    """Least squares Y ~ X @ B.T + c; returns B (Y_dim, X_dim)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    B, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    return B.T


def r2_insample(z, s):
    B = ols(z, s)
    pred = (z - z.mean(0)) @ B.T + s.mean(0)
    return 1.0 - ((s - pred) ** 2).sum() / ((s - s.mean(0)) ** 2).sum()


def orth(M):
    q, _ = np.linalg.qr(M)
    return q


def harmonics_l2(s):
    """Real l=2 spherical harmonics (unnormalized) of unit vectors s (n,3) -> (n,5)."""
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return np.stack([x * y, y * z, x * z, x ** 2 - y ** 2, 3 * z ** 2 - 1], axis=1)


def orth_complement(q, k):
    """Orthonormal basis of the complement of span(q) in R^k."""
    u, _, _ = np.linalg.svd(q, full_matrices=True)
    return u[:, q.shape[1]:]


def build_subspaces(npz, layers, n_rand=None):
    """Per layer: {cond: Q (16,d)} plus diagnostics."""
    pca = npz["pca"]
    s = npz["Yunit"].astype(np.float64)[npz["obj_ids"].astype(int)]
    y1, y2 = s, harmonics_l2(s)
    y8 = np.concatenate([y1, y2], axis=1)
    subs, diag = {}, {}
    for L in layers:
        z = pca[:, L, :K].astype(np.float64)
        a3 = ols(y1, z)                              # 16x3
        a8 = ols(y8, z)                              # 16x8
        q3, q8 = orth(a3), orth(a8)
        c8 = orth_complement(q8, K)
        q5r = orth(ols(y2, z))                       # l=2 fit span (overlaps span(A3))
        u5, _, _ = np.linalg.svd(q8 - q3 @ (q3.T @ q8))   # harm8 with the l=1 span removed
        q5p = u5[:, :q8.shape[1] - q3.shape[1]]      # pure quadrupole directions (16x5)
        qs = {"sky3": q3, "harm8": q8, "comp8": c8, "l2pure": q5p, "l2raw": q5r}
        subs[L] = qs
        zc = z - z.mean(0)
        var_tot = zc.var(0).sum()
        sv = np.linalg.svd(a8 / np.linalg.norm(a8, axis=0), compute_uv=False)
        d = {"r2_l1_insample_top16": round(float(r2_insample(z, y1)), 4),
             "r2_l2_insample_top16": round(float(r2_insample(z, y2)), 4),
             "a8_colnorm_sv_min_over_max": round(float(sv[-1] / sv[0]), 4),
             "sky3_in_harm8": round(float(np.linalg.norm(q8.T @ q3) ** 2 / 3), 6),
             "score_mean_abs_max": float(np.abs(z.mean(0)).max())}
        for c, q in qs.items():
            zp = zc - (zc @ q) @ q.T
            d[c] = {"dim": q.shape[1],
                    "var_frac_top16_removed": round(float(((zc @ q) ** 2).sum() / len(z) / var_tot), 4),
                    "r2_l1_insample_after": round(float(r2_insample(zp, y1)), 4),
                    "r2_l2_insample_after": round(float(r2_insample(zp, y2)), 4),
                    "overlap_with_harm8": round(float(np.linalg.norm(q8.T @ q) ** 2 / q.shape[1]), 4),
                    "overlap_with_sky3": round(float(np.linalg.norm(q3.T @ q) ** 2 / q.shape[1]), 4)}
        diag[L] = d
    return subs, diag


class Projector:
    """Forward hook: affine projection through the data centroid (see module doc)."""

    def __init__(self, module, mu, sd, pm, comps):
        import torch
        self.torch = torch
        dev = next(module.parameters()).device
        self.mu = torch.tensor(mu, dtype=torch.float32, device=dev)
        self.sd = torch.tensor(sd, dtype=torch.float32, device=dev)
        self.pm = torch.tensor(pm, dtype=torch.float32, device=dev)
        self.comps = torch.tensor(comps, dtype=torch.float32, device=dev)   # (16, hidden)
        self.U = None
        self.stats = [0.0, 0]          # sum ||dh||/||h|| at last prompt token, n
        self.handle = module.register_forward_hook(self)

    def set(self, Q):
        self.U = None if Q is None else (self.torch.tensor(Q.T, dtype=self.torch.float32,
                                                            device=self.comps.device) @ self.comps)
        self.stats = [0.0, 0]

    def project(self, hf):
        x = (hf - self.mu) / self.sd - self.pm
        return hf - ((x @ self.U.T) @ self.U) * self.sd

    def __call__(self, mod, inp, out):
        if self.U is None:
            return out
        t = out[0] if isinstance(out, tuple) else out
        hf = t.float()
        new = self.project(hf)
        if t.shape[1] > 1:                                  # prefill: log last prompt token
            self.stats[0] += ((hf[:, -1] - new[:, -1]).norm(dim=-1)
                              / hf[:, -1].norm(dim=-1)).sum().item()
            self.stats[1] += hf.shape[0]
        new = new.to(t.dtype)
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen32b", choices=sorted(MODELS))
    ap.add_argument("--npz", required=True)
    ap.add_argument("--r2-csv", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--conditions", default="none,sky3,harm8,comp8,l2pure,l2raw")
    ap.add_argument("--device-map", default="single", choices=["single", "auto"],
                    help="single = whole model on cuda:0 (required for FP8); auto = shard bf16 "
                         "models over all visible GPUs")
    ap.add_argument("--keep-experts-impl", type=int, default=1,
                    help="1 = keep the MoE experts implementation during decoding (no batched_mm swap)")
    ap.add_argument("--min-score-corr", type=float, default=0.99,
                    help="preflight gate: corr(recomputed, stored) top-16 PCA scores")
    ap.add_argument("--batch", type=int, default=235)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N samples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    repo, fmt = MODELS[args.model]

    layers, rmax = select_layers(args.r2_csv, args.frac)
    print(f"[{args.model}] r2 max={rmax:.4f}; layers >= {args.frac}*max ({len(layers)}): {layers}",
          flush=True)

    with np.load(args.npz, allow_pickle=True) as f:   # each f[key] access re-decompresses
        npz = {k: f[k] for k in ("pca", "Yunit", "obj_ids", "pr_ids", "names", "types",
                                 "feat_mean", "feat_std", "pca_mean")}
        npz["components"] = f["components"][:, :K].copy()
    names, types = npz["names"].astype(str), npz["types"].astype(str)
    obj_ids, pr_ids = npz["obj_ids"].astype(int), npz["pr_ids"].astype(int)
    prompts = load_prompts(args.prompts)
    phrases = [prompts[p].replace("X", names[o]) for p, o in zip(pr_ids, obj_ids)]
    t0 = time.time()
    subs, sub_diag = build_subspaces(npz, layers, args.n_rand)
    t_fit = time.time() - t0
    for L in layers:
        print(f"  L{L}: " + json.dumps(sub_diag[L]), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if fmt == "harmony":
        prefix = tok.apply_chat_template([{"role": "user", "content": ""}], tokenize=False,
                                         add_generation_prompt=True, reasoning_effort="low")
        prefix = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {HARMONY_DATE}", prefix)
        texts = [prefix + HARMONY_SEED_MID + p for p in phrases]
        tok_kw = dict(add_special_tokens=False)
        print(f"harmony example: {texts[0]!r}", flush=True)
    else:
        texts, tok_kw = phrases, {}
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype="auto", device_map={"": 0} if args.device_map == "single" else "auto").eval()
    if args.keep_experts_impl:
        # transformers 5.x generate() swaps MoE 'grouped_mm' -> 'batched_mm' for decoding, which
        # copies the selected expert weights per token (Mixtral-8x22B @ batch 235: 176 GiB OOM).
        import contextlib
        model._optimize_model_for_decode = lambda: contextlib.nullcontext()
    t_load = time.time() - t0
    in_dev = model.get_input_embeddings().weight.device
    blocks = model.model.layers
    n_hs = len(blocks) + 1
    print(f"loaded {repo} in {t_load:.0f}s; {len(blocks)} blocks; device_map={args.device_map}",
          flush=True)

    # ---- preflight: hook site per layer + transform reproduces npz scores ----
    pf_idx = list(range(0, len(texts), max(1, len(texts) // 64)))[:64]
    cap = {}
    hooks = []
    for L in layers:
        if L - 1 < len(blocks):
            hooks.append(blocks[L - 1].register_forward_hook(
                lambda m, i, o, L=L: cap.__setitem__(("blk", L), (o[0] if isinstance(o, tuple) else o).detach())))
    hooks.append(model.model.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__(("norm",), (o[0] if isinstance(o, tuple) else o).detach())))
    with torch.no_grad():
        enc = tok([texts[i] for i in pf_idx], return_tensors="pt", padding=True, **tok_kw).to(in_dev)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
    for h in hooks:
        h.remove()
    assert len(hs) == n_hs == npz["pca"].shape[1], (len(hs), n_hs, npz["pca"].shape)
    sites, preflight = {}, {}
    for L in layers:
        ref = hs[L][:, -1].float()
        d_blk = ((cap[("blk", L)][:, -1].float().to(ref.device) - ref).abs().max().item()
                 if ("blk", L) in cap else float("inf"))
        d_norm = (cap[("norm",)][:, -1].float().to(ref.device) - ref).abs().max().item()
        sites[L] = "blk" if d_blk <= d_norm else "norm"
        mu, sd, pm = npz["feat_mean"][L], npz["feat_std"][L], npz["pca_mean"][L]
        comps = npz["components"][L]
        score = (((ref.cpu().numpy() - mu) / sd - pm) @ comps.T)
        stored = npz["pca"][pf_idx, L, :K].astype(np.float32)
        err = np.abs(score - stored).max() / (np.abs(stored).max() + 1e-9)
        corr = float(np.corrcoef(score.ravel(), stored.ravel())[0, 1])
        preflight[L] = dict(site=sites[L], maxdiff_blk=d_blk, maxdiff_norm=d_norm,
                            score_rel_maxerr=float(err), score_corr=corr)
        print(f"  preflight L{L}: site={sites[L]} |blk-hs|={d_blk:.3g} |norm-hs|={d_norm:.3g} "
              f"score corr={corr:.5f} rel maxerr={err:.3g}", flush=True)
        assert min(d_blk, d_norm) < 1e-2, f"no hook site reproduces hidden_states[{L}]"
        assert corr > args.min_score_corr, f"PCA transform does not reproduce npz scores at L{L}"

    proj = {}
    for L in layers:
        mod = blocks[L - 1] if sites[L] == "blk" else model.model.norm
        proj[L] = Projector(mod, npz["feat_mean"][L], npz["feat_std"][L],
                            npz["pca_mean"][L], npz["components"][L])

    # ---- centroid check: the projection is affine through the data centroid ----
    for L in layers:
        p = proj[L]
        h = hs[L][:, -1].float().to(p.mu.device)
        cen = p.mu + p.sd * p.pm                         # data centroid in raw hidden space
        chk = {}
        for cond in ("harm8", "comp8", "l2pure"):
            p.set(subs[L][cond])
            pc = p.project(cen[None])[0]
            ph = p.project(h)
            removed = (((ph - p.mu) / p.sd - p.pm) @ p.U.T)    # subspace scores after projection
            before = (((h - p.mu) / p.sd - p.pm) @ p.U.T)
            chk[cond] = dict(
                centroid_moved_rel=float((pc - cen).norm() / cen.norm()),
                idempotent_rel=float((p.project(ph) - ph).norm() / ph.norm()),
                subspace_score_after_rms=float(removed.pow(2).mean().sqrt()),
                subspace_score_before_rms=float(before.pow(2).mean().sqrt()))
        p.set(None)
        preflight[L]["centroid_check"] = chk
        for cond, s in chk.items():
            print(f"  centroid L{L} {cond}: |P(c)-c|/|c|={s['centroid_moved_rel']:.2e} "
                  f"idempotent={s['idempotent_rel']:.2e} subspace score rms "
                  f"{s['subspace_score_before_rms']:.2f} -> {s['subspace_score_after_rms']:.2e}", flush=True)
            assert s["centroid_moved_rel"] < 1e-4 and s["subspace_score_after_rms"] < 1e-2 * max(
                s["subspace_score_before_rms"], 1.0), f"projection is not affine through centroid at L{L}"
    del hs, cap

    order = list(range(len(texts)))
    if args.limit:
        order = order[:args.limit]
    out_csv = os.path.join(args.out, f"completions_{args.model}.csv")
    fout = open(out_csv, "w", newline="", encoding="utf-8")
    w = csv.writer(fout)
    w.writerow(["condition", "pr_id", "obj_id", "object", "type", "prompt", "completion"])
    timings, hook_stats = {}, {}
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        for L in layers:
            proj[L].set(None if cond == "none" else subs[L][cond])
        t0 = time.time()
        n_tok = 0
        ratio = {L: [0.0, 0] for L in layers}
        for s0 in range(0, len(order), args.batch):
            idx = order[s0:s0 + args.batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, **tok_kw).to(in_dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     num_beams=1, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            n_tok += gen.numel()
            for L in layers:
                ratio[L][0] += proj[L].stats[0]
                ratio[L][1] += proj[L].stats[1]
                proj[L].stats = [0.0, 0]
            for i, g in zip(idx, gen):
                txt = tok.decode(g, skip_special_tokens=True)
                w.writerow([cond, pr_ids[i], obj_ids[i], names[obj_ids[i]], types[obj_ids[i]],
                            phrases[i], txt])
            print(f"  [{cond}] {min(s0 + args.batch, len(order))}/{len(order)} "
                  f"{time.time() - t0:.0f}s", flush=True)
        fout.flush()
        dt = time.time() - t0
        timings[cond] = dict(seconds=round(dt, 1), samples=len(order), gen_tokens=int(n_tok))
        hook_stats[cond] = {L: round(r[0] / max(r[1], 1), 5) for L, r in ratio.items()}
        print(f"### {cond}: {len(order)} samples in {dt:.1f}s; mean ||dh||/||h|| (last prompt tok): "
              + ", ".join(f"L{L}={v:.3f}" for L, v in hook_stats[cond].items()), flush=True)
    fout.close()

    diag = dict(model=args.model, repo=repo, prompt_format=fmt, frac=args.frac, r2_max=rmax,
                layers=layers, k=K, batch=args.batch, max_new=args.max_new, n_samples=len(order),
                device_map=args.device_map, n_gpus=torch.cuda.device_count(),
                keep_experts_impl=args.keep_experts_impl, min_score_corr=args.min_score_corr,
                gpu=torch.cuda.get_device_name(0), load_seconds=round(t_load, 1),
                fit_seconds=round(t_fit, 1), total_seconds=round(time.time() - t_start, 1),
                timings=timings, dh_over_h_last_prompt_token=hook_stats, preflight=preflight,
                subspaces=sub_diag)
    with open(os.path.join(args.out, f"diagnostics_{args.model}.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print("ABLATE DONE", flush=True)


if __name__ == "__main__":
    main()
