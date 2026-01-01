"""Layer-by-layer linear CKA between the J_l of one fitted Jacobian lens.

Reads a lens written by fit_lens_run.py, or a mid-fit checkpoint. Same quantity as the
naive pairwise form, but computed via the Gram identity so that it scales to
d=12288 / L=88:

    ||X_j^T X_i||_F^2 = <X_i X_i^T, X_j X_j^T>     and     ||X_i^T X_i||_F = ||X_i X_i^T||_F

so linear CKA is exactly the cosine similarity between flattened, Frobenius-normalised
Gram matrices. That is L matmuls of d^3 instead of L^2/2 of them -- the pairwise form
took 46 min on a d=5376 model and ran out of memory on d=8192.

  cka_out : X = J    (samples = output dims -> compares J J^T structure)
  cka_in  : X = J^T  (samples = input dims  -> compares J^T J structure)
  cos     : plain Frobenius cosine <J_i,J_j>/(||J_i|| ||J_j||)

TF32 is enabled for the Gram matmuls: the lenses are stored fp16 (10-bit mantissa), so
TF32 inputs (also 10-bit) with fp32 accumulation lose nothing relative to the data on
disk. --verify checks that against the exact fp32 pairwise form.
"""
import argparse, json, os, time
import numpy as np
import torch

DEV = "cuda"
GRAM_BUDGET = 90e9      # bytes of GPU held by one block of Gram matrices


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_J(path):
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "J" in obj:
        return obj["J"], obj.get("n_prompts", -1), "final"
    if "jacobian_sum" in obj:      # mid-fit checkpoint: unnormalised running sum.
        # CKA and Frobenius cosine are invariant to a per-matrix positive scale.
        return obj["jacobian_sum"], obj.get("n_done", -1), "checkpoint"
    raise KeyError(f"no J or jacobian_sum in {path}: keys={list(obj)}")


def _grams(J, layers, idx, mode):
    """Flattened, unit-Frobenius-norm Gram matrices for the given layer indices."""
    d = J[layers[0]].shape[0]
    G = torch.empty(len(idx), d * d, device=DEV, dtype=torch.float32)
    for k, i in enumerate(idx):
        X = J[layers[i]].to(DEV, torch.float32)
        if mode == "cos":
            G[k] = (X / X.norm().clamp_min(1e-30)).reshape(-1)
        else:
            if mode == "in":
                X = X.T.contiguous()
            X -= X.mean(0, keepdim=True)          # centre each feature over samples
            g = X @ X.T
            G[k] = (g / torch.linalg.matrix_norm(g).clamp_min(1e-30)).reshape(-1)
            del g
        del X
    return G


def _dot(A, B, chunk=1 << 24):
    """<A_i, B_j> over the flattened axis, accumulated in fp64 across chunks.

    Kept off TF32: this is only L^2 d^2 flops (0.03 s even at d=12288), but running it
    in TF32 puts a 2^-11 = 4.9e-4 floor on the whole result. fp32 here costs nothing
    and buys two orders of magnitude.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        M = torch.zeros(A.shape[0], B.shape[0], dtype=torch.float64, device=DEV)
        for s in range(0, A.shape[1], chunk):
            M += (A[:, s:s + chunk] @ B[:, s:s + chunk].T).double()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return M


def sim_matrix(J, layers, mode):
    """mode in {'out','in','cos'} -> LxL similarity, computed blockwise over layers."""
    L, d = len(layers), J[layers[0]].shape[0]
    blk = max(1, min(L, int(GRAM_BUDGET / (d * d * 4))))
    M = torch.zeros(L, L, dtype=torch.float64)
    for a in range(0, L, blk):
        ia = list(range(a, min(a + blk, L)))
        Ga = _grams(J, layers, ia, mode)
        for b in range(a, L, blk):
            ib = list(range(b, min(b + blk, L)))
            Gb = Ga if b == a else _grams(J, layers, ib, mode)
            m = _dot(Ga, Gb).cpu()
            M[a:a + len(ia), b:b + len(ib)] = m
            M[b:b + len(ib), a:a + len(ia)] = m.T
            if b != a:
                del Gb
            torch.cuda.empty_cache()
        del Ga
        torch.cuda.empty_cache()
    lo = 0.0 if mode != "cos" else -1.0     # CKA is nonnegative; Frobenius cosine is signed
    return M.clamp(lo, 1).numpy().astype(np.float32)


def reference(J, layers, mode, pairs):
    """Exact fp32 pairwise form of the same quantity, for --verify."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    try:
        for i, j in pairs:
            V = []
            for l in (layers[i], layers[j]):
                X = J[l].to(DEV, torch.float32)
                if mode == "in":
                    X = X.T.contiguous()
                V.append(X if mode == "cos" else X - X.mean(0, keepdim=True))
            if mode == "cos":
                v = float((V[0] * V[1]).sum() / (V[0].norm() * V[1].norm()))
            else:
                num = float(torch.linalg.matrix_norm(V[1].T @ V[0]).double() ** 2)
                den = float(torch.linalg.matrix_norm(V[0].T @ V[0]).double()
                            * torch.linalg.matrix_norm(V[1].T @ V[1]).double())
                v = num / den
            out[(i, j)] = v
            del V
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def write_csv(S, layers, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("layer," + ",".join(str(int(l)) for l in layers) + "\n")
        for i, l in enumerate(layers):
            fh.write(str(int(l)) + "," + ",".join(f"{v:.6f}" for v in S[i]) + "\n")


def process(slug, lensdir, outdir, verify=0):
    npz = os.path.join(outdir, slug + ".npz")
    if os.path.exists(npz):
        log(slug, "cached")
        return
    t0 = time.time()
    J, npr, status = load_J(os.path.join(lensdir, f"{slug}_jacobian_lens.pt"))
    layers = sorted(J.keys())
    L, d = len(layers), J[layers[0]].shape[0]
    log(f"{slug}: L={L} d={d} n_prompts={npr} ({status}) loaded in {time.time()-t0:.0f}s")

    off = ~np.eye(L, dtype=bool)
    res = {}
    for mode, key in (("out", "cka_out"), ("in", "cka_in"), ("cos", "cos")):
        t1 = time.time()
        S = sim_matrix(J, layers, mode)
        res[key] = S
        log(f"  {key}: {time.time()-t1:.0f}s diag_err={np.abs(np.diag(S)-1).max():.2e} "
            f"offdiag[min,mean,max]=[{S[off].min():.3f},{S[off].mean():.3f},{S[off].max():.3f}]")
        if verify:
            rng = np.random.default_rng(0)
            pairs = [(int(i), int(j)) for i, j in rng.integers(0, L, size=(verify, 2)) if i != j]
            ref = reference(J, layers, mode, pairs)
            err = max(abs(ref[p] - float(S[p[0], p[1]])) for p in ref)
            log(f"  {key}: VERIFY max|fast-exact| = {err:.2e} over {len(ref)} pairs")

    np.savez_compressed(npz, layers=np.array(layers), d_model=int(d),
                        n_prompts=npr, lens_status=status, **res)
    for key in ("cka_out", "cka_in", "cos"):
        write_csv(res[key], layers, os.path.join(outdir, f"{slug}_{key}.csv"))
    log(f"{slug}: DONE total {time.time()-t0:.0f}s -> {npz}")
    del J, res
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lensdir", default=os.environ.get("LENS_DIR", "lenses"),
                   help="holds <slug>_jacobian_lens.pt (default: $LENS_DIR, else ./lenses)")
    p.add_argument("--outdir", default=os.environ.get("CKA_DIR", "cka"),
                   help="where the CKA tables are written (default: $CKA_DIR, else ./cka)")
    p.add_argument("--slugs", nargs="*")
    p.add_argument("--verify", type=int, default=0,
                   help="check N random pairs per metric against the exact fp32 form")
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.outdir, exist_ok=True)
    slugs = a.slugs or [f[:-len("_jacobian_lens.pt")]
                        for f in sorted(os.listdir(a.lensdir))
                        if f.endswith("_jacobian_lens.pt")]
    # smallest first, so a problem shows up cheaply
    slugs.sort(key=lambda s: os.path.getsize(os.path.join(a.lensdir, f"{s}_jacobian_lens.pt")))
    log(f"{len(slugs)} models: {', '.join(slugs)}")
    log(f"GPU: {torch.cuda.get_device_name(0)} "
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    for s in slugs:
        try:
            process(s, a.lensdir, a.outdir, a.verify)
        except Exception as e:
            log(f"FAILED {s}: {type(e).__name__}: {e}")
    log("ALL DONE")
