#!/usr/bin/env python
"""
Build projection tables from Mistral-Large residual-stream PCA (PCA128).

top-4 / ortho-xyz  (layers 54..75): the unique unit direction in the top-4 PCA
    subspace whose projection has ZERO Pearson correlation with the object xyz
    sky unit-vector (Yunit, not ra/dec). Rows = objects, cols = prompts.

top-2 / ortho-type (layers 79..88): the unique unit direction in the top-2 PCA
    subspace whose projection has ZERO correlation with the "type" function
    (star=+1, constellation=-1, other=0). Rows = prompts, cols = objects.

The direction is the null vector of M^T where M = Zc^T Tc (Zc = centered PCA
scores, Tc = centered target(s)); zero correlation is exact. w is unit-norm,
sign fixed so its largest-|weight| coordinate is positive.
"""
import os
import csv
import numpy as np

NPZ = "PCA128/mistrallarge123b_pca128.npz"
EXAMPLE = "mistrallarge123b_L85_PC2.csv"     # source of prompt-text headers
OUTDIR = "ortho_projections"
XYZ_LAYERS = range(54, 76)                    # top-4, orthogonal to xyz
TYPE_LAYERS = range(79, 89)                   # top-2, orthogonal to type


def zero_corr_direction(z, T):
    """Unit vector w s.t. (z@w) has zero correlation with every column of T."""
    Zc = z - z.mean(0)
    Tc = T - T.mean(0)
    M = Zc.T @ Tc                     # (k x m)
    _, _, Vt = np.linalg.svd(M.T, full_matrices=True)   # M^T is (m x k)
    w = Vt[-1]                        # right singular vec, smallest sing. value = null space
    w = w / np.linalg.norm(w)
    if w[np.argmax(np.abs(w))] < 0:   # deterministic sign
        w = -w
    return w


def main():
    d = np.load(NPZ, allow_pickle=True)
    pca = d["pca"].astype(np.float64)          # (4700, 89, 128)
    obj = d["obj_ids"].astype(int)             # (4700,)
    pr = d["pr_ids"].astype(int)               # (4700,)
    Y = d["Yunit"].astype(np.float64)          # (188, 3) sky unit xyz
    names = d["names"].astype(str)             # (188,)
    types = d["types"].astype(str)             # (188,)
    n_obj, n_pr, n_layers = len(names), int(pr.max() + 1), pca.shape[1]

    # prompt-text headers, identical to the example file (p00..p24)
    ex_hdr = next(csv.reader(open(EXAMPLE, encoding="utf-8")))
    prompt_headers = ex_hdr[2:]                 # 25 prompt strings
    assert len(prompt_headers) == n_pr

    T_xyz = Y[obj]                              # (4700, 3) per-sample xyz
    tmap = {"star": 1.0, "constellation": -1.0, "other": 0.0}
    T_type = np.array([tmap[t] for t in types])[obj].reshape(-1, 1)   # (4700, 1)

    os.makedirs(OUTDIR, exist_ok=True)

    def to_grid(vals):
        g = np.full((n_obj, n_pr), np.nan)
        g[obj, pr] = vals
        return g

    # ---- top-4, orthogonal to xyz ; rows=objects, cols=prompts ----
    for L in XYZ_LAYERS:
        z = pca[:, L, :4]
        w = zero_corr_direction(z, T_xyz)
        g = to_grid(z @ w)
        out = os.path.join(OUTDIR, f"mistrallarge123b_L{L}_top4_orthoxyz.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["object", "type"] + prompt_headers)
            for i in range(n_obj):
                wtr.writerow([names[i], types[i]] + [round(float(v), 6) for v in g[i]])
        c = [np.corrcoef((z @ w), T_xyz[:, j])[0, 1] for j in range(3)]
        print(f"L{L:02d} top4 ortho-xyz  -> {os.path.basename(out)}  max|corr|={max(abs(v) for v in c):.1e}")

    # ---- top-2, orthogonal to type ; rows=prompts, cols=objects ----
    for L in TYPE_LAYERS:
        z = pca[:, L, :2]
        w = zero_corr_direction(z, T_type)
        g = to_grid(z @ w)                       # (obj, prompt)
        out = os.path.join(OUTDIR, f"mistrallarge123b_L{L}_top2_orthotype.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["object"] + list(names))
            wtr.writerow(["type"] + list(types))
            for j in range(n_pr):
                wtr.writerow([prompt_headers[j]] + [round(float(g[i, j]), 6) for i in range(n_obj)])
        c = np.corrcoef((z @ w), T_type[:, 0])[0, 1]
        print(f"L{L:02d} top2 ortho-type -> {os.path.basename(out)}  |corr|={abs(c):.1e}")

    print(f"\n{len(XYZ_LAYERS)} + {len(TYPE_LAYERS)} files written to {OUTDIR}/  (n_layers={n_layers})")


if __name__ == "__main__":
    main()
