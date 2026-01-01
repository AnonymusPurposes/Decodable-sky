#!/usr/bin/env python
"""
How much of the l=2 -> PCA map lies inside the l=1 subspace?

For every model and every "active" layer (per-prompt sky_r2 on the top-16 PCA >= FRAC of that
model's max, default FRAC = 0.8), in the top-16 PCA space of the location-prompt run:

    A1 = OLS fit of the l=1 harmonics into top-16 PCA space   (16 x 3)
    Q1 = orthonormal basis of span(A1);  P1 = Q1 Q1^T          (the l=1 subspace projector)
    M  = OLS fit of the l=2 harmonics into top-16 PCA space   (16 x 5)  -- the map whose
         alignment we are measuring
    Mhat = M / ||M||_F                                         (Frobenius norm 1)
    R  = P1 @ Mhat                                             (l=2 map, kept only where l=1 lives)

recorded per layer: ||R||_F  and the squared singular values of R, sorted descending. R has rank
<= 3, so three of them are non-zero and they sum to ||R||_F^2: each says how much of the
(unit-norm) l=2 map's energy lands along one direction of the l=1 subspace.

Also recorded: the three PRINCIPAL ANGLES between span(A1) (3 dims) and span(M) (5 dims),
theta_i = arccos(sigma_i) with sigma the singular values of Q1^T Q2, sorted ascending
(theta_1 is the smallest = the most nearly shared direction). These describe the SUBSPACES
only: unlike the Frobenius numbers they ignore how much of M's energy sits where, and they are
invariant to any invertible re-basing of the harmonics, so they are identical in both bases
below. Reference for a random 3-dim and 5-dim subspace of R^16 (Monte-Carlo, --ref-draws):
median 41.5 / 58.3 / 73.7 degrees.

Reference values for an l=2 map with no preference for the l=1 subspace (M isotropic in the
16-dim space): E||R||_F^2 = 3/16 = 0.1875, i.e. ||R||_F ~ 0.43, with each squared singular value
~ 0.0625.

Basis of the l=2 space (the result depends on it, because rescaling M to unit Frobenius norm is
not invariant to rescaling the five harmonics separately). Two conventions are reported:
  sh_orthonormal  real spherical harmonics normalised on the sphere (the canonical choice:
                  rotation-equivariant, independent of the object sample)
  sample_whitened the same space whitened over the 188 objects (Cov = I), which makes the
                  numbers invariant to ANY basis choice but ties them to this object set

Usage:
    python harmonic_alignment.py --indir PCA128 --r2dir correlations --out harmonic_alignment.csv
Writes the per-model medians to --out and the per-layer values to <out stem>_per_layer.csv.
"""
import argparse
import csv
import os

import numpy as np

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "gptoss120b_harmony", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]
K = 16
BASES = ["sh_orthonormal", "sample_whitened"]


def l1_basis(u):
    """Real l=1 spherical harmonics (orthonormal on the sphere) of unit vectors u (n,3)."""
    c = np.sqrt(3.0 / (4.0 * np.pi))
    return c * u


def l2_basis(u):
    """Real l=2 spherical harmonics (orthonormal on the sphere) of unit vectors u (n,3)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    a = 0.5 * np.sqrt(15.0 / np.pi)
    return np.stack([a * x * y,                              # Y(2,-2)
                     a * y * z,                              # Y(2,-1)
                     0.25 * np.sqrt(5.0 / np.pi) * (3 * z ** 2 - 1),   # Y(2,0)
                     a * x * z,                              # Y(2,1)
                     0.25 * np.sqrt(15.0 / np.pi) * (x ** 2 - y ** 2)], axis=1)  # Y(2,2)


def whiten(f):
    """Center and whiten columns so Cov = I (makes the map basis-independent)."""
    fc = f - f.mean(0)
    cov = (fc.T @ fc) / len(fc)
    w, v = np.linalg.eigh(cov)
    return fc @ (v @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ v.T)


def ols(X, Y):
    """Least squares Y ~ X @ B.T + c; returns B (Y_dim, X_dim)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    B, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    return B.T


def orth(M):
    q, _ = np.linalg.qr(M)
    return q


def select_layers(path, frac):
    with open(path, newline="") as f:
        rows = [(int(r["layer"]), float(r["r2"])) for r in csv.DictReader(f)]
    rmax = max(r for _, r in rows)
    return [L for L, r in rows if r >= frac * rmax], rmax


def principal_angles(qa, qb):
    """Principal angles (degrees, ascending) between two orthonormal bases."""
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0))).tolist()


def alignment(z, y1, y2):
    """(||P1 Mhat||_F, [sv^2 desc], [principal angles asc]) for one layer."""
    q1 = orth(ols(y1, z))                 # 16 x 3, the l=1 subspace
    m = ols(y2, z)                        # 16 x 5, the l=2 map
    mhat = m / np.linalg.norm(m)          # Frobenius norm 1
    r = q1 @ (q1.T @ mhat)                # projected onto the l=1 subspace
    sv = np.linalg.svd(r, compute_uv=False)
    return (float(np.linalg.norm(r)), (sv[:3] ** 2).tolist(),
            principal_angles(q1, orth(m)))


def random_angle_reference(k, d1, d2, draws, seed=0):
    """Median principal angles between random d1- and d2-dim subspaces of R^k."""
    rng = np.random.default_rng(seed)
    out = [principal_angles(orth(rng.standard_normal((k, d1))),
                            orth(rng.standard_normal((k, d2)))) for _ in range(draws)]
    return np.median(np.array(out), axis=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="PCA128", help="folder with <model>_pca128.npz")
    ap.add_argument("--r2dir", default="correlations", help="folder with <model>_sky_r2_16.csv")
    ap.add_argument("--frac", type=float, default=0.8, help="active layers: r2 >= frac * max")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default="harmonic_alignment.csv")
    ap.add_argument("--ref-draws", type=int, default=2000,
                    help="Monte-Carlo draws for the random-subspace angle reference")
    args = ap.parse_args()
    ref = random_angle_reference(K, 3, 5, args.ref_draws)
    print(f"random 3-dim vs 5-dim subspace of R^{K}: median principal angles "
          + " / ".join(f"{a:.1f}" for a in ref) + " deg\n")

    per_layer, summary = [], []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        npz_path = os.path.join(args.indir, f"{model}_pca128.npz")
        r2_path = os.path.join(args.r2dir, f"{model}_sky_r2_16.csv")
        if not (os.path.exists(npz_path) and os.path.exists(r2_path)):
            print(f"skip {model}: missing {npz_path if not os.path.exists(npz_path) else r2_path}")
            continue
        with np.load(npz_path, allow_pickle=True) as d:
            pca = d["pca"][:, :, :K].astype(np.float64)
            u = d["Yunit"].astype(np.float64)[d["obj_ids"].astype(int)]
        layers, rmax = select_layers(r2_path, args.frac)
        feats = {"sh_orthonormal": (l1_basis(u), l2_basis(u)),
                 "sample_whitened": (whiten(l1_basis(u)), whiten(l2_basis(u)))}
        for basis in BASES:
            y1, y2 = feats[basis]
            rows = []
            for L in layers:
                fro, sv2, ang = alignment(pca[:, L, :], y1, y2)
                rows.append((fro, sv2, ang))
                per_layer.append(dict(model=model, basis=basis, layer=L,
                                      fro=round(fro, 6), fro_sq=round(fro ** 2, 6),
                                      **{f"sv{i + 1}_sq": round(v, 6) for i, v in enumerate(sv2)},
                                      **{f"theta{i + 1}_deg": round(a, 4) for i, a in enumerate(ang)}))
            med = lambda vals: round(float(np.median(vals)), 4)
            summary.append(dict(
                model=model, basis=basis, n_layers=len(layers),
                layers=f"{layers[0]}-{layers[-1]}" if layers == list(range(layers[0], layers[-1] + 1))
                else ",".join(map(str, layers)),
                r2_max=round(rmax, 4), frac=args.frac,
                fro_median=med([r[0] for r in rows]),
                fro_sq_median=med([r[0] ** 2 for r in rows]),
                **{f"sv{i + 1}_sq_median": med([r[1][i] for r in rows]) for i in range(3)},
                **{f"theta{i + 1}_deg_median": med([r[2][i] for r in rows]) for i in range(3)},
                fro_sq_random_ref=round(3 / K, 4), sv_sq_random_ref=round(1 / K, 4),
                **{f"theta{i + 1}_deg_random_ref": round(ref[i], 2) for i in range(3)}))
        print(f"{model:19s} {basis_line(summary[-2])} | {basis_line(summary[-1])}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    stem = args.out[:-4] if args.out.endswith(".csv") else args.out
    with open(stem + "_per_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader()
        w.writerows(per_layer)
    print(f"\nsaved {args.out} ({len(summary)} rows) and {stem}_per_layer.csv ({len(per_layer)} rows)")


def basis_line(s):
    return (f"{s['basis'][:4]}: |R|_F={s['fro_median']:.3f} "
            f"sv2=({s['sv1_sq_median']:.3f},{s['sv2_sq_median']:.3f},{s['sv3_sq_median']:.3f}) "
            f"theta=({s['theta1_deg_median']:.0f},{s['theta2_deg_median']:.0f},"
            f"{s['theta3_deg_median']:.0f})deg")


if __name__ == "__main__":
    main()
