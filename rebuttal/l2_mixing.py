#!/usr/bin/env python
"""Predicted damage of the (l=2)-perp ablation to an l=2-blind read-out of the l=1 sky position.

Model of one layer's top-16 PCA scores: z = A y1 + B y2 + rest, with A (16x3) and B (16x5) the OLS maps of
the l=1 and l=2 real spherical harmonics (sphere-orthonormal), as in harmonic_alignment.py.
If span(A) and span(B) are not orthogonal, the orthogonal projection onto span(A) also carries P1 B y2. A
read-out W with W A = I and W B = 0 (kernel containing the l=2 span) still recovers y1 exactly; it cancels
P1 B y2 using the l=2 directions orthogonal to l=1, V = span((I - P1) B).
The (l=2)-perp ablation sets the V-components to their mean. P1 z is unchanged, but W now returns
  y1 + C (y2 - mean y2),   C = A^+ P1 B
because the part of the l=2 signal inside span(A) can no longer be cancelled. This script evaluates that
distortion on the 188 objects: the angle between the true direction u and the decoded direction
u + C (y2 - mean y2) / c (y1 = c u), median over objects, then median over the ablated layers
(sky R^2 on the top-16 PCA >= 80% of its maximum, as in the ablation).
||C||_F is reported too. Both are predictions for an idealised l=2-blind linear reader, not a test of
what the model's later layers actually do.

Usage: l2_mixing.py --indir <PCA128 dir> --r2dir <dir with <model>_sky_r2_16.csv>
                    --ablation results/ablation_intervals.csv --out results/l2_mixing.csv
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonic_alignment import K, MODELS, l1_basis, l2_basis, ols, orth, select_layers  # noqa: E402


def mixing(z, y1, y2, u):
    a = ols(y1, z)                                   # 16 x 3
    b = ols(y2, z)                                   # 16 x 5
    q1 = orth(a)
    c = np.linalg.pinv(a) @ (q1 @ (q1.T @ b))        # 3 x 5
    scale = np.sqrt(3.0 / (4.0 * np.pi))             # y1 = scale * u
    d = (y2 - y2.mean(0)) @ c.T / scale
    v = u + d
    cosang = (v * u).sum(1) / np.linalg.norm(v, axis=1)
    return float(np.linalg.norm(c)), float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1)))))


def spear(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def exact_p(a, b):
    s = spear(a, b)
    hits = sum(spear(a, np.array(p)) >= s - 1e-12 for p in itertools.permutations(b))
    return s, hits / np.prod(np.arange(1, len(a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--r2dir", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    a = ap.parse_args()
    abl = {r["model"]: r for r in csv.DictReader(open(a.ablation, encoding="utf-8")) if r["condition"] == "l2pure"}
    rows = []
    for m in MODELS:
        with np.load(os.path.join(a.indir, m + "_pca128.npz"), allow_pickle=True) as d:
            pca = d["pca"][:, :, :K].astype(np.float64)
            u = d["Yunit"].astype(np.float64)[d["obj_ids"].astype(int)]
            obj = d["obj_ids"].astype(int)
        layers, _ = select_layers(os.path.join(a.r2dir, m + "_sky_r2_16.csv"), a.frac)
        y1, y2 = l1_basis(u), l2_basis(u)
        first = np.unique(obj, return_index=True)[1]       # one row per object for the evaluation
        res = []
        for L in layers:
            z = pca[:, L, :]
            cn, _ = mixing(z, y1, y2, u)
            # distortion evaluated per object (harmonics are per object, so any row of the object will do)
            aa, bb = ols(y1, z), ols(y2, z)
            q1 = orth(aa)
            c = np.linalg.pinv(aa) @ (q1 @ (q1.T @ bb))
            uo, y2o = u[first], y2[first]
            v = uo + (y2o - y2o.mean(0)) @ c.T / np.sqrt(3.0 / (4.0 * np.pi))
            ang = np.degrees(np.arccos(np.clip((v * uo).sum(1) / np.linalg.norm(v, axis=1), -1, 1)))
            res.append((cn, float(np.median(ang))))
        res = np.array(res)
        rows.append(dict(model=m, n_layers=len(layers), C_fro_median=round(float(np.median(res[:, 0])), 4),
                         pred_angle_deg_median=round(float(np.median(res[:, 1])), 2),
                         l2perp_change_deg=float(abl[m]["change_from_none"])))
        print(rows[-1], flush=True)
    x = np.array([r["pred_angle_deg_median"] for r in rows])
    y = np.array([r["l2perp_change_deg"] for r in rows])
    names = [r["model"] for r in rows]
    for label, keep in (("all 8", [True] * len(rows)),
                        ("7, harmony gpt-oss only", [n != "gptoss120b" for n in names]),
                        ("6, no gpt-oss", ["gptoss" not in n for n in names])):
        k = np.array(keep)
        s, p = exact_p(x[k], y[k])
        print(f"{label}: Spearman {s:+.3f} (exact one-sided p={p:.4f}), Pearson {np.corrcoef(x[k], y[k])[0, 1]:+.3f}")
        rows.append(dict(model="spearman " + label, n_layers="", C_fro_median="", pred_angle_deg_median=round(s, 3),
                         l2perp_change_deg=round(p, 4)))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
