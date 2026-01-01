#!/usr/bin/env python
"""Does the strict second-harmonic ablation effect follow the l=2 / l=1 subspace overlap?

Reviewer question on Figure 16. For each model: x = overlap of the l=2 map with the l=1 span
(||R||^2 of the unit-norm l=2 -> top-16 PCA map projected on the l=1 span, median over active
layers; 0.1875 for random subspaces), from rebuttal/harmonic_alignment.py (results/harmonic_alignment_l2_in_l1.csv), and y = the increase in
median angle when only the l=2 directions orthogonal to l=1 are ablated. Exact permutation p over
all orderings of the models, for Spearman's rank correlation.

Usage: l2_overlap.py <harmonic_alignment_l2_in_l1.csv> <ablation_intervals.csv> <out.csv>
"""
import csv
import itertools
import sys

import numpy as np
from scipy.stats import rankdata


def main():
    al, ab, out = sys.argv[1:4]
    A = {r["model"]: r for r in csv.DictReader(open(al, encoding="utf-8")) if r["basis"] == "sh_orthonormal"}
    B = {r["model"]: r for r in csv.DictReader(open(ab, encoding="utf-8")) if r["condition"] == "l2pure"}
    ms = sorted(set(A) & set(B))
    x = np.array([float(A[m]["fro_sq_median"]) for m in ms])
    y = np.array([float(B[m]["change_from_none"]) for m in ms])
    rows = [dict(model=m, overlap_l2_on_l1=float(A[m]["fro_sq_median"]), theta1_deg=float(A[m]["theta1_deg_median"]),
                 l2perp_change_deg=float(B[m]["change_from_none"]), ci95_lo=float(B[m]["change_ci95_lo"]),
                 ci95_hi=float(B[m]["change_ci95_hi"])) for m in ms]

    def spear(a, b):
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    def exact(a, b):
        s = spear(a, b)
        n = sum(spear(a, np.array(p)) >= s - 1e-12 for p in itertools.permutations(b))
        tot = np.prod(np.arange(1, len(a) + 1))
        return s, n / tot
    s, p = exact(x, y)
    nog = np.array(["gptoss" not in m for m in ms])
    s2, p2 = exact(x[nog], y[nog])
    print(f"n={len(ms)}: pearson {np.corrcoef(x, y)[0, 1]:+.3f}, spearman {s:+.3f}, one-sided exact p={p:.4f}")
    print(f"without gpt-oss (n={nog.sum()}): pearson {np.corrcoef(x[nog], y[nog])[0, 1]:+.3f}, spearman {s2:+.3f}, one-sided exact p={p2:.4f}")
    rows.append(dict(model="ALL spearman", overlap_l2_on_l1=round(s, 3), theta1_deg="", l2perp_change_deg=round(p, 4), ci95_lo="", ci95_hi=""))
    rows.append(dict(model="no-gpt-oss spearman", overlap_l2_on_l1=round(s2, 3), theta1_deg="", l2perp_change_deg=round(p2, 4), ci95_lo="", ci95_hi=""))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
