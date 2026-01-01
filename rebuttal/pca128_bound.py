#!/usr/bin/env python
"""How much can the common 128-PCA basis (fitted per layer on all 4700 prompt rows, standardized features)
restrict the per-split top-8 PCA? Per layer:

  var_in_128   fraction of the total (standardized) variance inside the 128 PCs (total = hidden size)
  ratio        lambda_8 of the training rows of split 0 (type-centred, as in metric_gain_typed.py) divided by
               an upper bound on the variance of ANY direction outside the 128 PCs on those rows:
               (n_all / n_train) * lambda_128(all rows). (A direction u orthogonal to the 128 PCs has variance
               <= lambda_129 <= lambda_128 on all rows; restricting to a subset of rows and centring per type can
               only lower the sum of squares, so per training row it is at most n_all / n_train times that.)
               ratio > 1 means no direction outside the basis can carry more variance than the 8th PC used.

Output <out>/pca128_bound.csv: model, layer, band (1 = active-band layer), var_in_128, ratio.
"""
import argparse
import csv
import os

import numpy as np

MODELS = ["mistrallarge123b", "glm45air", "qwen235b", "qwen32b", "llama33_70b", "mixtral8x22b", "gptoss120b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--typed", required=True, help="metric_gain_typed.py output (splits)")
    ap.add_argument("--gains", required=True, help="metric_gain_curves.py output (band layers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for m in MODELS:
        d = np.load(os.path.join(a.indir, f"{m}_pca128.npz"), allow_pickle=True)
        X, obj, types, D = d["pca"].astype(np.float64), d["obj_ids"], d["types"].astype(str), d["components"].shape[2]
        band = {int(r["layer"]) for r in csv.DictReader(open(os.path.join(a.gains, f"{m}_linmap_chart.csv"))) if r["band"] == "1"}
        te = np.load(os.path.join(a.typed, m + ".npz"))["tests"][0]
        tr = np.ones(len(types), bool); tr[te] = False
        rm, rt = tr[obj], types[obj]
        for L in range(X.shape[1]):
            Z = X[:, L]
            lam = Z.var(0)                                   # the scores are the global PCs: their variances
            Zt = Z[rm].copy()
            for t in np.unique(types):
                Zt[rt[rm] == t] -= Zt[rt[rm] == t].mean(0)
            l8 = np.sort(np.linalg.eigvalsh(np.cov(Zt.T, bias=True)))[::-1][7]
            rows.append(dict(model=m, layer=L, band=int(L in band), var_in_128=round(float(lam.sum() / D), 4),
                             ratio=round(float(l8 / (lam[-1] * len(Z) / rm.sum())), 3)))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pca128_bound.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
