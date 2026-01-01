#!/usr/bin/env python
"""Per-layer tables from split_pca.py output, for redrawing the layer charts (Figures 2, 3, 12).

One file per model and k: <out>/<model>_split_r2_<k>.csv with, per layer, the median over the
80/20 splits and the 5th/95th percentile of splits for
  r2, ang          in-split PCA (standardization and PCA fitted on the training objects only)
  r2_g, ang_g      the paper's PCA fitted on all objects, scored on the same splits
ang is the median angular error over the test objects of a split, in degrees.

Usage: split_layer_csv.py --split-dir <d> --out <d>
"""
import argparse
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.split_dir, "*.npz"))):
        m = os.path.basename(f)[:-4]
        d = np.load(f)
        n = int(d["done"])
        for j, k in enumerate(d["ks"]):
            cols = {lab: d[src][:n, :, j] for lab, src in
                    (("r2", "r2_insplit"), ("ang", "ang_insplit"), ("r2_g", "r2_global"), ("ang_g", "ang_global"))}
            rows = []
            for L in range(cols["r2"].shape[1]):
                row = {"layer": L}
                for lab, arr in cols.items():
                    v = arr[:, L]
                    row[lab] = round(float(np.median(v)), 4)
                    row[lab + "_p05"] = round(float(np.percentile(v, 5)), 4)
                    row[lab + "_p95"] = round(float(np.percentile(v, 95)), 4)
                rows.append(row)
            path = os.path.join(a.out, f"{m}_split_r2_{int(k)}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("wrote", path, n, "splits")


if __name__ == "__main__":
    main()
