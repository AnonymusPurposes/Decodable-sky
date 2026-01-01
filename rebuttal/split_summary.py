#!/usr/bin/env python
"""Tables from split_pca.py and metric_heldout.py output.

split_summary.csv, one row per model and k:
  paper_layer            the layer in Table 5 (best prompt-averaged leave-one-out R^2, selected on
                         all objects)
  fixed_*                in-split PCA at that layer: median over splits, 5th/95th percentile of splits
  global_fixed_*         the same with the original PCA fitted on all objects, on the same splits
  nested_*               layer chosen inside the training objects, per split; test score at it
  nested_layer_*         the median and 5-95% range of the chosen layer
split_by_layer.csv       median test R^2 and angle per layer (in-split and global), for plots

metric_heldout_summary.csv, one row per model:
  median held-out R^2 per candidate (averaged over the active layers, NNLS fit), and the held-out
  incremental gain of adding each map's own coordinates, with the 5-95% range over splits and the
  share of splits with gain > 0.

Usage: split_summary.py --split-dir <d> --metric-dir <d> --corr-dir <d> --out <d>
"""
import argparse
import csv
import glob
import os

import numpy as np

K = (4, 8, 16)


def paper_layer(corr_dir, model):
    best, bv = None, -1e9
    for r in csv.DictReader(open(os.path.join(corr_dir, f"{model}_sky_r2_8.csv"), encoding="utf-8")):
        if float(r["r2_avg"]) > bv:
            bv, best = float(r["r2_avg"]), int(r["layer"])
    return best


def rng3(a, nd=3):
    return round(float(np.median(a)), nd), round(float(np.percentile(a, 5)), nd), round(float(np.percentile(a, 95)), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--metric-dir", required=True)
    ap.add_argument("--corr-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, by = [], []
    for f in sorted(glob.glob(os.path.join(a.split_dir, "*.npz"))):
        m = os.path.basename(f)[:-4]
        d = np.load(f)
        n = int(d["done"])
        if not n:
            continue
        L0 = paper_layer(a.corr_dir, m)
        ks = list(d["ks"])
        for j, k in enumerate(ks):
            ri, ai, rg, ag, inn = [d[x][:n, :, j] for x in ("r2_insplit", "ang_insplit", "r2_global", "ang_global", "inner_r2")]
            sel = inn.argmax(1)
            idx = np.arange(n)
            row = dict(model=m, k=int(k), n_splits=n, paper_layer=L0)
            for lab, arr in (("fixed_r2", ri[:, L0]), ("fixed_ang", ai[:, L0]), ("global_fixed_r2", rg[:, L0]),
                             ("global_fixed_ang", ag[:, L0]), ("nested_r2", ri[idx, sel]), ("nested_ang", ai[idx, sel])):
                row[lab], row[lab + "_p05"], row[lab + "_p95"] = rng3(arr)
            row["nested_layer_median"] = float(np.median(sel))
            row["nested_layer_p05"] = float(np.percentile(sel, 5))
            row["nested_layer_p95"] = float(np.percentile(sel, 95))
            rows.append(row)
            if k == 8:
                for L in range(ri.shape[1]):
                    by.append(dict(model=m, layer=L, r2_insplit=round(float(np.median(ri[:, L])), 4),
                                   ang_insplit=round(float(np.median(ai[:, L])), 3),
                                   r2_global=round(float(np.median(rg[:, L])), 4),
                                   ang_global=round(float(np.median(ag[:, L])), 3)))

    def w(path, rs):
        if not rs:
            return
        with open(path, "w", newline="") as fh:
            c = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); c.writeheader(); c.writerows(rs)
        print("wrote", path, len(rs))
    w(os.path.join(a.out, "split_summary.csv"), rows)
    w(os.path.join(a.out, "split_by_layer.csv"), by)

    mr = []
    for f in sorted(glob.glob(os.path.join(a.metric_dir, "*.npz"))):
        m = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        n = int(d["done"])
        if not n:
            continue
        names = [str(x) for x in d["candidates"]]
        fitters = [str(x) for x in d["fitters"]]
        R = d["heldout_r2"][:n].mean(1)                       # average over active layers, per split
        for fi, fn in enumerate(fitters):
            row = dict(model=m, fitter=fn, n_splits=n, n_layers=d["heldout_r2"].shape[1])
            for ci, c in enumerate(names):
                row[c], row[c + "_p05"], row[c + "_p95"] = rng3(R[:, ci, fi])

            def gain(lab, big, small):
                dl = R[:, names.index(big), fi] - R[:, names.index(small), fi]
                row[lab], row[lab + "_p05"], row[lab + "_p95"] = rng3(dl, 4)
                row[lab + "_frac_pos"] = round(float((dl > 0).mean()), 3)
            gain("add_y_z_to_cylinder", "sphere+cylinder", "cylinder")
            gain("add_cosra_sinra_to_sphere", "sphere+cylinder", "sphere")
            gain("add_y_z_to_chart", "sphere+chart", "chart")
            gain("add_dec_ra_to_sphere", "sphere+chart", "sphere")
            gain("sphere_minus_cylinder", "sphere", "cylinder")
            gain("sphere_minus_chart", "sphere", "chart")
            mr.append(row)
    w(os.path.join(a.out, "metric_heldout_summary.csv"), mr)


if __name__ == "__main__":
    main()
