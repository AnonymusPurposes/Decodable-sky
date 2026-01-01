#!/usr/bin/env python
"""Held-out comparison of candidate sky metrics (sphere / cylinder / flat chart).

Appendix F.1 regresses the neural squared-distance matrix (RDM) on one RDM per coordinate and
reads the weights. The reviewer's objections: weights of strongly correlated predictors do not
show that a representation is absent, and the fit is in-sample. This script keeps the same RDM
and coordinates but scores candidates on objects the fit never saw.

For each random split of the non-constellation objects (the F.1 pool) into train/test:
  * standardization and top-8 PCA are fitted on the training objects' per-prompt vectors, and
    scaled so PC-1 has variance 1, as in F.1;
  * the metric of each candidate is fitted on train-train object pairs;
  * it is scored by R^2 on held-out pairs (each with at least one test object).

Candidates are sets of coordinate RDMs, each fitted three ways: OLS, non-negative least squares
(a mixture of squared-distance RDMs has non-negative weights), and ridge with the penalty chosen
by cross-validation over training objects.

    sphere    x y z            cylinder  x cos_ra sin_ra       chart  dec ra
    sphere+cylinder, sphere+chart, cylinder+chart, all

Incremental value, which is what collinearity leaves identified: the gain in held-out R^2 from
adding a map's own coordinates to a model that already has the others. If a representation were
present beyond what the sphere explains, adding its coordinates would improve held-out
prediction. Additivity of squared distances is still assumed, and the conclusion is limited to
these candidates.

Output <out>/<model>.npz (resumable): heldout_r2 (n_splits, n_layers, n_candidates, n_fitters).
Usage: metric_heldout.py --model <m> --raw-dir <d> --r2-table <csv> --out <d>
"""
import argparse
import csv
import os

import numpy as np
import torch
from scipy.optimize import nnls

from split_pca import pca_fit

SETS = {"sphere": ["x", "y", "z"], "cylinder": ["x", "cos_ra", "sin_ra"], "chart": ["dec", "ra"],
        "sphere+cylinder": ["x", "y", "z", "cos_ra", "sin_ra"],
        "sphere+chart": ["x", "y", "z", "dec", "ra"],
        "cylinder+chart": ["x", "cos_ra", "sin_ra", "dec", "ra"],
        "all": ["x", "y", "z", "cos_ra", "sin_ra", "dec", "ra"]}
FITTERS = ("ols", "nnls", "ridge")
RIDGE = np.logspace(-4, 3, 15)


def coords(rows):
    dec = np.radians([float(r["a_dec_deg"]) for r in rows])
    ra = np.radians([float(r["l_ra_deg"]) for r in rows])
    return {"x": np.sin(dec), "y": np.cos(dec) * np.sin(ra), "z": np.cos(dec) * np.cos(ra),
            "cos_ra": np.cos(ra), "sin_ra": np.sin(ra), "dec": dec, "ra": ra}


def fit(Xz, y, how, groups=None):
    """returns (intercept, beta) for standardized features Xz"""
    ym = y.mean()
    if how == "ols":
        b = np.linalg.lstsq(Xz, y - ym, rcond=None)[0]
    elif how == "nnls":
        b = nnls(np.column_stack([Xz, np.ones(len(Xz)), -np.ones(len(Xz))]), y)[0]
        return b[-2] - b[-1], b[:-2]
    else:
        G = Xz.T @ Xz
        fi, fj = groups                                       # object fold of each pair's two ends
        best, bl = np.inf, 1.0
        for lam in RIDGE:                                     # inner CV: leave one fold of objects out
            sse = 0.0
            for g in np.unique(fi):
                te = (fi == g) | (fj == g)
                tr = ~te
                if te.sum() < 2 or tr.sum() < 2:
                    continue
                m = y[tr].mean()
                b = np.linalg.solve(Xz[tr].T @ Xz[tr] + lam * np.eye(Xz.shape[1]) * tr.sum(), Xz[tr].T @ (y[tr] - m))
                sse += ((y[te] - m - Xz[te] @ b) ** 2).sum()
            if sse < best:
                best, bl = sse, lam
        b = np.linalg.solve(G + bl * np.eye(Xz.shape[1]) * len(y), Xz.T @ (y - ym))
    return ym, b


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--r2-table", help="<model>_sky_r2_16.csv: analyse layers with r2 >= 0.8 max")
    ap.add_argument("--layers", help="comma list, overrides --r2-table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    rows = list(csv.DictReader(open(a.objects, newline="", encoding="utf-8")))
    no = len(rows)
    keep = np.array([i for i, r in enumerate(rows) if r["type"] != "constellation"])   # as in F.1
    C = coords([rows[i] for i in keep])
    nk = len(keep)
    iu, ju = np.triu_indices(nk, 1)
    feats = {c: (C[c][iu] - C[c][ju]) ** 2 for c in C}

    raw = np.load(os.path.join(a.raw_dir, a.model + "_raw.npy"), mmap_mode="r")
    nl = raw.shape[1]
    np_ = raw.shape[0] // no
    if a.layers:
        layers = [int(x) for x in a.layers.split(",")]
    else:
        t = list(csv.DictReader(open(a.r2_table, encoding="utf-8")))
        mx = max(float(r["r2"]) for r in t)
        layers = [int(r["layer"]) for r in t if float(r["r2"]) >= 0.8 * mx and int(r["layer"]) < nl]
    print(f"[{a.model}] {len(layers)} layers {layers[0]}-{layers[-1]}, {nk} objects", flush=True)

    obj_np = np.tile(np.arange(no), np_)
    idx = np.where(np.isin(obj_np, keep))[0]
    remap = -np.ones(no, int); remap[keep] = np.arange(nk)
    obj = torch.as_tensor(remap[obj_np[idx]], device=dev)
    X = torch.from_numpy(np.array(raw[idx][:, layers])).to(dev)

    ntest = max(1, int(round(nk * a.test_frac)))
    rng = np.random.default_rng(a.seed)
    tests = np.stack([rng.permutation(nk)[:ntest] for _ in range(a.n_splits)])
    names = list(SETS)
    res = np.full((a.n_splits, len(layers), len(names), len(FITTERS)), np.nan)
    path = os.path.join(a.out, a.model + ".npz")
    done = 0
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if (d["tests"] == tests).all() and list(d["layers"]) == layers:
            done, res = int(d["done"]), d["heldout_r2"]
            print("resuming after", done, flush=True)

    def save(nd):
        np.savez(path + ".tmp.npz", done=np.int64(nd), tests=tests, layers=np.array(layers),
                 candidates=np.array(names), fitters=np.array(FITTERS), heldout_r2=res)
        os.replace(path + ".tmp.npz", path)

    for s in range(done, a.n_splits):
        is_te = np.zeros(nk, bool); is_te[tests[s]] = True
        te_mask = torch.as_tensor(is_te, device=dev)[obj]
        pair_tr = ~is_te[iu] & ~is_te[ju]
        pair_te = ~pair_tr
        fold = np.random.default_rng(a.seed + 1000 + s).integers(0, 4, nk)
        groups = (fold[iu][pair_tr], fold[ju][pair_tr])
        for li in range(len(layers)):
            x = X[:, li].to(torch.float64)
            ztr, (zte,) = pca_fit(x[~te_mask], [x[te_mask]], a.k)
            sc = ztr[:, 0].std(correction=0)
            ztr, zte = ztr / sc, zte / sc
            cent = torch.zeros(nk, a.k, dtype=torch.float64, device=dev)
            for g in range(nk):
                m = obj[~te_mask] == g if not is_te[g] else obj[te_mask] == g
                cent[g] = (ztr[m] if not is_te[g] else zte[m]).mean(0)
            cent = cent.cpu().numpy()
            y = ((cent[iu] - cent[ju]) ** 2).sum(1)
            ytr, yte = y[pair_tr], y[pair_te]
            for ci, n in enumerate(names):
                A = np.column_stack([feats[c] for c in SETS[n]])
                m, sd = A[pair_tr].mean(0), A[pair_tr].std(0) + 1e-12
                Z = (A - m) / sd
                for fi, how in enumerate(FITTERS):
                    b0, b = fit(Z[pair_tr], ytr, how, groups)
                    res[s, li, ci, fi] = r2(yte, b0 + Z[pair_te] @ b)
        if (s + 1) % a.save_every == 0 or s + 1 == a.n_splits:
            save(s + 1)
        print(f"split {s + 1}/{a.n_splits}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
