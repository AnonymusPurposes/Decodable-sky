#!/usr/bin/env python
"""Two reviewer questions the paper does not need: harmonic R^2 of the top-16 PCA, and decoding of distance.

harmonics_r2.csv
    The top-16 PCA scores (averaged over prompts per object, the paper's PCA) regressed on the l=1
    harmonics alone (3) and on l=1 and l=2 together (3 + 5), with one intercept or with one per type.
    In-sample R^2, and held-out R^2 on the 100 object splits of split_pca.py (median, 5-95% of splits).
    R^2 pools the 16 PCs, so it is the share of top-16 variance explained. Layers: the Table 5 layer,
    and the mean over the band of layers with sky R^2 at least 80% of the maximum.

distance_decode.csv
    log10 distance of the 85 stars from the top-k PCA scores (object means, the location prompts).
    20 repeats of 5-fold cross-validation over stars, at the Table 5 layer and at the layer with the
    best cross-validated R^2 (optimistic, since it is selected on the same folds). Baseline: the same
    cross-validation from apparent magnitude and text frequency alone. The permutation p-value at the
    Table 5 layer shuffles the distances over stars (1000 permutations, k=8).

Usage: curiosity.py --pca-dir <d> --meta <astro_metadata_full.csv> --split-dir <d> --corr-dir <d> --band-dir <d> --out <d>
"""
import argparse
import csv
import glob
import os

import numpy as np

KS = (4, 8, 16)


def harmonics(Y):
    x, y, z = Y.T
    return Y.copy(), np.column_stack([x * y, y * z, z * x, x * x - y * y, 3 * z * z - 1])


def design(H, T):
    return np.column_stack([T, H]) if H is not None else T


def r2(pred, truth, base):
    """pooled over columns: 1 - SSE / SST, SST around the mean of `base`"""
    return float(1 - ((truth - pred) ** 2).sum() / ((truth - base.mean(0)) ** 2).sum())


def fit(X, Z):
    return np.linalg.lstsq(X, Z, rcond=None)[0]


def paper_layer(d, m):
    best, bv = None, -1e9
    for r in csv.DictReader(open(os.path.join(d, m + "_sky_r2_8.csv"), encoding="utf-8")):
        if float(r["r2_avg"]) > bv:
            bv, best = float(r["r2_avg"]), int(r["layer"])
    return best


def band(d, m):
    rs = list(csv.DictReader(open(os.path.join(d, m + "_sky_r2_16.csv"), encoding="utf-8")))
    mx = max(float(r["r2"]) for r in rs)
    return [int(r["layer"]) for r in rs if float(r["r2"]) >= 0.8 * mx]


def cv_r2(X, y, seeds):
    """repeated 5-fold CV of OLS with intercept -> mean over repeats of the pooled out-of-fold R^2"""
    n = len(y)
    out = []
    for s in seeds:
        p = np.random.default_rng(s).permutation(n)
        pred = np.zeros(n)
        for f in range(5):
            te = p[f::5]
            tr = np.setdiff1d(p, te)
            A = np.column_stack([np.ones(len(tr)), X[tr]])
            pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ np.linalg.lstsq(A, y[tr], rcond=None)[0]
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    for k in ("pca-dir", "meta", "split-dir", "corr-dir", "band-dir", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    meta = {r["name"]: r for r in csv.DictReader(open(a.meta, encoding="utf-8"))}
    hrows, drows = [], []
    for f in sorted(glob.glob(os.path.join(a.pca_dir, "*_pca128.npz"))):
        m = os.path.basename(f)[:-len("_pca128.npz")]
        sf = os.path.join(a.split_dir, m + ".npz")
        if not os.path.exists(sf):
            continue
        d = np.load(f, allow_pickle=True)
        names, types, Y = [str(x) for x in d["names"]], np.array([str(x) for x in d["types"]]), d["Yunit"].astype(float)
        obj, pca = d["obj_ids"], d["pca"]
        no = len(names)
        cnt = np.bincount(obj, minlength=no)[:, None]
        L0 = paper_layer(a.corr_dir, m)
        B = band(a.band_dir, m)
        tests = np.load(sf)["test_objects"]
        h1, h2 = harmonics(Y)
        one = np.ones((no, 1))
        typ = np.column_stack([(types == t).astype(float) for t in np.unique(types)])
        means = {}

        def obj_mean(L):
            if L not in means:
                e = np.zeros((no, 128))
                np.add.at(e, obj, pca[:, L, :].astype(float))
                means[L] = e / cnt
            return means[L]

        for ilab, T in (("one", one), ("typed", typ)):
            for hlab, H in (("none", None), ("l1", h1), ("l1_l2", np.column_stack([h1, h2]))):
                X = design(H, T)

                def scores(L):
                    Z = obj_mean(L)[:, :16]
                    ins = r2(X @ fit(X, Z), Z, Z)
                    held = []
                    for te in tests:
                        mask = np.zeros(no, bool); mask[te] = True
                        held.append(r2(X[mask] @ fit(X[~mask], Z[~mask]), Z[mask], Z[mask]))
                    return ins, np.array(held)
                ins0, h0 = scores(L0)
                bb = [scores(L) for L in B]
                hb = np.mean([b[1] for b in bb], 0)
                hrows.append(dict(model=m, intercept=ilab, harmonics=hlab, n_regressors=X.shape[1],
                                  layer=L0, insample=round(ins0, 3), heldout=round(float(np.median(h0)), 3),
                                  heldout_p05=round(float(np.percentile(h0, 5)), 3), heldout_p95=round(float(np.percentile(h0, 95)), 3),
                                  band_layers=len(B), band_insample=round(float(np.mean([b[0] for b in bb])), 3),
                                  band_heldout=round(float(np.median(hb)), 3)))
        print(m, [(r["intercept"], r["harmonics"], r["insample"], r["heldout"]) for r in hrows if r["model"] == m], flush=True)

        idx = [i for i, n in enumerate(names) if types[i] == "star" and meta.get(n, {}).get("distance_ly")]
        dist = np.array([float(meta[names[i]]["distance_ly"]) for i in idx])
        keep = dist > 0
        idx, y = np.array(idx)[keep], np.log10(dist[keep])
        ctl = np.array([[float(meta[names[i]]["Vmag"]), float(meta[names[i]]["zipf"])] for i in idx])
        seeds = range(20)
        base = cv_r2(ctl, y, seeds)
        for k in KS:
            vals = np.array([cv_r2(obj_mean(L)[idx, :k], y, range(5)) for L in range(pca.shape[1])])
            Lb = int(vals.argmax())
            row = dict(model=m, n_stars=len(y), k=k, layer=L0, cv_r2=round(cv_r2(obj_mean(L0)[idx, :k], y, seeds), 3),
                       best_layer=Lb, cv_r2_best_layer=round(cv_r2(obj_mean(Lb)[idx, :k], y, seeds), 3),
                       baseline_vmag_zipf=round(base, 3), cv_r2_with_baseline=round(cv_r2(np.column_stack([obj_mean(L0)[idx, :k], ctl]), y, seeds), 3))
            if k == 8:
                rng = np.random.default_rng(7)
                X = obj_mean(L0)[idx, :k]
                obs = cv_r2(X, y, range(2))
                null = [cv_r2(X, y[rng.permutation(len(y))], range(2)) for _ in range(1000)]
                row["perm_p"] = (1 + sum(n >= obs for n in null)) / 1001
            else:
                row["perm_p"] = ""
            drows.append(row)
        print(m, [(r["k"], r["cv_r2"], r["cv_r2_best_layer"], r["baseline_vmag_zipf"], r["perm_p"]) for r in drows if r["model"] == m], flush=True)
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("harmonics_r2.csv", hrows), ("distance_decode.csv", drows)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
