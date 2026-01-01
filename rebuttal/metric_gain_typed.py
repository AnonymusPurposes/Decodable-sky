#!/usr/bin/env python
"""metric_gain_curves.py --all-layers with the constellations added and the neural rows centred per type, and
one chart cut / cylinder frame per model chosen on the average over all layers.

Objects: all 188 (85 stars, 15 other, 88 constellations as points at their area centroids,
data_astro_objects.csv). Splits: 100, each holding out 20% of every type (17 / 3 / 18 = 38 test objects),
rng seed 7000 + split. Neural side per split and layer: the rows of each type are centred on the mean of that
type's TRAINING rows, then the top-8 PCA is fitted on the training rows (scaled so PC1 has SD 1), centroids are
averaged over templates; target = squared distance of centroids. Fitter: linmap (ridge coordinates -> 8 PCs,
lambda by 4-fold object CV on training objects, distances calibrated on training pairs), scored by R^2 on pairs
with at least one test object (metric_sensitivity.score).

Every layer: the sphere (equatorial x, y, z) alone; each chart@F/c (3 frames x 12 cuts) and cylinder@F alone and
joined with the sphere. Gains: sphere_to_X = R2(sphere + X) - R2(X), X_to_sphere = R2(sphere + X) - R2(sphere).

Choice, one per model (--summarize):
  chart cut     the equatorial cut with the highest median over splits of the all-layer mean R2(chart alone)
  cylinder      the frame with the highest median over splits of the all-layer mean R2(cylinder alone)
Output <out>/<model>.npz; <out>/<model>_linmap_<chart|cylinder>.csv (as metric_gain_curves.py, without nulls:
per layer median and 5/95% over splits of the gains at the chosen cut / frame, and the min / max over all
coordinate choices of the median gain); <out>/linmap_cut_choice.csv, <out>/linmap_frame_choice.csv.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh

from metric_sensitivity import CUTS, FRAMES, candidates, score

CHARTS = [f"chart@{f}/{c}" for f in FRAMES for c in CUTS]
CYLS = [f"cylinder@{f}" for f in FRAMES]
NAMES = CHARTS + CYLS
MODELS = ["qwen235b", "mistrallarge123b", "llama33_70b", "qwen32b", "mixtral8x22b", "glm45air", "gptoss120b"]


def load(model, indir, objects):
    d = np.load(os.path.join(indir, model + "_pca128.npz"), allow_pickle=True)
    rows = list(csv.DictReader(open(objects, newline="", encoding="utf-8")))
    assert [r["name"] for r in rows] == list(d["names"].astype(str))
    types = np.array([r["type"] for r in rows])
    cand = candidates(np.array([float(r["l_ra_deg"]) for r in rows]), np.array([float(r["a_dec_deg"]) for r in rows]))
    return d["pca"], d["obj_ids"], types, cand


def splits(types, n, frac=0.2):
    out = []
    for s in range(n):
        rng = np.random.default_rng(7000 + s)
        out.append(np.sort(np.concatenate([rng.choice(np.where(types == t)[0], int(round(frac * (types == t).sum())), replace=False)
                                           for t in np.unique(types)])))
    return out


def task(job):
    model, indir, objects, s0, s1, nsplits = job
    X, obj, types, cand = load(model, indir, objects)
    nk, nl = len(types), X.shape[1]
    sph = cand["sphere@equatorial"]
    iu, ju = np.triu_indices(nk, 1)
    rtype = types[obj]
    SPH = np.full((s1 - s0, nl), np.nan); AL = np.full((s1 - s0, nl, len(NAMES)), np.nan); JO = AL.copy()
    for k, te in enumerate(splits(types, nsplits)[s0:s1]):
        s = s0 + k
        tr_obj = np.ones(nk, bool); tr_obj[te] = False
        pair_tr = tr_obj[iu] & tr_obj[ju]
        fold = np.random.default_rng(1000 + s).integers(0, 4, nk)
        rmask = tr_obj[obj]
        for L in range(nl):
            Z = X[:, L].astype(np.float64)
            for t in np.unique(types):                            # centre each type on its training rows
                Z[rtype == t] -= Z[(rtype == t) & rmask].mean(0)
            mu = Z[rmask].mean(0)
            _, V = eigh((Z[rmask] - mu).T @ (Z[rmask] - mu) / rmask.sum())
            P = (Z - mu) @ V[:, ::-1][:, :8]
            P /= P[rmask, 0].std()
            cent = np.zeros((nk, 8)); np.add.at(cent, obj, P)
            cent /= np.bincount(obj, minlength=nk)[:, None]

            def sc(C):
                return score(C, cent, tr_obj, iu, ju, pair_tr, fold, "linmap")
            SPH[k, L] = sc(sph)
            for j, n in enumerate(NAMES):
                AL[k, L, j] = sc(cand[n])
                JO[k, L, j] = sc(np.column_stack([sph, cand[n]]))
    return model, s0, SPH, AL, JO


def q(v):
    return [round(float(np.median(v)), 5), round(float(np.percentile(v, 5)), 5), round(float(np.percentile(v, 95)), 5)]


def summarize(out, models):
    cut_rows, fr_rows = [], []
    for m in models:
        A = np.load(os.path.join(out, m + ".npz"))
        SPH, AL, JO = A["sphere"], A["alone"], A["joined"]
        allmean = lambda n: np.median(AL[:, :, NAMES.index(n)].mean(1))
        cs = {c: allmean(f"chart@equatorial/{c}") for c in CUTS}
        fs = {f: allmean(f"cylinder@{f}") for f in FRAMES}
        bc, bf = max(cs, key=cs.get), max(fs, key=fs.get)
        cut_rows += [dict(model=m, cut=c, chart_alone_allmean=round(float(v), 5), picked=int(c == bc)) for c, v in cs.items()]
        fr_rows += [dict(model=m, frame=f, cylinder_alone_allmean=round(float(v), 5), picked=int(f == bf)) for f, v in fs.items()]
        for kind, n, choice in (("chart", f"chart@equatorial/{bc}", bc), ("cylinder", f"cylinder@{bf}", bf)):
            env = [i for i, x in enumerate(NAMES) if x.startswith(kind)]
            i = NAMES.index(n)
            with open(os.path.join(out, f"{m}_linmap_{kind}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                keys = (f"sphere_to_{kind}", f"{kind}_to_sphere")
                w.writerow(["layer", "choice", f"{kind}_alone", "sphere_alone"] + [f"{k}{x}" for k in keys for x in ("", "_p05", "_p95")] +
                           [f"{k}{x}" for k in keys for x in ("_envmin", "_envmax")])
                for L in range(SPH.shape[1]):
                    al, jo, sp = AL[:, L, i], JO[:, L, i], SPH[:, L]
                    toX = np.median(JO[:, L, env] - AL[:, L, env], 0)
                    Xto = np.median(JO[:, L, env] - sp[:, None], 0)
                    w.writerow([L, choice, round(float(np.median(al)), 5), round(float(np.median(sp)), 5)] + q(jo - al) + q(jo - sp) +
                               [round(float(v), 5) for v in (toX.min(), toX.max(), Xto.min(), Xto.max())])
    for name, rows in (("linmap_cut_choice.csv", cut_rows), ("linmap_frame_choice.csv", fr_rows)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--procs", type=int, default=18)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.makedirs(a.out, exist_ok=True)
    models = a.models or MODELS
    if not a.summarize:
        todo = [m for m in models if not os.path.exists(os.path.join(a.out, m + ".npz"))]
        jobs = [(m, a.indir, a.objects, s, min(s + a.chunk, a.splits), a.splits) for m in todo for s in range(0, a.splits, a.chunk)]
        parts = {m: [] for m in todo}
        with Pool(a.procs) as p:
            for m, s0, S, A, J in p.imap_unordered(task, jobs):
                parts[m].append((s0, S, A, J))
                if len(parts[m]) == len(range(0, a.splits, a.chunk)):
                    parts[m].sort(key=lambda t: t[0])
                    np.savez(os.path.join(a.out, m + ".npz"), sphere=np.concatenate([t[1] for t in parts[m]]),
                             alone=np.concatenate([t[2] for t in parts[m]]), joined=np.concatenate([t[3] for t in parts[m]]),
                             names=np.array(NAMES), tests=np.array(splits(load(m, a.indir, a.objects)[2], a.splits)))
                    parts[m] = None
                    print("done", m, flush=True)
    summarize(a.out, models)


if __name__ == "__main__":
    main()
