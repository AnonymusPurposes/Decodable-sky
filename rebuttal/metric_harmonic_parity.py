#!/usr/bin/env python
"""Degrees-of-freedom parity for metric_l2_control.py: there only the sphere got its second-order terms.
Here every map gets its own:
    sphere      x, y, z                         +Y2:  xy, xz, yz, y^2-z^2, 3x^2-1           (3 -> 8)
    cylinder@F  u, cos l, sin l   (u = sin lat) +H2:  u^2, cos 2l, sin 2l                    (3 -> 6)
                                                +H2full: and u cos l, u sin l             (3 -> 8, = sphere+Y2)
    chart@F/c   b, t   (lat, cut longitude)    +Q2:  b^2, t^2                              (2 -> 4)
                                                +Q2full: and b t                           (2 -> 5)
                                                +Q3full: all monomials of degree <= 3      (2 -> 9, > sphere+Y2)
F = equatorial, ecliptic, galactic; c = 0, 30, ..., 330; t is centred (t - pi) before taking powers.
Each block is scored alone, joined with the plain sphere, and joined with sphere+Y2 (linmap fitter, as
metric_sensitivity.py; same splits and layers as metric_l2_control.py).

Output <out>/<model>.npz: r2 (splits, layers, blocks, 3: alone / +sphere / +sphere+Y2), names;
<out>/metric_harmonic_parity_summary.csv by --summarize.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh

from metric_sensitivity import CUTS, FRAMES, frames, score

JOIN = ("alone", "+sphere", "+sphere+Y2")


def blocks(ra, dec):
    fr = frames(ra, dec)
    out = {}
    lat, lon = fr["equatorial"]
    x, y, z = np.sin(lat), np.cos(lat) * np.sin(lon), np.cos(lat) * np.cos(lon)
    sph = np.column_stack([x, y, z])
    sphY2 = np.column_stack([sph, x * y, x * z, y * z, y * y - z * z, 3 * x * x - 1])
    out["sphere"], out["sphere+Y2"] = sph, sphY2
    for f in FRAMES:
        lat, lon = fr[f]
        u, c1, s1 = np.sin(lat), np.cos(lon), np.sin(lon)
        cyl = np.column_stack([u, c1, s1])
        h2 = np.column_stack([u * u, np.cos(2 * lon), np.sin(2 * lon)])
        out[f"cylinder@{f}"] = cyl
        out[f"cylinder@{f}+H2"] = np.column_stack([cyl, h2])
        out[f"cylinder@{f}+H2full"] = np.column_stack([cyl, h2, u * c1, u * s1])
        for c in CUTS:
            b, t = lat, np.mod(lon - np.radians(c), 2 * np.pi) - np.pi
            out[f"chart@{f}/{c}"] = np.column_stack([b, t])
            out[f"chart@{f}/{c}+Q2"] = np.column_stack([b, t, b * b, t * t])
            out[f"chart@{f}/{c}+Q2full"] = np.column_stack([b, t, b * b, t * t, b * t])
            out[f"chart@{f}/{c}+Q3full"] = np.column_stack([b, t, b * b, t * t, b * t, b ** 3, t ** 3, b * b * t, b * t * t])
    return out, sph, sphY2


def run(job):
    model, indir, objects, heldout_dir, out, max_layers = job
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    path = os.path.join(out, model + ".npz")
    if os.path.exists(path):
        return model
    H = np.load(os.path.join(heldout_dir, model + ".npz"), allow_pickle=True)
    tests, layers = H["tests"], [int(v) for v in H["layers"]]
    lsel = np.unique(np.round(np.linspace(0, len(layers) - 1, min(len(layers), max_layers))).astype(int))
    layers = [layers[i] for i in lsel]
    d = np.load(os.path.join(indir, model + "_pca128.npz"), allow_pickle=True)
    rows = list(csv.DictReader(open(objects, newline="", encoding="utf-8")))
    assert [r["name"] for r in rows] == list(d["names"].astype(str))
    keep = np.array([i for i, r in enumerate(rows) if r["type"] != "constellation"])
    nk = len(keep)
    remap = -np.ones(len(rows), int); remap[keep] = np.arange(nk)
    sel = np.isin(d["obj_ids"], keep)
    obj = remap[d["obj_ids"][sel]]
    X = d["pca"][sel][:, layers].astype(np.float32)
    B, sph, sphY2 = blocks(np.array([float(rows[i]["l_ra_deg"]) for i in keep]),
                           np.array([float(rows[i]["a_dec_deg"]) for i in keep]))
    names = list(B)
    iu, ju = np.triu_indices(nk, 1)
    res = np.full((len(tests), len(layers), len(names), 3), np.nan)
    for s, te in enumerate(tests):
        tr_obj = np.ones(nk, bool); tr_obj[te] = False
        pair_tr = tr_obj[iu] & tr_obj[ju]
        fold = np.random.default_rng(1000 + s).integers(0, 4, nk)
        rmask = tr_obj[obj]
        for li in range(len(layers)):
            Z = X[:, li].astype(np.float64)
            mu = Z[rmask].mean(0)
            _, V = eigh((Z[rmask] - mu).T @ (Z[rmask] - mu) / rmask.sum())
            P = (Z - mu) @ V[:, ::-1][:, :8]
            P /= P[rmask, 0].std()
            cent = np.zeros((nk, 8)); np.add.at(cent, obj, P)
            cent /= np.bincount(obj, minlength=nk)[:, None]
            for ci, nm in enumerate(names):
                C = B[nm]
                res[s, li, ci, 0] = score(C, cent, tr_obj, iu, ju, pair_tr, fold, "linmap")
                if not nm.startswith("sphere"):
                    res[s, li, ci, 1] = score(np.column_stack([sph, C]), cent, tr_obj, iu, ju, pair_tr, fold, "linmap")
                    res[s, li, ci, 2] = score(np.column_stack([sphY2, C]), cent, tr_obj, iu, ju, pair_tr, fold, "linmap")
        if (s + 1) % 10 == 0:
            print(f"[{model}] split {s + 1}/{len(tests)}", flush=True)
    np.savez(path, r2=res, names=np.array(names), layers=np.array(layers))
    return model


def summarize(out, models):
    """per split: mean over layers; then median and 5/95% over splits. For charts, also best and median over
    the 36 frame x cut variants (best chosen per split, so selection-inflated)."""
    rows = []
    for m in models:
        d = np.load(os.path.join(out, m + ".npz"), allow_pickle=True)
        R, names = d["r2"].mean(1), [str(v) for v in d["names"]]          # (split, block, join)
        ix = {n: i for i, n in enumerate(names)}
        sph, sphY2 = R[:, ix["sphere"], 0], R[:, ix["sphere+Y2"], 0]

        def add(label, alone, j1, j2):
            def st(v):
                return np.median(v), np.percentile(v, 5), np.percentile(v, 95), (v > 0).mean()
            r = dict(model=m, block=label, alone=np.median(alone))
            if j1 is not None:
                for k, v in (("adds_to_sphere", j1 - sph), ("adds_to_sphereY2", j2 - sphY2),
                             ("sphere_adds", j1 - alone), ("sphereY2_adds", j2 - alone)):
                    r[k], r[k + "_p05"], r[k + "_p95"], r[k + "_frac_pos"] = st(v)
            rows.append({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})

        add("sphere", sph, None, None)
        add("sphere+Y2", sphY2, None, None)
        for f in FRAMES:
            for ext in ("", "+H2", "+H2full"):
                i = ix[f"cylinder@{f}{ext}"]
                add(f"cylinder@{f}{ext}", R[:, i, 0], R[:, i, 1], R[:, i, 2])
        for ext in ("", "+Q2", "+Q2full", "+Q3full"):
            idx = [ix[f"chart@{f}/{c}{ext}"] for f in FRAMES for c in CUTS]
            A = R[:, idx]                                                   # (split, 36, join)
            add(f"chart{ext} median of 36", np.median(A[:, :, 0], 1), np.median(A[:, :, 1], 1), np.median(A[:, :, 2], 1))
            # best variant per split by alone R^2; its joined scores taken for the same variant
            b = A[:, :, 0].argmax(1); s = np.arange(len(A))
            add(f"chart{ext} best-alone of 36", A[s, b, 0], A[s, b, 1], A[s, b, 2])
            add(f"chart@equatorial/0{ext} (paper)", *(R[:, ix[f"chart@equatorial/0{ext}"], k] for k in range(3)))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "metric_harmonic_parity_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    models = a.models or ["mistrallarge123b", "qwen32b", "gptoss120b"]
    os.makedirs(a.out, exist_ok=True)
    if not a.summarize:
        with Pool(a.procs) as p:
            for m in p.imap_unordered(run, [(m, a.indir, a.objects, a.heldout, a.out, a.max_layers) for m in models]):
                print("done", m, flush=True)
    summarize(a.out, models)


if __name__ == "__main__":
    main()
