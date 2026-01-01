#!/usr/bin/env python
"""Control for metric_sensitivity.py: do the cylinder / chart coordinates still add held-out signal
once the sphere is allowed its l=2 harmonics?

A cut longitude (sawtooth) or an unshrunk longitude circle carries l>=2 content that the sphere's three
linear coordinates cannot express, so a gain over the bare sphere may be smooth non-spherical structure
rather than a flat or cylindrical map. Here every candidate is scored joined with
    sphere            x, y, z                                   (as in metric_sensitivity.py)
    sphere+Y2         x, y, z and the five real l=2 harmonics   xy, xz, yz, y^2-z^2, 3x^2-1
    sphere+Y2+Y3      and the ten cubic monomials of (x, y, z) (span l=3 and l=1)
with the linmap fitter (ridge linear map to the 8 PCs, i.e. a PSD metric with cross terms; the psd fitter
agreed with it in metric_sensitivity.py and is ~20x slower here), same splits, same neural side.

Output <out>/<model>.npz: r2 (splits, layers, candidates, base 0=sphere 1=sphere+Y2 2=sphere+Y2+Y3), plus the two bases;
<out>/metric_l2_control_summary.csv by --summarize.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh

from metric_sensitivity import candidates, score

FITTERS = ("linmap",)
BASES = ("sphere", "sphere+Y2", "sphere+Y2+Y3")


def run(job):
    model, indir, objects, heldout_dir, out, max_layers = job
    path = os.path.join(out, model + ".npz")
    if os.path.exists(path):
        return model
    H = np.load(os.path.join(heldout_dir, model + ".npz"), allow_pickle=True)
    tests, layers = H["tests"], [int(x) for x in H["layers"]]
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
    cand = candidates(np.array([float(rows[i]["l_ra_deg"]) for i in keep]), np.array([float(rows[i]["a_dec_deg"]) for i in keep]))
    sph = cand["sphere@equatorial"]
    x, y, z = sph.T
    y2 = np.column_stack([x * y, x * z, y * z, y * y - z * z, 3 * x * x - 1])
    mono3 = np.column_stack([a * b * c for i, a in enumerate((x, y, z)) for j, b in enumerate((x, y, z)) if j >= i
                             for k, c in enumerate((x, y, z)) if k >= j])     # 10 cubic monomials: span l=3 and l=1
    bases = (sph, np.column_stack([sph, y2]), np.column_stack([sph, y2, mono3]))
    cnames = [c for c in cand if c.startswith(("cylinder@", "chart@")) or c in ("dec", "ra/120")]
    iu, ju = np.triu_indices(nk, 1)
    res = np.full((len(tests), len(layers), len(cnames), 3, len(FITTERS)), np.nan)
    base = np.full((len(tests), len(layers), 3, len(FITTERS)), np.nan)
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
            for fi, how in enumerate(FITTERS):
                for bi, B in enumerate(bases):
                    base[s, li, bi, fi] = score(B, cent, tr_obj, iu, ju, pair_tr, fold, how)
                    for ci, nm in enumerate(cnames):
                        res[s, li, ci, bi, fi] = score(np.column_stack([B, cand[nm]]), cent, tr_obj, iu, ju, pair_tr, fold, how)
        if (s + 1) % 10 == 0:
            print(f"[{model}] split {s + 1}/{len(tests)}", flush=True)
    np.savez(path, r2=res, base=base, names=np.array(cnames), fitters=np.array(FITTERS), layers=np.array(layers))
    return model


def summarize(out, models):
    rows = []
    for m in models:
        d = np.load(os.path.join(out, m + ".npz"), allow_pickle=True)
        R, B, names = d["r2"].mean(1), d["base"].mean(1), [str(x) for x in d["names"]]   # mean over layers
        for fi, how in enumerate(FITTERS):
            for lo, hi, nm in ((0, 1, "Y2 (l=2 harmonics)"), (1, 2, "Y3 (l=3 harmonics)")):
                g2 = B[:, hi, fi] - B[:, lo, fi]
                rows.append(dict(model=m, fitter=how, candidate=nm, base=BASES[lo],
                                 gain=np.median(g2), p05=np.percentile(g2, 5), p95=np.percentile(g2, 95), frac_pos=(g2 > 0).mean(),
                                 base_r2=np.median(B[:, lo, fi])))
            for ci, nm in enumerate(names):
                for bi, bn in enumerate(BASES):
                    g = R[:, ci, bi, fi] - B[:, bi, fi]
                    rows.append(dict(model=m, fitter=how, candidate=nm, base=bn, gain=np.median(g), p05=np.percentile(g, 5),
                                     p95=np.percentile(g, 95), frac_pos=(g > 0).mean(), base_r2=np.median(B[:, bi, fi])))
    with open(os.path.join(out, "metric_l2_control_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    models = a.models or ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b", "mixtral8x22b", "mistrallarge123b", "glm45air"]
    os.makedirs(a.out, exist_ok=True)
    if not a.summarize:
        with Pool(a.procs) as p:
            for m in p.imap_unordered(run, [(m, a.indir, a.objects, a.heldout, a.out, a.max_layers) for m in models]):
                print("done", m, flush=True)
    summarize(a.out, models)


if __name__ == "__main__":
    main()
