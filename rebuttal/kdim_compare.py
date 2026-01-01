#!/usr/bin/env python
"""Compare linmap results across centroid dimensionality (metric_kdim_cent.py + metric_fullpca_allscore.py --key):
per k, the picks (cut, cylinder frame by all-layer mean of the rival alone), and per layer the medians over splits
of sphere alone, chosen chart / cylinder alone and the four gains; summary: band means (band = layers where
sphere-alone median >= 0.5 * its max) and max |difference| vs k8 over layers and over all 79 choices.
Usage: kdim_compare.py <kdim results dir> <model>"""
import os
import sys

import numpy as np

from metric_gain_typed import NAMES
from metric_sensitivity import CUTS, FRAMES

d, m = sys.argv[1], sys.argv[2]
KEYS = ["k8", "k32", "k128", "kfull"]
R = {k: np.load(os.path.join(d, f"score_{k}", m + ".npz")) for k in KEYS if os.path.exists(os.path.join(d, f"score_{k}", m + ".npz"))}
med = lambda x: np.median(x, 0)
rows = {}
for k, A in R.items():
    SPH, AL, JO = A["sphere"], A["alone"], A["joined"]
    cs = {c: np.median(AL[:, :, NAMES.index(f"chart@equatorial/{c}")].mean(1)) for c in CUTS}
    fs = {f: np.median(AL[:, :, NAMES.index(f"cylinder@{f}")].mean(1)) for f in FRAMES}
    bc, bf = max(cs, key=cs.get), max(fs, key=fs.get)
    ic, iy = NAMES.index(f"chart@equatorial/{bc}"), NAMES.index(f"cylinder@{bf}")
    rows[k] = dict(cut=bc, frame=bf, sphere=med(SPH),
                   sph_to_chart=med(JO[:, :, ic] - AL[:, :, ic]), chart_to_sph=med(JO[:, :, ic] - SPH),
                   sph_to_cyl=med(JO[:, :, iy] - AL[:, :, iy]), cyl_to_sph=med(JO[:, :, iy] - SPH),
                   all_to=med(JO - AL), all_from=med(JO - SPH[:, :, None]))
band = rows["k8"]["sphere"] >= 0.5 * rows["k8"]["sphere"].max()
print(f"{m}: band = {band.sum()} of {len(band)} layers (k8 sphere-alone >= half its max)")
print(f"{'k':6s} {'cut':>4s} {'frame':>10s} {'sphere':>7s} {'S->chart':>8s} {'chart->S':>8s} {'S->cyl':>7s} {'cyl->S':>7s}"
      "   max|d| vs k8 over layers x 79 choices (to / from sphere)")
for k, r in rows.items():
    bm = lambda x: x[band].mean()
    line = f"{k:6s} {r['cut']:4d} {r['frame']:>10s} {bm(r['sphere']):7.3f} {bm(r['sph_to_chart']):8.3f} {bm(r['chart_to_sph']):8.3f} {bm(r['sph_to_cyl']):7.3f} {bm(r['cyl_to_sph']):7.3f}"
    if k != "k8":
        line += f"   {np.abs(r['all_to'] - rows['k8']['all_to']).max():.3f} / {np.abs(r['all_from'] - rows['k8']['all_from']).max():.3f}"
        line += f"  (band: {np.abs(r['all_to'] - rows['k8']['all_to'])[band].max():.3f} / {np.abs(r['all_from'] - rows['k8']['all_from'])[band].max():.3f})"
    print(line)
# sign agreement of the key comparisons in the band
for k, r in rows.items():
    print(f"{k:6s} band layers with S->chart > chart->S: {(r['sph_to_chart'] > r['chart_to_sph'])[band].sum()}/{band.sum()},"
          f" S->cyl > cyl->S: {(r['sph_to_cyl'] > r['cyl_to_sph'])[band].sum()}/{band.sum()}")
