#!/usr/bin/env python
"""Saturation of Q_text with corpus size (Issue 3), on the footing of Figure 10.

For every nested stage of the corpus (pilot web prefixes web_f1_16 ... web_f1_1, then the pilot web plus a
growing prefix of the extended FineWeb parts, ext_f1_16 ... ext_all; Wikipedia alone as a separate corpus),
and for each Q_text construction, two numbers:

  fit          Pearson r between the Q_text RDM and the neural RDM on constellation pairs, mean over the six
               strong models; the neural RDM is the paper's (spectral/fit_models.py: band of layers with
               R^2 >= 80% of max, unit-normalized, averaged), so the last stage equals Figure 10's column.
               95% interval: constellations resampled with replacement (pairs of a constellation with its own
               copy dropped), same resamples for every stage.
  split_half   Pearson r between the Q_text RDMs of the two halves of the stage (even / odd count files,
               merge_stage_halves.py): how reproducible the corpus RDM is at half that size. Spearman-Brown
               2r/(1+r) estimates the reliability at the full stage size.

Constructions (gated counts, positive eigenvalues, all modes unless the name says top10):
  faithful     M* as in appendix F.3 (corpus-wide baseline), zero diagonal: the published Q_text
  within       baseline from the object-object table's own margins, observed diagonal: Figure 10's 975B column

Book sets alone, every eigenvalue treatment:
  --stages books_paper4:books,books_sky_guide:books,... --all-factors

Usage: qtext_saturation.py --spectral <spectral dir> --sets <dir> [...] --halves <dir> [...]
                           --forms surface_forms_v2.csv --objects astro_objects.csv --out <csv>
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtext_spectrum as Q  # noqa: E402

STAGES = [("wiki", "wikipedia"), ("web_f1_16", "nested"), ("web_f1_8", "nested"), ("web_f1_4", "nested"),
          ("web_f1_2", "nested"), ("web_f1_1", "nested"), ("ext_f1_16", "nested"), ("ext_f1_8", "nested"),
          ("ext_f1_4", "nested"), ("ext_f1_2", "nested"), ("ext_all", "nested"), ("ext_everything", "everything")]
CONS = {"faithful": ("mstar", "zero", "positive", None), "within": ("mstar_tt", "observed", "positive", None),
        "faithful_top10": ("mstar", "zero", "positive", 10), "within_top10": ("mstar_tt", "observed", "positive", 10)}
# --all-factors adds the other eigenvalue treatments: |Lambda| (abs) and keeping signs (sgn), all modes
MORE = {"faithful_abs": ("mstar", "zero", "abs", None), "faithful_sgn": ("mstar", "zero", "signed", None),
        "within_abs": ("mstar_tt", "observed", "abs", None), "within_sgn": ("mstar_tt", "observed", "signed", None)}
STRONG6 = Q.STRONG6


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--halves", nargs="+", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--stages", default=None, help="comma list name[:series], default the corpus stages above")
    ap.add_argument("--all-factors", action="store_true")
    a = ap.parse_args()
    stages = STAGES if not a.stages else [tuple((x + ":" + "custom").split(":")[:2]) for x in a.stages.split(",")]
    cons = dict(CONS, **(MORE if a.all_factors else {}))
    sys.path.insert(0, a.spectral)
    import pandas as pd
    import fit_models as F
    import paths

    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist()
    fobj = {}
    for r in csv.DictReader(open(a.forms, encoding="utf-8")):
        fobj.setdefault(r["form"], r["object"])
    R = F.setup("const", cat, np.load(os.path.join(a.spectral, "graphs", "graph_constellation.npz")),
                np.load(os.path.join(a.spectral, "graphs", "graph_cooccurrence.npz")))
    nodes, ii, jj = R["nodes"], R["ii"], R["jj"]
    neural = {}
    for m in STRONG6:
        layers, _ = F.band(m)
        pca, obj, cnt = F.load_pca(m, names)
        neural[m] = F.neural_rdm(pca, obj, cnt, layers, nodes, R["T"], ii, jj)
    sphere = R["sphere"]
    n = len(nodes)
    pid = np.full((n, n), -1)
    pid[ii, jj] = pid[jj, ii] = np.arange(len(ii))
    rng = np.random.default_rng(20261115)
    boots = []
    for _ in range(a.nboot):
        b = rng.integers(0, n, n)
        s = pid[b[ii], b[jj]]
        boots.append(s[s >= 0])

    def corr(x, y):
        return float(F.corr(x, y)[0])

    def mean6(v, sel=slice(None)):
        return float(np.mean([corr(v[sel], neural[m][sel]) for m in STRONG6]))

    def table(f, con):
        kind, diag, factor, k = con
        c, u, nn, ntok, _ = Q.counts(f, "art", names, fobj)
        kk = np.where(u >= Q.MIN_U)[0]
        m = Q.stat(c, u, nn, kind, diag)[np.ix_(kk, kk)]
        w, v = np.linalg.eigh(0.5 * (m + m.T))
        d = Q.rdm(w, v, kk, factor, k, len(names))
        return d[nodes[ii], nodes[jj]], ntok

    rows = [dict(stage="sphere", series="reference", construction="sphere", tokens_B="", const_covered=n, fit=round(mean6(sphere), 4),
                 fit_lo=round(float(np.percentile([mean6(sphere, s) for s in boots], 2.5)), 4),
                 fit_hi=round(float(np.percentile([mean6(sphere, s) for s in boots], 97.5)), 4),
                 split_half="", spearman_brown="")]
    for stage, series in stages:
        f = find(a.sets, stage)
        if f is None:
            print("skip", stage)
            continue
        h0, h1 = find(a.halves, stage + "_h0"), find(a.halves, stage + "_h1")
        for cname, con in cons.items():
            v, ntok = table(f, con)
            ok = ~np.isnan(v)
            fb = [mean6(v, s[ok[s]]) for s in boots]
            row = dict(stage=stage, series=series, construction=cname, tokens_B=round(ntok / 1e9, 4),
                       const_covered=int(ok.sum() and len(set(ii[ok]) | set(jj[ok]))),
                       fit=round(mean6(v, ok), 4), fit_lo=round(float(np.percentile(fb, 2.5)), 4),
                       fit_hi=round(float(np.percentile(fb, 97.5)), 4), split_half="", spearman_brown="")
            if h0 and h1:
                x, _ = table(h0, con)
                y, _ = table(h1, con)
                g = ~(np.isnan(x) | np.isnan(y))
                r = corr(x[g], y[g])
                row.update(split_half=round(r, 4), spearman_brown=round(2 * r / (1 + r), 4))
            rows.append(row)
            print(stage, cname, row["tokens_B"], row["fit"], row["split_half"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", a.out, len(rows))


if __name__ == "__main__":
    main()
