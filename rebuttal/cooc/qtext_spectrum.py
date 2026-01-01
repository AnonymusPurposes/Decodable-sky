#!/usr/bin/env python
"""Q_text under every construction the reviewer asks about, at every corpus size.

Issue 3 asks whether M* has negative eigenvalues, and whether a signed or a separate
input/context factorization should be used. For each merged count set this script reports the
share of the spectrum that is negative and the fit under:

  statistic  mstar    corpus-wide baseline, e = 2 S_f u_i u_j / N (the form in appendix F.3)
             mstar_tt baseline from the object-object table, e' = R_i R_j / T (cannot saturate)
  diagonal   zero     what the published code did: M_ii = 0, so trace M = 0 and negative
                      eigenvalues are forced by construction
             observed M_ii computed from the observed self co-occurrence like any other entry
  factor     positive W = V+ sqrt(L+)           the published code
             abs      W = V sqrt(|L|)           appendix F.3 as printed; equal to the input
                                                vectors of an SVD input/context factorization
             signed   D_ij^2 = M_ii + M_jj - 2M_ij   keeps the sign of each mode (a distance only
                                                in the pseudo-Euclidean sense)
  k          all modes, or the top 10 by the size that factorization uses (positive/abs only)

Everything else follows build_qwem_rdm.py: f(d) = 17 - d, L = 16, joint factorization over every
catalogue object with u >= 10, squared distances read on the constellation pairs. Fit is the
Pearson r with 1 - cos(angle) and with the band-averaged neural RDM. Intervals resample
constellations with replacement.

Usage: qtext_spectrum.py --sets <dir> [<dir> ...] --neural <neural_const_rdms.npz>
                         --forms surface_forms_v2.csv --objects data/astro_objects.csv --out <csv>
"""
import argparse
import csv
import glob
import os

import numpy as np

LW = np.array([17.0 - d for d in range(1, 17)])
STRONG6 = ["mistrallarge123b", "qwen235b", "qwen32b", "llama33_70b", "mixtral8x22b", "glm45air"]
MIN_U = 10
BOOT_SETS = {"wiki_old", "wiki", "web_f1_1", "c4", "books_sky_guide", "books_all_books", "ext_all", "ext_everything"}


def counts(f, var, names, fobj):
    d = np.load(f)
    forms = [str(x) for x in d["forms"]]
    oi = {n: i for i, n in enumerate(names)}
    mm = np.zeros((len(forms), len(names)))
    for i, x in enumerate(forms):
        mm[i, oi[fobj[x]]] = 1
    c = np.tensordot(d["Cd_" + var].astype(float), LW, axes=([2], [0]))
    c = mm.T @ (c + c.T) @ mm
    u = mm.T @ d["u_" + var].astype(float)
    n = float(d["n_tok" if var == "raw" else "n_tok_" + var])
    return c, u, n, int(d["n_tok"]), int(d["n_mention_raw" if var == "raw" else "n_mention_art"])


def stat(c, u, n, kind, diag):
    c = c.copy()
    if diag == "zero":
        np.fill_diagonal(c, 0)
    keep = u >= MIN_U
    if kind == "mstar":
        e = 2.0 * LW.sum() * np.outer(u, u) / n
    else:
        r = c[:, keep].sum(1)
        e = np.outer(r, r) / max(r[keep].sum(), 1.0)
    m = 2.0 * (c - e) / (c + e + 1e-12)
    if diag == "zero":
        np.fill_diagonal(m, 0)
    return m


def rdm(w, v, kk, factor, k, size):
    if factor == "positive":
        idx = np.where(w > 1e-9)[0]
        idx = idx[np.argsort(-w[idx])]
        lam = w
    elif factor == "abs":
        idx = np.where(np.abs(w) > 1e-9)[0]
        idx = idx[np.argsort(-np.abs(w[idx]))]
        lam = np.abs(w)
    else:
        idx, lam = np.arange(len(w)), w
    if k is not None:
        idx = idx[:k]
    x = v[:, idx]
    d = np.full((size, size), np.nan)
    g = (x * x * lam[idx]).sum(1)
    d[np.ix_(kk, kk)] = g[:, None] + g[None, :] - 2 * (x * lam[idx]) @ x.T
    return d


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--neural", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(a.objects, encoding="utf-8")))
    names = [r["name"] for r in cat]
    types = np.array([r["type"] for r in cat])
    y = np.array([[float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])] for r in cat])
    fobj = {}
    for r in csv.DictReader(open(a.forms, encoding="utf-8")):
        fobj.setdefault(r["form"], r["object"])
    nodes = np.where(types == "constellation")[0]
    ii, jj = np.triu_indices(len(nodes), 1)
    sph_m = (1 - y @ y.T)[np.ix_(nodes, nodes)]
    z = np.load(a.neural)
    neural = {m: z[m] for m in z.files}
    nmat = {}
    for m, v in neural.items():
        t = np.zeros((len(nodes), len(nodes))); t[ii, jj] = v; nmat[m] = t + t.T
    rng = np.random.default_rng(20261108)
    boots = [rng.integers(0, len(nodes), len(nodes)) for _ in range(a.nboot)]

    rows = []
    files = [f for d in a.sets for f in sorted(glob.glob(os.path.join(d, "*.npz"))) if not f.endswith(("_h0.npz", "_h1.npz"))]
    for f in files:
        name = os.path.basename(f)[:-4]
        for var, lab in (("raw", "pure"), ("art", "gated")):
            c, u, n, ntok, ment = counts(f, var, names, fobj)
            kk = np.where(u >= MIN_U)[0]
            for kind in ("mstar", "mstar_tt"):
                for diag in ("zero", "observed"):
                    m = stat(c, u, n, kind, diag)
                    ms = m[np.ix_(kk, kk)]
                    w, v = np.linalg.eigh(0.5 * (ms + ms.T))
                    neg = float(np.abs(w[w < 0]).sum() / (np.abs(w).sum() + 1e-12))
                    keepc = np.isin(kk, nodes)
                    wc = np.linalg.eigvalsh(0.5 * (ms + ms.T)[np.ix_(keepc, keepc)])
                    negc = float(np.abs(wc[wc < 0]).sum() / (np.abs(wc).sum() + 1e-12))
                    for factor in ("positive", "abs", "signed"):
                        for k in ((None, 10) if factor != "signed" else (None,)):
                            d = rdm(w, v, kk, factor, k, len(names))[np.ix_(nodes, nodes)]
                            ok = ~np.isnan(d[ii, jj])
                            if ok.sum() < 30:
                                continue
                            row = dict(set=name, count=lab, statistic=kind, diagonal=diag, factor=factor,
                                       k=k or "all", tokens_B=round(ntok / 1e9, 4), mentions=ment,
                                       const_covered=int(np.isin(nodes, kk).sum()),
                                       neg_spectrum_frac=round(neg, 4), neg_spectrum_frac_const=round(negc, 4),
                                       r_sphere=round(corr(d[ii, jj][ok], sph_m[ii, jj][ok]), 4))
                            rs = [corr(d[ii, jj][ok], nmat[x][ii, jj][ok]) for x in STRONG6]
                            row["mean6"] = round(float(np.mean(rs)), 4)
                            for x in sorted(neural):
                                row["r_" + x] = round(corr(d[ii, jj][ok], nmat[x][ii, jj][ok]), 4)
                            if name in BOOT_SETS and k in (None, 10):
                                b6, bs, dif = [], [], []
                                for b in boots:
                                    db = d[np.ix_(b, b)]
                                    # drop pairs of a constellation with its own resampled copy
                                    good = (b[ii] != b[jj]) & ~np.isnan(db[ii, jj])
                                    if good.sum() < 30:
                                        continue
                                    x = db[ii, jj][good]
                                    sb = sph_m[np.ix_(b, b)][ii, jj][good]
                                    bs.append(corr(x, sb))
                                    nb = [nmat[q][np.ix_(b, b)][ii, jj][good] for q in STRONG6]
                                    b6.append(np.mean([corr(x, q) for q in nb]))
                                    # paired on the same resample: how far the sphere is ahead of this Q_text
                                    dif.append(np.mean([corr(sb, q) for q in nb]) - b6[-1])
                                row.update(sphere_minus_mean6=round(float(np.mean(
                                               [corr(sph_m[ii, jj][ok], nmat[x][ii, jj][ok]) for x in STRONG6]) - row["mean6"]), 4),
                                           sphere_minus_mean6_lo=round(float(np.percentile(dif, 2.5)), 4),
                                           sphere_minus_mean6_hi=round(float(np.percentile(dif, 97.5)), 4),
                                           mean6_lo=round(float(np.percentile(b6, 2.5)), 4),
                                           mean6_hi=round(float(np.percentile(b6, 97.5)), 4),
                                           sphere_lo=round(float(np.percentile(bs, 2.5)), 4),
                                           sphere_hi=round(float(np.percentile(bs, 97.5)), 4))
                            rows.append(row)
        print("done", name, flush=True)
    cols = ["set", "count", "statistic", "diagonal", "factor", "k", "tokens_B", "mentions", "const_covered",
            "neg_spectrum_frac", "neg_spectrum_frac_const", "r_sphere", "sphere_lo", "sphere_hi", "mean6", "mean6_lo",
            "mean6_hi", "sphere_minus_mean6", "sphere_minus_mean6_lo", "sphere_minus_mean6_hi"] + ["r_" + x for x in sorted(neural)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("->", a.out, len(rows))


if __name__ == "__main__":
    main()
