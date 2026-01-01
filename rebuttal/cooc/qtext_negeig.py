#!/usr/bin/env python
"""Negative eigenvalues of the M* matrices behind Q_text (Issue 3).

For each corpus set and construction, the symmetric M* over catalogue objects with u >= MIN_U (the matrix that
qtext_saturation.py / Figure 10 factorize) is diagonalized and the negative part summarized:
  n_obj        objects in the matrix (u >= MIN_U); n_const of them are constellations
  n_neg        eigenvalues < -1e-9 (n_pos > 1e-9)
  neg_mass     sum |lambda| over negative eigenvalues / sum |lambda| over all
  neg_top      |most negative eigenvalue| / largest positive eigenvalue
  neg_in_top10 negative eigenvalues among the 10 largest |lambda|
  uni_cos      |cosine| between the most negative eigenvector and the uniform vector: that mode is the -2 floor of
               never-co-occurring pairs (M* ~ -2(11^T - I)), nearly constant, so it barely moves (v_i - v_j)^2
  *_rest       the same without that mode: neg_mass_rest, neg_top_rest = |2nd most negative| / largest positive
Constructions: within (baseline from the object-object table, observed diagonal: Figure 10's 975B column and Q^+-)
and faithful (M* as in appendix F.3, zero diagonal: the published Q_text).

Usage: qtext_negeig.py --spectral <spectral dir> --sets <dir> [...] --forms surface_forms_v2.csv --out <csv>
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtext_spectrum as Q  # noqa: E402

SETS = ["wiki_old", "wiki", "books_sky_guide", "books_all_books", "web_f1_1", "ext_all", "ext_everything"]
CONS = {"within": ("mstar_tt", "observed"), "faithful": ("mstar", "zero")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.spectral)
    import pandas as pd
    import paths

    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist()
    is_const = (cat["type"].astype(str).str.lower().str.startswith("const")).to_numpy()
    fobj = {}
    for r in csv.DictReader(open(a.forms, encoding="utf-8")):
        fobj.setdefault(r["form"], r["object"])
    rows = []
    for s in SETS:
        f = next((os.path.join(d, s + ".npz") for d in a.sets if os.path.exists(os.path.join(d, s + ".npz"))), None)
        if f is None:
            print("skip", s)
            continue
        c, u, nn, ntok, _ = Q.counts(f, "art", names, fobj)
        kk = np.where(u >= Q.MIN_U)[0]
        for cname, (kind, diag) in CONS.items():
            m = Q.stat(c, u, nn, kind, diag)[np.ix_(kk, kk)]
            w, v = np.linalg.eigh(0.5 * (m + m.T))
            neg, pos = w[w < -1e-9], w[w > 1e-9]
            rest = np.sort(neg)[1:]
            top = w[np.argsort(-np.abs(w))[:10]]
            rows.append(dict(set=s, construction=cname, tokens_B=round(ntok / 1e9, 4), n_obj=len(kk),
                             n_const=int(is_const[kk].sum()), n_pos=len(pos), n_neg=len(neg),
                             neg_mass=round(float(-neg.sum() / np.abs(w).sum()), 4),
                             neg_top=round(float(-neg.min() / pos.max()), 4) if len(neg) else 0.0,
                             neg_in_top10=int((top < 0).sum()),
                             uni_cos=round(float(abs(v[:, 0].sum()) / np.sqrt(len(kk))), 3),
                             neg_mass_rest=round(float(-rest.sum() / (np.abs(w).sum() + neg.min())), 4),
                             neg_top_rest=round(float(-rest.min() / pos.max()), 4)))
            print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("->", a.out, len(rows))


if __name__ == "__main__":
    main()
