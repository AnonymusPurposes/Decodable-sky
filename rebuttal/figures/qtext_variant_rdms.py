#!/usr/bin/env python
"""Q_text RDM tables for every construction of Issue 3, in the format of spectral/rdms/*.csv.

Figure 10/20 score each target against the neural RDM with spectral/fit_models.py. To put the Q_text
variants on exactly that footing, this script writes each one as a 188 x 188 table of squared
distances (objects not covered by the counts are left empty; every constellation is covered) and
spectral_intervals.py then scores them with the paper's neural RDM and bootstrap.

Variant name: q_<set>_<statistic>_<diagonal>_<factor>_<k>, built with cooc/qtext_spectrum.py
(gated counts, as the paper's qwem_art_L16):
  statistic  mstar (appendix F.3) | mstartt (baseline from the object-object table)
  diagonal   zero (published code) | obs (observed self co-occurrence)
  factor     pos (positive eigenvalues, published code) | abs (|Lambda|, as printed) | sgn (signed)
  k          all | 10

Usage: qtext_variant_rdms.py --sets <npz> [<npz> ...] --forms <surface_forms_v2.csv>
                             --objects <astro_objects.csv> --out <dir>/rdms
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cooc"))
import qtext_spectrum as Q  # noqa: E402

SHORT = {"mstar": "mstar", "mstar_tt": "mstartt", "zero": "zero", "observed": "obs",
         "positive": "pos", "abs": "abs", "signed": "sgn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(a.objects, encoding="utf-8")))
    names = [r["name"] for r in cat]
    fobj = {}
    for r in csv.DictReader(open(a.forms, encoding="utf-8")):
        fobj.setdefault(r["form"], r["object"])
    os.makedirs(a.out, exist_ok=True)
    for f in a.sets:
        set_name = os.path.basename(f)[:-4]
        c, u, n, _, _ = Q.counts(f, "art", names, fobj)
        kk = np.where(u >= Q.MIN_U)[0]
        for kind in ("mstar", "mstar_tt"):
            for diag in ("zero", "observed"):
                m = Q.stat(c, u, n, kind, diag)
                ms = m[np.ix_(kk, kk)]
                w, v = np.linalg.eigh(0.5 * (ms + ms.T))
                for factor in ("positive", "abs", "signed"):
                    for k in ((None, 10) if factor != "signed" else (None,)):
                        d = Q.rdm(w, v, kk, factor, k, len(names))
                        tag = "q_%s_%s_%s_%s_%s" % (set_name, SHORT[kind], SHORT[diag], SHORT[factor], k or "all")
                        with open(os.path.join(a.out, tag + ".csv"), "w", newline="", encoding="utf-8") as fh:
                            wr = csv.writer(fh)
                            wr.writerow([""] + names)
                            for i, nm in enumerate(names):
                                wr.writerow([nm] + ["" if np.isnan(x) else "%.7g" % x for x in d[i]])
        print("done", set_name, "covered", len(kk), flush=True)


if __name__ == "__main__":
    main()
