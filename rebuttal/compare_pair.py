#!/usr/bin/env python
"""Steering with the map fitted without each pair's A and B, against the paper's fit (Issue 5).

Both runs have the same trials (source A, target B, template). Intervals are 95% percentile bootstrap,
resampling source constellations as clusters; a difference uses the same resample for both terms.
Rich-scored trials (rescore_multi.py) are used on both sides.

Usage: compare_pair.py --pair <dir with results_rich, results_l1_rich> --paper <multimodel dir> --out <csv>
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intervals import boot_mean, load_trials, pct  # noqa: E402

NB = 5000
RUNS = (("l=1,2", "results_rich", "results_fit_heldout_typed_rich_v2"),
        ("l=1", "results_l1_rich", "results_fit_heldout_typed_l1_rich_v2"))
SHOW = (("target", "near_B"), ("target", "near_B_excl_B"), ("target", "equals_B"), ("target", "near_A"),
        ("self", "near_A"), ("untouched", "near_A"), ("untouched", "near_B"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261109)
    rows = []
    for modes, sub, psub in RUNS:
        for f in sorted(glob.glob(os.path.join(a.pair, sub, "trials_rich_*.csv"))):
            m = os.path.basename(f)[len("trials_rich_"):-4]
            g = os.path.join(a.paper, psub, "trials_rich_%s.csv" % m)
            if not os.path.exists(g):
                print("no paper trials for", m, modes)
                continue
            new, s1 = load_trials(f)
            old, s2 = load_trials(g)
            assert s1 == s2, m
            idx = rng.integers(0, len(s1), (NB, len(s1)))

            def row(arm, metric, x, y, lab):
                nb, ob = boot_mean(x, idx), boot_mean(y, idx)
                (lo, hi), (dlo, dhi) = pct(nb), pct(nb - ob)
                rows.append(dict(model=m, modes=modes, arm=arm, metric=lab, n_sources=len(s1),
                                 pair_holdout=round(float(np.nanmean(x)), 4), pair_lo=round(float(lo), 4), pair_hi=round(float(hi), 4),
                                 paper_fit=round(float(np.nanmean(y)), 4),
                                 difference=round(float(np.nanmean(x) - np.nanmean(y)), 4),
                                 diff_lo=round(float(dlo), 4), diff_hi=round(float(dhi), 4)))
            for arm, metric in SHOW:
                row(arm, metric, new[arm][metric], old[arm][metric], metric)
            row("target-untouched", "near_B", new["target"]["near_B"] - new["untouched"]["near_B"],
                old["target"]["near_B"] - old["untouched"]["near_B"], "near_B_gain")
            row("target-untouched", "near_B_excl_B", new["target"]["near_B_excl_B"] - new["untouched"]["near_B_excl_B"],
                old["target"]["near_B_excl_B"] - old["untouched"]["near_B_excl_B"], "near_B_excl_B_gain")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
