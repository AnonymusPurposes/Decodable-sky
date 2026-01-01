#!/usr/bin/env python
"""
Score every <resdir>/<model>/completions_<model>.csv (score_ablation.py) and merge the
per-condition summaries + run diagnostics into one table.

Usage: python collect_ablation.py results_harm_frac80 --objects ../data/astro_objects.csv
Writes <resdir>/all_models_summary.csv (type=all rows) and prints a markdown table.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ORDER = ["qwen32b", "qwen235b", "gptoss120b", "gptoss120b_harmony", "llama33_70b",
         "mixtral8x22b", "mistrallarge123b", "glm45air"]
CONDS = ["none", "sky", "sky_dec", "rand0", "rand1", "rand2",      # ablate_sky.py
         "sky3", "harm8", "comp8", "l2pure", "l2raw"]              # ablate_harm.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resdir")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    rows = []
    for m in ORDER:
        d = os.path.join(args.resdir, m)
        comp = os.path.join(d, f"completions_{m}.csv")
        diag_p = os.path.join(d, f"diagnostics_{m}.json")
        if not (os.path.exists(comp) and os.path.exists(diag_p)):
            continue
        summ = os.path.join(d, f"completions_{m}_summary.csv")
        if args.rescore or not os.path.exists(summ):
            subprocess.run([sys.executable, os.path.join(here, "score_ablation.py"), comp,
                            "--objects", args.objects], check=True, stdout=subprocess.DEVNULL)
        diag = json.load(open(diag_p))
        with open(summ, newline="") as f:
            summary = {r["condition"]: r for r in csv.DictReader(f) if r["type"] == "all"}
        L = diag["layers"]
        for c in [c for c in CONDS if c in summary] + [c for c in summary if c not in CONDS]:
            s = summary[c]
            dh = diag["dh_over_h_last_prompt_token"].get(c, {})
            rows.append(dict(
                model=m, r2_max=round(diag["r2_max"], 3), n_layers=len(L),
                layers=f"{L[0]}-{L[-1]}" if L == list(range(L[0], L[-1] + 1)) else ",".join(map(str, L)),
                condition=c, **{k: s[k] for k in s if k not in ("condition", "type")},
                dh_over_h_mean=round(float(np.mean(list(dh.values()))), 4) if dh else "",
                gen_seconds=diag["timings"].get(c, {}).get("seconds", ""),
                load_seconds=diag["load_seconds"], total_seconds=diag["total_seconds"]))
    out = os.path.join(args.resdir, "all_models_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}\n")
    print("| model | layers | cond | parsed | self | median° [CI] | ≤30° | same as none | ‖Δh‖/‖h‖ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['layers']} ({r['n_layers']}) | {r['condition']} | {r['frac_parsed']} | "
              f"{r['frac_self']} | {r['median_ang_deg']} [{r['median_ci95_lo']}, {r['median_ci95_hi']}] | "
              f"{r['frac_within_30deg']} | {r['frac_same_answer_as_none']} | {r['dh_over_h_mean']} |")


if __name__ == "__main__":
    main()
