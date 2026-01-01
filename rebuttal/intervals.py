#!/usr/bin/env python
"""Confidence intervals and baselines for the intervention experiments (no GPU).

steering_intervals.csv (only with --steering; the first version's two-template battery)
    Rate of each answer category per model, mode set (l=1 or l=1,2) and arm, with a 95%
    percentile bootstrap interval that resamples SOURCE constellations as clusters (88 sources,
    two templates each). Paired differences use the same resampled sources for both terms.
    near_B counts B itself or a neighbour within 30 deg; near_B_excl_B leaves B itself out.
    Superseded in the paper by steering_paired.py (five templates, A and B held out of the fit).

ablation_intervals.csv
    Median angle per model and condition as in score_ablation.py (category "object"), with a
    95% interval on the median and on its paired change from no ablation, both resampling
    query objects as clusters. Also the 5th/95th percentiles of single answers (the spread,
    a different quantity from the interval) and the share of top-16 PCA variance each condition
    removes, averaged over the ablated layers.

baseline_angle.csv
    Two null baselines for the median angular error.
    protocol_null: model-independent. The "activations" are the true sky vectors, labels are
    shuffled across objects, and the 80/20 and leave-one-out protocols run unchanged.
    answers_shuffled: the reviewers' suggestion. A model's named answers are kept and the query
    objects are shuffled across them, per model, for the no-ablation answers.

Usage: intervals.py --ablation rebuttal/data/ablation [--steering <dir>] --objects data/astro_objects.csv
                    --bright ablation/bright_stars_v4.csv --out rebuttal/results
"""
import argparse
import csv
import glob
import json
import os
import gzip

import numpy as np


def scored_path(d, m):
    """completions_<m>_scored.csv in d, or its gzipped copy (as shipped in rebuttal/data/ablation)"""
    f = os.path.join(d, f"completions_{m}_scored.csv")
    return f if os.path.exists(f) else (f + ".gz" if os.path.exists(f + ".gz") else None)


def open_text(f):
    return gzip.open(f, "rt", encoding="utf-8", newline="") if f.endswith(".gz") else open(f, encoding="utf-8", newline="")

NB = 5000
CATS = ["no_constellation", "ramble_A", "equals_A", "within30_A_not_A", "equals_B", "within30_B_not_B", "other"]
ABL = ["none", "comp8", "l2pure", "sky3", "l2raw", "harm8"]


def pct(a, axis=0):
    return np.percentile(a, 2.5, axis=axis), np.percentile(a, 97.5, axis=axis)


def load_trials(path):
    """-> {arm: {category: (sources x templates) 0/1 array}}, sorted source names"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    sources, temps = sorted({r["A"] for r in rows}), sorted({r["template"] for r in rows})
    si, ti = {a: i for i, a in enumerate(sources)}, {t: i for i, t in enumerate(temps)}
    out = {}
    for arm in ("untouched", "self", "target"):
        d = {c: np.full((len(sources), len(temps)), np.nan) for c in CATS}
        for r in rows:
            if r["arm"] == arm:
                for c in CATS:
                    d[c][si[r["A"]], ti[r["template"]]] = float(r["category"] == c)
        d["near_A"] = d["equals_A"] + d["within30_A_not_A"]
        d["near_B"] = d["equals_B"] + d["within30_B_not_B"]
        d["near_B_excl_B"] = d["within30_B_not_B"]
        out[arm] = d
    return out, sources


def boot_mean(x, idx):
    return np.nanmean(x, axis=1)[idx].mean(1)


def steering(root, out):
    rows, diffs = [], []
    runs = {"l=1,2": "results_fit_heldout_typed_rich_v2", "l=1": "results_fit_heldout_typed_l1_rich_v2"}
    data = {}
    for modes, sub in runs.items():
        for f in sorted(glob.glob(os.path.join(root, sub, "trials_rich_*.csv"))):
            m = os.path.basename(f)[len("trials_rich_"):-4]
            data[(m, modes)] = load_trials(f)
    rng = np.random.default_rng(20261108)
    n = len(next(iter(data.values()))[1])
    idx = rng.integers(0, n, (NB, n))
    for (m, modes), (d, srcs) in data.items():
        assert len(srcs) == n
        for arm in d:
            for c in CATS + ["near_A", "near_B", "near_B_excl_B"]:
                b = boot_mean(d[arm][c], idx)
                lo, hi = pct(b)
                rows.append(dict(model=m, modes=modes, arm=arm, metric=c, n_sources=n,
                                 rate=round(float(np.nanmean(d[arm][c])), 4), ci95_lo=round(float(lo), 4),
                                 ci95_hi=round(float(hi), 4)))

        def paired(name, a, b):
            obs = float(np.nanmean(a) - np.nanmean(b))
            lo, hi = pct(boot_mean(a, idx) - boot_mean(b, idx))
            diffs.append(dict(model=m, modes=modes, contrast=name, difference=round(obs, 4),
                              ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4)))
        paired("target near_B - untouched near_B", d["target"]["near_B"], d["untouched"]["near_B"])
        paired("target near_B_excl_B - untouched near_B_excl_B", d["target"]["near_B_excl_B"], d["untouched"]["near_B_excl_B"])
        paired("untouched near_A - target near_A", d["untouched"]["near_A"], d["target"]["near_A"])
        paired("self near_A - untouched near_A", d["self"]["near_A"], d["untouched"]["near_A"])
    for (m, modes), (d, _) in data.items():
        if modes == "l=1,2" and (m, "l=1") in data:
            a, b = d["target"]["near_B"], data[(m, "l=1")][0]["target"]["near_B"]
            lo, hi = pct(boot_mean(a, idx) - boot_mean(b, idx))
            diffs.append(dict(model=m, modes="l=1,2 vs l=1", contrast="target near_B: l=1,2 - l=1",
                              difference=round(float(np.nanmean(a) - np.nanmean(b)), 4),
                              ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4)))
    write(os.path.join(out, "steering_intervals.csv"), rows)
    write(os.path.join(out, "steering_paired_differences.csv"), diffs)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows))


def ablation(root, out):
    rows = []
    rng = np.random.default_rng(20261108)
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        m = os.path.basename(d.rstrip("/\\"))
        f = scored_path(d, m)
        if f is None:
            continue
        sc = list(csv.DictReader(open_text(f)))
        diag = json.load(open(os.path.join(d, f"diagnostics_{m}.json"), encoding="utf-8"))
        per = {c: {} for c in ABL}
        for r in sc:
            if r["category"] == "object" and r["condition"] in per:
                per[r["condition"]].setdefault(int(r["obj_id"]), []).append(float(r["ang_deg"]))
        objs = sorted({o for c in ABL for o in per[c]})
        idx = rng.integers(0, len(objs), (NB, len(objs)))

        def boot(c):
            arr = [np.asarray(per[c].get(o, []), float) for o in objs]
            return np.array([np.median(np.concatenate([arr[i] for i in s])) for s in idx])
        base = boot("none")
        var = {c: float(np.mean([v[c]["var_frac_top16_removed"] for v in diag["subspaces"].values() if c in v]))
               for c in ABL if c != "none"}
        for c in ABL:
            a = np.concatenate([np.asarray(v, float) for v in per[c].values()])
            b = boot(c)
            lo, hi = pct(b)
            dlo, dhi = pct(b - base)
            obs0 = np.median(np.concatenate([np.asarray(v, float) for v in per["none"].values()]))
            rows.append(dict(model=m, condition=c, n_answers=int(a.size), median_deg=round(float(np.median(a)), 2),
                             ci95_lo=round(float(lo), 2), ci95_hi=round(float(hi), 2),
                             change_from_none=round(float(np.median(a) - obs0), 2),
                             change_ci95_lo=round(float(dlo), 2), change_ci95_hi=round(float(dhi), 2),
                             single_answer_p05=round(float(np.percentile(a, 5)), 2),
                             single_answer_p95=round(float(np.percentile(a, 95)), 2),
                             var_frac_top16_removed=round(var[c], 4) if c in var else ""))
    write(os.path.join(out, "ablation_intervals.csv"), rows)


def unit(raw):
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def ang(p, u):
    return np.degrees(np.arccos(np.clip((unit(p) * unit(u)).sum(1), -1, 1)))


def baselines(root, objects, bright, out):
    cat = list(csv.DictReader(open(objects, encoding="utf-8")))
    Y = np.array([[float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])] for r in cat])
    no = len(Y)
    rows = []
    rng = np.random.default_rng(20261108)

    def fit_pred(xtr, ytr, xte):
        A = np.column_stack([np.ones(len(xtr)), xtr])
        b = np.linalg.lstsq(A, ytr, rcond=None)[0]
        return np.column_stack([np.ones(len(xte)), xte]) @ b

    meds = {"protocol_null_80_20": [], "protocol_null_loo": []}
    for _ in range(2000):
        lab = Y[rng.permutation(no)]
        te = rng.permutation(no)[:38]
        mask = np.zeros(no, bool); mask[te] = True
        meds["protocol_null_80_20"].append(np.median(ang(fit_pred(Y[~mask], lab[~mask], Y[mask]), lab[mask])))
    for _ in range(300):
        lab = Y[rng.permutation(no)]
        e = []
        for i in range(no):
            k = np.arange(no) != i
            e.append(ang(fit_pred(Y[k], lab[k], Y[[i]]), lab[[i]])[0])
        meds["protocol_null_loo"].append(np.median(e))
    for name, v in meds.items():
        rows.append(dict(baseline=name, model="any", n_null=len(v), median_deg=round(float(np.median(v)), 2),
                         p05=round(float(np.percentile(v, 5)), 2), p95=round(float(np.percentile(v, 95)), 2)))

    coord = {r["name"]: Y[i] for i, r in enumerate(cat)}
    if bright and os.path.exists(bright):
        for r in csv.DictReader(open(bright, encoding="utf-8")):
            if r.get("name") and r["name"] not in coord:
                try:
                    coord[r["name"]] = np.array([float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])])
                except (KeyError, ValueError):
                    pass
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        m = os.path.basename(d.rstrip("/\\"))
        f = scored_path(d, m)
        if f is None:
            continue
        rs = [r for r in csv.DictReader(open_text(f)) if r["condition"] == "none"
              and r["category"] == "object" and r["answer"] in coord]
        ans = np.array([coord[r["answer"]] for r in rs])
        q = np.array([Y[int(r["obj_id"])] for r in rs])
        ids = np.array([int(r["obj_id"]) for r in rs])
        v = []
        for _ in range(2000):                       # relabel whole objects, so a query's prompts move together
            perm = rng.permutation(no)
            v.append(np.median(np.degrees(np.arccos(np.clip((Y[perm[ids]] * ans).sum(1), -1, 1)))))
        rows.append(dict(baseline="answers_shuffled", model=m, n_null=len(v), median_deg=round(float(np.median(v)), 2),
                         p05=round(float(np.percentile(v, 5)), 2), p95=round(float(np.percentile(v, 95)), 2)))
    write(os.path.join(out, "baseline_angle.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering", default=None)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--bright", default="ablation/bright_stars_v4.csv")
    ap.add_argument("--out", default="rebuttal/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.steering:
        steering(a.steering, a.out)
    ablation(a.ablation, a.out)
    baselines(a.ablation, a.objects, a.bright, a.out)


if __name__ == "__main__":
    main()
