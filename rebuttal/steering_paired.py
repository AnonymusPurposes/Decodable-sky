#!/usr/bin/env python
"""Steering denominators and paired effects over the five evaluation templates (0, 18, 25, 26, 27).

Same inputs as steering_templates.py (rescore_multi.py trials_rich_<model>.csv, rich_v3 parser).
Writes <out>/steering_paired.csv, one row per model, modes, arm:
  n                       prompts in the arm (88 pairs x 5 templates = 440)
  usable                  prompts whose completion names a constellation (n - no_constellation)
  near_A, near_B,         fractions of n, pooled over templates, with _min/_max over the 5 templates
  near_B_excl_B           (near_B_excl_B = within 30 deg of B but not B itself)
  gain_*                  self/target arms only: paired difference with the untouched arm on the same
                          prompt (same pair, same template), pooled, with _min/_max over templates
  into_* / out_of_*       self/target arms only: prompts that enter / leave the category relative to
                          the untouched completion of the same prompt (discordant pairs)
  *_lo / *_hi             95% bootstrap interval of the pooled rate (and of the paired gain), resampling the
                          88 source constellations with all their templates (2000 resamples, seed 0)
Also writes <out>/steering_paired_l12_vs_l1.csv, one row per model and category (near_B, near_B_excl_B,
near_A): target arm with l=1,2 modes minus target arm with l=1 modes on the same prompt (same pair, same
template), pooled, with the 95% paired bootstrap interval over source constellations (same resamples for both
terms). The untouched arms of the two runs are checked to be identical.

Usage: steering_paired.py --old <steering_pair dir> --new <steering_templates dir> --out <dir>
"""
import argparse
import csv
import os

import numpy as np

ARMS = ('untouched', 'self', 'target')
TEMPLATES = (0, 18, 25, 26, 27)
N_BOOT = 2000


def boot(vals, sources, src_idx):
    """95% interval of the pooled mean of vals {prompt: value}, resampling source constellations (prompt[0])"""
    per = np.array([[vals[p] for p in vals if p[0] == a] for a in sources], dtype=float)   # sources x templates
    means = per[src_idx].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def metrics(r):
    near_b_excl = int(r['within30_B_not_B'])
    return dict(near_A=int(r['equals_A']) + int(r['within30_A_not_A']),
                near_B=int(r['equals_B']) + near_b_excl,
                near_B_excl_B=near_b_excl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rich', default='rich_v3')
    a = ap.parse_args()
    out = []
    kept = {}
    for modes, sub in (('l=1,2', 'results_' + a.rich), ('l=1', 'results_l1_' + a.rich)):
        for f in sorted(f for f in os.listdir(os.path.join(a.new, sub)) if f.startswith('trials_rich_')):
            m = f[len('trials_rich_'):-4]
            trials = [r for d in (a.old, a.new) for r in csv.DictReader(open(os.path.join(d, sub, f), encoding='utf-8'))]
            by = {}
            for r in trials:
                key = (r['arm'], r['A'], r['B'], int(r['template']))
                assert key not in by, (m, modes, key)
                by[key] = r
            prompts = sorted({k[1:] for k in by if k[0] == 'untouched'})
            assert len(prompts) == 88 * len(TEMPLATES) and sorted({p[2] for p in prompts}) == list(TEMPLATES), (m, modes)
            base = {p: metrics(by[('untouched',) + p]) for p in prompts}
            sources = sorted({p[0] for p in prompts})
            src_idx = np.random.default_rng(0).integers(0, len(sources), (N_BOOT, len(sources)))
            for arm in ARMS:
                rr = {p: by[(arm,) + p] for p in prompts}
                met = {p: metrics(rr[p]) for p in prompts}
                row = dict(model=m, modes=modes, arm=arm, n=len(prompts),
                           usable=sum(1 - int(rr[p]['no_constellation']) for p in prompts))
                for k in ('near_A', 'near_B', 'near_B_excl_B'):
                    per_t = [np.mean([met[p][k] for p in prompts if p[2] == t]) for t in TEMPLATES]
                    row[k] = round(float(np.mean([met[p][k] for p in prompts])), 4)
                    row[k + '_min'], row[k + '_max'] = round(float(min(per_t)), 4), round(float(max(per_t)), 4)
                    row[k + '_lo'], row[k + '_hi'] = boot({p: met[p][k] for p in prompts}, sources, src_idx)
                kept[(m, modes, arm)] = met
                for k in ('near_B', 'near_B_excl_B'):
                    if arm == 'untouched':
                        row.update({'gain_' + k: '', 'gain_%s_min' % k: '', 'gain_%s_max' % k: '',
                                    'gain_%s_lo' % k: '', 'gain_%s_hi' % k: '', 'into_' + k: '', 'out_of_' + k: ''})
                        continue
                    d = {p: met[p][k] - base[p][k] for p in prompts}
                    per_t = [np.mean([d[p] for p in prompts if p[2] == t]) for t in TEMPLATES]
                    row['gain_' + k] = round(float(np.mean(list(d.values()))), 4)
                    row['gain_%s_min' % k], row['gain_%s_max' % k] = round(float(min(per_t)), 4), round(float(max(per_t)), 4)
                    row['gain_%s_lo' % k], row['gain_%s_hi' % k] = boot(d, sources, src_idx)
                    row['into_' + k] = sum(v == 1 for v in d.values())
                    row['out_of_' + k] = sum(v == -1 for v in d.values())
                out.append(row)
    os.makedirs(a.out, exist_ok=True)
    diff = []
    for m in sorted({k[0] for k in kept}):
        if (m, 'l=1', 'target') not in kept or (m, 'l=1,2', 'target') not in kept:
            continue
        assert kept[(m, 'l=1', 'untouched')] == kept[(m, 'l=1,2', 'untouched')], (m, 'untouched arms differ')
        t2, t1 = kept[(m, 'l=1,2', 'target')], kept[(m, 'l=1', 'target')]
        assert set(t2) == set(t1)
        sources = sorted({p[0] for p in t2})
        src_idx = np.random.default_rng(0).integers(0, len(sources), (N_BOOT, len(sources)))
        for k in ('near_B', 'near_B_excl_B', 'near_A'):
            d = {p: t2[p][k] - t1[p][k] for p in t2}
            lo, hi = boot(d, sources, src_idx)
            diff.append(dict(model=m, category=k, n=len(d), l12=round(float(np.mean([t2[p][k] for p in t2])), 4),
                             l1=round(float(np.mean([t1[p][k] for p in t1])), 4),
                             diff=round(float(np.mean(list(d.values()))), 4), diff_lo=lo, diff_hi=hi,
                             l12_only=sum(v == 1 for v in d.values()), l1_only=sum(v == -1 for v in d.values())))
    with open(os.path.join(a.out, 'steering_paired_l12_vs_l1.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(diff[0]))
        w.writeheader()
        w.writerows(diff)
    path = os.path.join(a.out, 'steering_paired.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print('wrote', len(out), 'rows ->', path)


if __name__ == '__main__':
    main()
