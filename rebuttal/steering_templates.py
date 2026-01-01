#!/usr/bin/env python
"""Pair-held-out steering per evaluation template: 0 and 18 (the published battery) plus the three of
steering_templates_extra.csv (run_templates.sbatch). The spread over the five templates is the
uncertainty shown in the rebuttal's Figure 7.

Both inputs are rescore_multi.py outputs (trials_rich_<model>.csv, rich parser), read from
results_<rich>/ and results_l1_<rich>/; rich_v3 is the parser that also strips "borders A". Writes
  <out>/steering_templates.csv         one row per model, modes, template, arm: category fractions in
                                       the column order of all_models_summary_rich.csv, plus near_A/near_B
  <out>/steering_templates_spread.csv  per model, modes, arm, metric: pooled rate over the five templates,
                                       min, max and SD over templates

Usage: steering_templates.py --old <steering_pair dir> --new <steering_templates dir> --out <dir>
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ['no_constellation', 'ramble_A', 'equals_A', 'equals_B', 'within30_A_not_A', 'within30_B_not_B', 'other']
ARMS = ('untouched', 'self', 'target')
METRICS = ('near_A', 'near_B', 'equals_B', 'no_constellation')


def templates():
    loc = [s.strip().strip('"') for s in open(os.path.join(HERE, '..', 'data', 'astro_prompts_location.csv'), encoding='utf-8') if s.strip()]
    extra = [s.strip().strip('"') for s in open(os.path.join(HERE, 'steering_templates_extra.csv'), encoding='utf-8') if s.strip()]
    return {**{t: loc[t] for t in (0, 18)}, **{len(loc) + i: e for i, e in enumerate(extra)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rich', default='rich_v3', help='suffix of the rescored folders, results_<rich>, results_l1_<rich>')
    a = ap.parse_args()
    text = templates()
    rows, spread = [], []
    for modes, sub in (('l=1,2', 'results_' + a.rich), ('l=1', 'results_l1_' + a.rich)):
        new_files = sorted(f for f in os.listdir(os.path.join(a.new, sub)) if f.startswith('trials_rich_'))
        for f in new_files:
            m = f[len('trials_rich_'):-4]
            trials = [r for d in (a.old, a.new) for r in csv.DictReader(open(os.path.join(d, sub, f), encoding='utf-8'))]
            tids = sorted({int(r['template']) for r in trials})
            assert tids == sorted(text), (m, modes, tids)
            for arm in ARMS:
                per = {}
                for t in tids:
                    g = [r for r in trials if r['arm'] == arm and int(r['template']) == t]
                    assert len(g) == 88, (m, modes, arm, t, len(g))
                    row = dict(model=m, modes=modes, template=t, template_text=text[t], arm=arm, n=len(g))
                    row.update({c: round(float(np.mean([int(r[c]) for r in g])), 4) for c in CATS})
                    row['near_B'] = round(row['equals_B'] + row['within30_B_not_B'], 4)
                    row['near_A'] = round(row['equals_A'] + row['within30_A_not_A'], 4)
                    rows.append(row)
                    per[t] = row
                for met in METRICS:
                    v = np.array([per[t][met] for t in tids])
                    spread.append(dict(model=m, modes=modes, arm=arm, metric=met, n_templates=len(v),
                                       pooled=round(float(v.mean()), 4), min=round(float(v.min()), 4),
                                       max=round(float(v.max()), 4), sd=round(float(v.std(ddof=1)), 4),
                                       **{f't{t}': per[t][met] for t in tids}))
    os.makedirs(a.out, exist_ok=True)
    for name, rr in (('steering_templates.csv', rows), ('steering_templates_spread.csv', spread)):
        with open(os.path.join(a.out, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    print('wrote', len(rows), 'template rows,', len(spread), 'spread rows ->', a.out)


if __name__ == '__main__':
    main()
