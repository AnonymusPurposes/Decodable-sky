"""Five-way geographic outcome scoring for the multi-model orthogonal battery.

Every arm is scored against BOTH endpoints, so the same five mutually exclusive
categories describe all three arms:

  no_constellation     the frozen leading-name parser finds no catalogue
                       constellation at the head of the completion
  equals_A             the answer is the source constellation itself
  equals_B             the answer is the destination constellation itself
  within30_A_not_A     within 30 degrees of A, but is neither A nor B
  within30_B_not_B     within 30 degrees of B, but is neither A nor B

Destinations are near-orthogonal to their source, so the two within-30 classes
cannot both fire; `other` collects parsed constellations far from both. The
categories are reported per arm and per model, and `scores_sum` must be 1.

The parser is `core.parse_answer`, unchanged, and reads only the leading name.
No answer is discarded for equalling the source: for the untouched and self arms
the source is the intended answer's neighbourhood. `rescore_multi.py` re-reads
the same completions with the forgiving parser, without a GPU, and adds the
`ramble_A` verdict for a completion that only restates its own source.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import angle, display, parse_answer


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def classify(answer, source, target, catalog, threshold):
    """Exactly one of the five categories, plus `other`."""
    if answer is None or catalog[answer]['type'] != 'constellation':
        return 'no_constellation', None, None
    da = float(angle(catalog[answer]['unit'], catalog[source]['unit']))
    db = float(angle(catalog[answer]['unit'], catalog[target]['unit']))
    if answer == source:
        return 'equals_A', da, db
    if answer == target:
        return 'equals_B', da, db
    if da <= threshold:
        return 'within30_A_not_A', da, db
    if db <= threshold:
        return 'within30_B_not_B', da, db
    return 'other', da, db


CATEGORIES = ['no_constellation', 'equals_A', 'equals_B',
              'within30_A_not_A', 'within30_B_not_B', 'other']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threshold', type=float, default=30.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.prepared / 'manifest.json').read_text())
    model = meta.get('model_key', meta['model'])
    catalog = json.loads((args.prepared / 'catalog.json').read_text())
    specs = {s['id']: s for s in json.loads((args.prepared / 'spec.json').read_text())}
    baselines = {r['id']: r for r in json.loads((args.run / 'baselines.json').read_text())}
    rows = [json.loads(l) for l in (args.run / 'results.jsonl').read_text().splitlines()]
    odds = lambda s: s[1]['logprob'] - s[0]['logprob']

    expected = len(specs) * 3
    if len({(r['arm'], r['id']) for r in rows}) != expected or len(rows) != expected:
        raise AssertionError(f'Expected {expected} trials, found {len(rows)}')

    trials = []
    for r in rows:
        spec = specs[r['id']]
        i = parse_answer(r['completion'], catalog)
        category, da, db = classify(i, spec['source'], spec['target'], catalog, args.threshold)
        # Per-trial columns are kept narrow enough to read and edit by hand: the
        # model, PCA width, layer group and pair separation are constant for the
        # whole file and live in the summary instead. Names drop the
        # " constellation" suffix every catalogue entry carries.
        trials.append(dict(
            arm=r['arm'], id=r['id'],
            A=display(catalog[spec['source']]['name']), B=display(catalog[spec['target']]['name']),
            template=spec['template'],
            answer=display(catalog[i]['name']) if i is not None else '',
            category=category,
            deg_from_A=None if da is None else round(da, 4),
            deg_from_B=None if db is None else round(db, 4),
            **{c: int(category == c) for c in CATEGORIES},
            logodds_change=round(odds(r['scores']) - odds(baselines[r['id']]['baseline_scores']), 6),
            completion=r['completion'].replace('\n', ' ').strip()))
    trials.sort(key=lambda t: (t['arm'], t['A'], t['template']))
    write_csv(args.out / f'trials_{model}.csv', trials)

    angles = list(csv.DictReader((args.prepared / 'harmonic_angles.csv').open(encoding='utf-8')))
    medians = meta['harmonic_angle_medians']
    summary = []
    for arm in ['untouched', 'self', 'target']:
        g = [t for t in trials if t['arm'] == arm]
        row = dict(model=model, repo=meta['model'], arm=arm, n=len(g),
                   layers=f"{meta['layers'][0]}-{meta['layers'][-1]}", n_layers=len(meta['layers']),
                   k=meta['ks'][0], r2_max=round(meta['r2_max'], 4), r2_argmax=meta['r2_argmax'])
        row.update({c: round(float(np.mean([t[c] for t in g])), 4) for c in CATEGORIES})
        row['scores_sum'] = round(sum(row[c] for c in CATEGORIES), 6)
        row['mean_logodds_change'] = round(float(np.mean([t['logodds_change'] for t in g])), 4)
        row.update({k: round(v, 4) for k, v in medians.items()})
        summary.append(row)
    write_csv(args.out / f'summary_{model}.csv', summary)

    print(json.dumps(dict(status='analyzed', model=model, trials=len(trials),
                          harmonic_angle_medians={k: round(v, 4) for k, v in medians.items()},
                          n_angle_layers=len(angles),
                          summary=[{k: v for k, v in s.items()
                                    if k in ['arm'] + CATEGORIES} for s in summary]), indent=2), flush=True)


if __name__ == '__main__':
    main()
