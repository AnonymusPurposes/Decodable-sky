"""Re-score finished batteries with the forgiving parser, without re-running them.

Reads the `trials_<model>.csv` files an `analyze_multi.py` run produced, applies
`parse_rich.classify` to every completion, and writes a matching set of
`trials_rich_<model>.csv` plus combined summary and off-target tables.

Each trial keeps the shape `analyze_multi.py` gave it, with the columns from
`answer` to `other` rewritten by the forgiving parser, so one row carries one
verdict and can be read or corrected by hand. `answer` lists every candidate a
multiple-choice answer offered, `|`-separated, and the distances are that set's
means. The frozen `core.parse_answer` reading survives in the summary, as
`frozen_no_constellation` and `recovered_fraction`.

Rows are ordered by template first, since a template is a different question
and the two are usually read apart, then by arm and source as before.

Needs only the trial tables and a `catalog.json` from any prepared bundle of the
same catalogue; no GPU and no activation archive.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import display
from parse_rich import CATEGORIES, build_lookup, classify

ORDER = ['qwen32b', 'glm45air', 'llama33_70b', 'mistrallarge123b', 'mixtral8x22b', 'qwen235b']
ARMS = ['untouched', 'self', 'target']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verdict(row):
    """(category, answer, kind) from either shape of a scored trial table.

    A merged table states the forgiving parser's reading in the ordinary
    columns; an older one kept it in `rich_*` beside the frozen reading.
    """
    if 'rich_category' in row:
        return row['rich_category'], row['rich_answer'], row['rich_kind']
    return row['category'], row['answer'], ''


def merge(row, fields):
    """The trial row with `answer`..`other` replaced by the forgiving reading."""
    scores = {c: int(row['rich_category'] == c) for c in CATEGORIES}
    fresh = dict(answer=row['rich_answer'], category=row['rich_category'],
                 deg_from_A=row['rich_deg_from_A'], deg_from_B=row['rich_deg_from_B'], **scores)
    out = {}
    for k in fields:
        out[k] = fresh.get(k, row[k])
        if k == CATEGORIES[0]:
            # A category the run predates, such as ramble_A, belongs beside its
            # siblings rather than at the end of the row.
            out.update({c: scores[c] for c in CATEGORIES if c not in fields})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=Path, required=True, help='Directory of trials_<model>.csv')
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--templates', type=Path, default=None,
                    help='templates.json, so the dump of unexpected rows carries real prompts')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compare', type=Path, default=None,
                    help='An earlier rescore directory; lists every trial whose verdict moved')
    ap.add_argument('--threshold', type=float, default=30.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(args.catalog.read_text(encoding='utf-8'))
    templates = json.loads(args.templates.read_text(encoding='utf-8')) if args.templates else None
    forms, pattern = build_lookup(catalog)
    # Trial tables carry display names; accept the full catalogue form too.
    ident = {}
    for c in catalog:
        ident[c['name']] = c['id']
        ident[display(c['name'])] = c['id']

    # What the steering is supposed to produce: the source neighbourhood for the
    # control arms, the destination neighbourhood for the steered arm.
    expected = {'untouched': ('equals_A', 'within30_A_not_A'),
                'self': ('equals_A', 'within30_A_not_A'),
                'target': ('equals_B', 'within30_B_not_B')}
    summary, unexpected, changes = [], [], []
    for path in sorted(args.trials.glob('trials_*.csv')):
        model = path.name[len('trials_'):-len('.csv')]
        if model.startswith('rich_'):
            continue
        reader = csv.DictReader(path.open(encoding='utf-8'))
        rows, fields = list(reader), reader.fieldnames
        before = {}
        if args.compare is not None:
            old = args.compare / f'trials_rich_{model}.csv'
            if old.exists():
                before = {(r['arm'], r['id']): r for r in csv.DictReader(old.open(encoding='utf-8'))}
        for r in rows:
            r.update(classify(r['completion'], ident[r['A']], ident[r['B']],
                              catalog, forms, pattern, args.threshold))
            r['frozen_category'] = next((c for c in CATEGORIES if r.get(c) == '1'), 'other')
            r['recovered'] = int(r['frozen_category'] == 'no_constellation'
                                 and r['rich_category'] != 'no_constellation')
            r['as_expected'] = int(r['rich_category'] in expected[r['arm']])
            if not r['as_expected']:
                prompt = ''
                if templates is not None:
                    prompt = templates[int(r['template'])].replace('X', r['A'])
                unexpected.append(dict(
                    model=model, arm=r['arm'], aim='A' if r['arm'] != 'target' else 'B',
                    A=r['A'], B=r['B'], template=r['template'], prompt=prompt,
                    rich_category=r['rich_category'], frozen_category=r['frozen_category'],
                    rich_kind=r['rich_kind'], rich_answer=r['rich_answer'],
                    deg_from_A=r['rich_deg_from_A'], deg_from_B=r['rich_deg_from_B'],
                    completion=r['completion']))
            old = before.get((r['arm'], r['id']))
            if old is not None:
                was_category, was_answer, was_kind = verdict(old)
                if was_category != r['rich_category'] or was_answer != r['rich_answer']:
                    changes.append(dict(
                        model=model, arm=r['arm'], id=r['id'], A=r['A'], B=r['B'],
                        template=r['template'],
                        was_category=was_category, now_category=r['rich_category'],
                        was_answer=was_answer, now_answer=r['rich_answer'],
                        was_kind=was_kind, now_kind=r['rich_kind'],
                        completion=r['completion']))
        merged = sorted((merge(r, fields) for r in rows),
                        key=lambda t: (int(t['template']), t['arm'], t['A']))
        write_csv(args.out / f'trials_rich_{model}.csv', merged)

        for arm in ARMS:
            g = [r for r in rows if r['arm'] == arm]
            if not g:
                continue
            row = dict(model=model, arm=arm, n=len(g), parser='rich')
            row.update({c: round(float(np.mean([r['rich_category'] == c for r in g])), 4)
                        for c in CATEGORIES})
            row['scores_sum'] = round(sum(row[c] for c in CATEGORIES), 6)
            row['frozen_no_constellation'] = round(
                float(np.mean([r['frozen_category'] == 'no_constellation' for r in g])), 4)
            row['recovered_fraction'] = round(float(np.mean([r['recovered'] for r in g])), 4)
            row['option_list_fraction'] = round(
                float(np.mean([r['rich_kind'] == 'options' for r in g])), 4)
            row['near_B'] = round(row['equals_B'] + row['within30_B_not_B'], 4)
            row['near_A'] = round(row['equals_A'] + row['within30_A_not_A'], 4)
            row['as_expected'] = round(float(np.mean([r['as_expected'] for r in g])), 4)
            summary.append(row)

    rank = {m: i for i, m in enumerate(ORDER)}
    summary.sort(key=lambda r: (rank.get(r['model'], len(ORDER)), ARMS.index(r['arm'])))
    write_csv(args.out / 'all_models_summary_rich.csv', summary)
    if unexpected:
        unexpected.sort(key=lambda r: (rank.get(r['model'], len(ORDER)),
                                       ARMS.index(r['arm']), r['rich_category'], r['A']))
        write_csv(args.out / 'unexpected_rows.csv', unexpected)
    if changes:
        changes.sort(key=lambda r: (rank.get(r['model'], len(ORDER)),
                                    ARMS.index(r['arm']), r['A']))
        write_csv(args.out / 'parser_changes.csv', changes)
    print(json.dumps(dict(status='rescored', models=len({s['model'] for s in summary}),
                          rows=len(summary), unexpected=len(unexpected),
                          changed=len(changes), out=str(args.out)), indent=2), flush=True)


if __name__ == '__main__':
    main()
