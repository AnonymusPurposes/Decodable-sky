"""Merge the per-model summaries of the multi-model battery into one table."""
import argparse
import csv
from pathlib import Path

ORDER = ['qwen32b', 'glm45air', 'llama33_70b', 'mistrallarge123b', 'mixtral8x22b', 'qwen235b']
ARMS = ['untouched', 'self', 'target']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True, help='Directory of summary_<model>.csv files')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.results.glob('summary_*.csv')):
        rows += list(csv.DictReader(path.open(encoding='utf-8')))
    if not rows:
        raise SystemExit(f'No summary_*.csv under {args.results}')
    rank = {m: i for i, m in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r['model'], len(ORDER)), ARMS.index(r['arm'])))

    with args.out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows from {len({r["model"] for r in rows})} models -> {args.out}')


if __name__ == '__main__':
    main()
