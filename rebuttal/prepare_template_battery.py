#!/usr/bin/env python
"""Same steering battery on extra evaluation templates.

Copies a finished heldout_typed bundle (prepare_multi.py) and replaces its battery with the same 88
(A, B) pairs and candidates asked through the templates of --templates (steering_templates_extra.csv),
appended to templates.json as indices 25, 26, ... Nothing else changes: the fitted charts, the
archive checks and the manifest's eval_templates (0 and 18, the ones the fit excludes) stay as they
are. The extra templates are not in the activation archive, so the fit never saw them either.
prepare_pair_holdout.py then refits the per-pair maps exactly as for templates 0 and 18.

Usage: prepare_template_battery.py --bundle <prepared_x_heldout_typed> --templates <csv> --out <dir>
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--templates', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists')
    extra = [s.strip().strip('"') for s in a.templates.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert all(t.count('X') == 1 for t in extra), extra
    shutil.copytree(a.bundle, a.out)
    meta = json.loads((a.bundle / 'manifest.json').read_text())
    specs = json.loads((a.bundle / 'spec.json').read_text())
    catalog = json.loads((a.bundle / 'catalog.json').read_text())
    templates = json.loads((a.bundle / 'templates.json').read_text())
    first = meta['eval_templates'][0]
    base = [s for s in specs if s['template'] == first]
    assert len(base) == len({s['source'] for s in specs}) == 88
    ids = list(range(len(templates), len(templates) + len(extra)))
    new = []
    for t, text in zip(ids, extra):
        for s in base:
            new.append(dict(s, id=len(new), template=t,
                            prompt=text.replace('X', catalog[s['source']]['name']),
                            target_prompt=text.replace('X', catalog[s['target']]['name'])))
    meta['battery_templates'] = ids
    meta['battery_note'] = ('Extra evaluation templates (rebuttal/steering_templates_extra.csv); the fit '
                            'still excludes eval_templates, and the extra templates are not in the archive.')
    (a.out / 'templates.json').write_text(json.dumps(templates + extra, indent=2))
    (a.out / 'spec.json').write_text(json.dumps(new, indent=2))
    (a.out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(status='prepared', out=str(a.out), templates=dict(zip(ids, extra)), specs=len(new))))


if __name__ == '__main__':
    main()
