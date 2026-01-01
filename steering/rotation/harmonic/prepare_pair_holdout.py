"""Hold the steering pair's own constellations out of the fitted map.

Takes a finished heldout_typed bundle (prepare_multi.py with --fit-templates exclude-eval
--fit-scope typed) and, for every source constellation A and its destination B, refits the map
F(u) = b + W phi(u) with A and B removed from the fit. The evaluation prompts name A, the edit
writes the prediction for B, and neither object contributed to the map used on that pair. The two
evaluation templates were already excluded from the fit.

The standardization and PCA basis stay those of the archive (fitted on all objects); only the map
inside the top-K space is refitted. Output: a copy of the bundle plus, per layer,
pair_charts_L<l>_K<k>.npz with b (n_specs, K) and W (n_specs, K, 8), read by run_pair_holdout.py.

Usage: prepare_pair_holdout.py --bundle <prepared_x> --out <dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from core import unit
from prepare_multi import fit_typed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists')
    meta = json.loads((a.bundle / 'manifest.json').read_text())
    if meta['fit_templates'] != 'exclude-eval' or meta['fit_scope'] != 'typed':
        raise SystemExit('expected a heldout_typed bundle')
    shutil.copytree(a.bundle, a.out)
    specs = json.loads((a.bundle / 'spec.json').read_text())
    k = meta['ks'][0]
    d = np.load(meta['archive'])
    ids, y = d['obj_ids'], unit(d['Yunit'])
    types = d['types'].astype(str)
    pca = d['pca'].astype(np.float64)
    keep = ~np.isin(d['pr_ids'], meta['eval_templates'])
    diag = []
    for layer in meta['layers']:
        cent = np.stack([pca[(ids == i) & keep, layer].mean(0)[:k] for i in range(len(y))])
        base = np.load(a.bundle / f'chart_L{layer}_K{k}.npz')
        cache = {}
        bs, ws = [], []
        for s in specs:
            pair = (s['source'], s['target'])
            if pair not in cache:
                m = np.ones(len(y), bool)
                m[list(pair)] = False
                b, w = fit_typed(y[m], cent[m], types[m], meta['degree'])
                if meta['degree'] == 1:
                    w = np.hstack([w, np.zeros((w.shape[0], 5))])
                cache[pair] = (b, w)
            bs.append(cache[pair][0])
            ws.append(cache[pair][1])
        bs, ws = np.stack(bs), np.stack(ws)
        np.savez_compressed(a.out / f'pair_charts_L{layer}_K{k}.npz', b=bs, W=ws)
        rel = float(np.median(np.linalg.norm(ws - base['W'], axis=(1, 2)) / np.linalg.norm(base['W'])))
        diag.append(dict(layer=layer, median_relative_change_of_W=round(rel, 5)))
    meta['pair_holdout'] = True
    meta['fitting'] += ' Each evaluated pair uses a map refitted without its own source and destination.'
    (a.out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    (a.out / 'pair_holdout_diagnostics.json').write_text(json.dumps(diag, indent=2))
    print(json.dumps(dict(status='prepared', out=str(a.out), layers=len(diag),
                          median_relative_change_of_W=float(np.median([x['median_relative_change_of_W'] for x in diag])))))


if __name__ == '__main__':
    main()
