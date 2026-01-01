"""Fit harmonic sky charts for one model and freeze the orthogonal-pair battery.

What this freezes, for any checkpoint in the `models_multi` registry:

  * layers are the sky_r2 >= frac * max set of `<model>_sky_r2_16.csv`, minus
    the final hidden state, which is post-final-norm and never hooked;
  * every catalogue constellation is a source once, and its destination is
    maximally orthogonal to it: B minimises (u_A . u_B)^2;
  * each object's centroid is averaged over prompt templates before the fit;
  * the l=1 and l=2 blocks of the fitted map are compared by principal angles.

By default nothing is held out: the fit uses every template, including the two
that are evaluated, so the reported fit quality is in-sample and describes the
fit rather than estimating generalisation. `--fit-templates exclude-eval` drops
the evaluated templates from the fit and makes it a held-out measurement;
`--fit-scope typed` fits the harmonic block across all object types at once,
giving each type its own intercept while the chart keeps the constellation's.
Both make the fit harder and are reported separately.

Principal angles: within the joint degree-2 fit F(u)=b+W phi(u), the image of
the dipole block col(W[:, :3]) and of the quadrupole block col(W[:, 3:]) are
both subspaces of the K-dimensional PCA space. With orthonormal bases Q1, Q2,
the singular values of Q1.T @ Q2 are the cosines of the principal angles: 1
means a shared direction, 0 means mutually orthogonal. Three values exist
because the dipole image is three-dimensional. `separate` repeats this for maps
fitted to each degree alone, the quantity the sky-subspace ablation reports.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import angle, digest, fit_forward, harmonics, unit
from models_multi import ARCHIVE_DIR, MODELS, PROMPTS, R2_DIR, REGIME, archive, r2_table


def orthonormal(M):
    q, s, _ = np.linalg.svd(M, full_matrices=False)
    return q[:, : int(np.sum(s > s[0] * 1e-10))]


def principal_cosines(A, B):
    """Cosines of the principal angles between col(A) and col(B), descending."""
    s = np.linalg.svd(orthonormal(A).T @ orthonormal(B), compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def select_layers(path, frac):
    with open(path, newline='') as f:
        rows = {int(r['layer']): float(r['r2']) for r in csv.DictReader(f)}
    rmax = max(rows.values())
    rarg = max(rows, key=rows.get)
    final = max(rows)                        # post-final-norm hidden state
    chosen = [l for l, r in sorted(rows.items()) if r >= frac * rmax and l != final]
    return chosen, rmax, rarg, final


def fit_typed(u, z, types, degree, anchor='constellation'):
    """Shared harmonic coefficients over all objects, one intercept per type.

    Returns (intercept of `anchor`, W). Stars and deep-sky objects then help
    estimate the direction map without forcing their own mean onto the
    constellation prompts the battery actually steers.
    """
    phi = harmonics(u)[:, :3 if degree == 1 else 8]
    levels = sorted(set(types))
    dummies = np.column_stack([(types == t).astype(float) for t in levels])
    design = np.column_stack([dummies, phi])
    coef, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Rank-deficient typed harmonic design')
    return coef[levels.index(anchor)], coef[len(levels):].T


def fit_block(u, z, columns):
    """OLS of z on an intercept plus the selected harmonic columns; returns (K, len(columns))."""
    phi = harmonics(u)[:, columns]
    design = np.column_stack([np.ones(len(phi)), phi])
    coef, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Rank-deficient harmonic design')
    return coef[1:].T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--archive-dir', default=ARCHIVE_DIR)
    ap.add_argument('--r2-dir', default=R2_DIR)
    ap.add_argument('--prompts', type=Path, default=Path(PROMPTS))
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--frac', type=float, default=0.8)
    ap.add_argument('--degree', type=int, default=2, choices=[1, 2],
                    help='1 edits only the l=1 dipole (3 dimensions); 2 edits l=1+l=2 (8)')
    ap.add_argument('--fit-templates', choices=['all', 'exclude-eval'], default='all',
                    help='"exclude-eval" averages centroids over every template EXCEPT the '
                         'two the battery evaluates on, making the evaluation template-held-out')
    ap.add_argument('--fit-scope', choices=['constellations', 'typed'], default='constellations',
                    help='"typed" fits the harmonic coefficients on ALL catalogue objects with a '
                         'separate intercept per type (constellation / star / other), so stars and '
                         'deep-sky objects inform the direction map while keeping their own offset; '
                         'the chart keeps the constellation intercept, since the prompts name '
                         'constellations')
    ap.add_argument('--templates', type=int, nargs='+', default=[0, 18])
    args = ap.parse_args()
    spec_model = MODELS[args.model]

    layers, rmax, rarg, final = select_layers(r2_table(args.model, args.r2_dir), args.frac)
    if not layers:
        ap.error('No layer clears the threshold')
    args.out.mkdir(parents=True, exist_ok=False)

    arch = Path(archive(args.model, args.archive_dir))
    d = np.load(arch)
    ids = d['obj_ids']
    y = unit(d['Yunit'])
    types, names = d['types'].astype(str), d['names'].astype(str)
    pca = d['pca'].astype(np.float64)
    means, stds, centers, components = [d[k] for k in ['feat_mean', 'feat_std', 'pca_mean', 'components']]
    n_states = pca.shape[1]
    if final != n_states - 1:
        raise SystemExit(f'r2 table ends at {final} but archive holds {n_states} hidden states')

    templates = [s.strip().strip('"') for s in args.prompts.read_text().splitlines() if s.strip()]
    catalog = [dict(id=i, name=str(n), type=str(types[i]), unit=y[i].tolist()) for i, n in enumerate(names)]
    const = [c['id'] for c in catalog if c['type'] == 'constellation']

    specs = []
    for a in const:
        b = min([i for i in const if i != a], key=lambda i: float((y[a] @ y[i]) ** 2))
        available = [i for i in const if i not in (a, b)]
        ca = min(available, key=lambda i: float(angle(y[a], y[i])))
        cb = min([i for i in available if i != ca], key=lambda i: float(angle(y[b], y[i])))
        for t in args.templates:
            specs.append(dict(
                id=len(specs), source=int(a), target=int(b), template=int(t),
                candidates=[int(ca), int(cb)],
                prompt=templates[t].replace('X', catalog[a]['name']),
                target_prompt=templates[t].replace('X', catalog[b]['name']),
                separation_deg=float(angle(y[a], y[b])),
                abs_cosine=float(abs(y[a] @ y[b])),
                source_heldout=0, target_heldout=0))

    pr = d['pr_ids']
    keep = (np.ones(len(ids), bool) if args.fit_templates == 'all'
            else ~np.isin(pr, args.templates))
    obj_types = np.array([c['type'] for c in catalog])
    angles, metrics = [], []
    for layer in layers:
        cent = np.stack([pca[(ids == i) & keep, layer].mean(0) for i in range(len(y))])
        z = cent[const, : args.k]
        u = y[const]
        if args.fit_scope == 'typed':
            b, fitted = fit_typed(y, cent[:, : args.k], obj_types, args.degree)
        else:
            b, fitted = fit_forward(u, z, args.degree)
        # A degree-1 chart keeps the 8-column layout with the quadrupole block
        # zeroed, so col(W) is the 3-dimensional dipole image and core.Chart
        # edits exactly that subspace with no special-casing.
        W = fitted if args.degree == 2 else np.hstack([fitted, np.zeros((fitted.shape[0], 5))])
        # Angles always describe the degree-2 fit, so they stay comparable
        # across runs; they are a property of the model, not of the edit.
        _, W2 = fit_forward(u, z, 2)
        joint = principal_cosines(W2[:, :3], W2[:, 3:])
        separate = principal_cosines(fit_block(u, z, [0, 1, 2]), fit_block(u, z, [3, 4, 5, 6, 7]))
        angles.append(dict(model=args.model, layer=layer,
                           **{f'joint_cos{i + 1}': float(v) for i, v in enumerate(joint)},
                           **{f'separate_cos{i + 1}': float(v) for i, v in enumerate(separate)}))
        pred = harmonics(u) @ W.T + b
        metrics.append(dict(layer=layer, k=args.k, degree=args.degree, rank=int(np.linalg.matrix_rank(W)),
                            insample_r2=float(1 - np.sum((z - pred) ** 2) / np.sum((z - z.mean(0)) ** 2)),
                            insample_rmse=float(np.sqrt(np.mean((z - pred) ** 2)))))
        np.savez_compressed(args.out / f'chart_L{layer}_K{args.k}.npz',
                            mu=means[layer].astype(float), sigma=stds[layer].astype(float),
                            pca_mean=centers[layer].astype(float),
                            V=components[layer, : args.k].astype(float), b=b, W=W)

    with (args.out / 'harmonic_angles.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(angles[0]))
        w.writeheader()
        w.writerows(angles)
    medians = {f'{kind}_cos{i}': float(np.median([a[f'{kind}_cos{i}'] for a in angles]))
               for kind in ['joint', 'separate'] for i in [1, 2, 3]}

    checks = []
    for row in specs[:2]:
        hit = np.flatnonzero((ids == row['source']) & (d['pr_ids'] == row['template']))
        checks.append(dict(prompt=row['prompt'], scores={str(l): pca[int(hit[0]), l].tolist() for l in layers}))

    metadata = dict(
        model=spec_model['repo'], model_key=args.model,
        revision=spec_model['revision'], prompt_regime=REGIME,
        device_map=spec_model['device_map'], keep_experts_impl=spec_model['keep_experts_impl'],
        layers=layers, ks=[args.k], layer_groups=[layers], layer_mode='joint',
        degree=args.degree, edited_dimensions=3 if args.degree == 1 else 8,
        max_hidden_state=final - 1, final_hidden_state=final,
        r2_max=rmax, r2_argmax=rarg, r2_frac=args.frac,
        archive=str(arch), archive_sha256=digest(arch), prompts_sha256=digest(args.prompts),
        train_objects=const if args.fit_scope == 'constellations' else list(range(len(y))),
        heldout_objects=[],
        heldout_templates=args.templates if args.fit_templates == 'exclude-eval' else [],
        eval_templates=args.templates,
        fit_templates=args.fit_templates, fit_scope=args.fit_scope,
        fitting=(
            f"OLS of top-{args.k} PCA centroids on real l<={args.degree} harmonics. Centroids average "
            + ("every prompt template, so the evaluation templates are IN the fit"
               if args.fit_templates == 'all' else
               f"every prompt template EXCEPT the evaluated {args.templates}, so the evaluation is "
               f"template-held-out")
            + "; objects: "
            + ("constellations only, one shared intercept"
               if args.fit_scope == 'constellations' else
               "ALL catalogue objects with one intercept per type (constellation/star/other) and "
               "shared harmonic coefficients; the chart keeps the constellation intercept")
            + f". No object is held out. The replaced subspace is the "
              f"{3 if args.degree == 1 else 8}-dimensional image of those harmonics in top-{args.k} "
              f"PCA space."),
        pca_policy='Frozen globally fitted archive transform',
        metric='Euclidean in standardized neuron/PCA coordinates; all omitted PC components retained',
        arms=['untouched', 'self', 'target'],
        patch='last prompt token and each subsequent active input token, every selected layer per pass',
        pair_policy='Every catalogue constellation is a source; destination minimises the squared '
                    'cosine between centroid unit vectors. No outcome selection.',
        candidate_policy='Nearest constellation center to each endpoint, excluding both endpoints',
        harmonic_angle_medians=medians)
    (args.out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    (args.out / 'spec.json').write_text(json.dumps(specs, indent=2))
    (args.out / 'catalog.json').write_text(json.dumps(catalog, indent=2))
    (args.out / 'templates.json').write_text(json.dumps(templates, indent=2))
    (args.out / 'archive_checks.json').write_text(json.dumps(checks, indent=2))
    (args.out / 'fit_metrics.json').write_text(json.dumps(metrics, indent=2))

    seps = np.array([s['separation_deg'] for s in specs])
    print(json.dumps(dict(
        status='prepared', model=args.model, degree=args.degree,
        fit_templates=args.fit_templates, fit_scope=args.fit_scope,
        edited_dimensions=3 if args.degree == 1 else 8, repo=spec_model['repo'], out=str(args.out),
        layers=[layers[0], layers[-1]], n_layers=len(layers), dropped_final_state=final,
        r2_max=round(rmax, 4), sources=len(const), specs=len(specs), conditions=len(specs) * 3,
        separation_deg=dict(min=float(seps.min()), median=float(np.median(seps)), max=float(seps.max())),
        insample_r2=dict(min=round(min(m['insample_r2'] for m in metrics), 4),
                         max=round(max(m['insample_r2'] for m in metrics), 4)),
        harmonic_angle_medians={k: round(v, 4) for k, v in medians.items()}), indent=2), flush=True)


if __name__ == '__main__':
    main()
