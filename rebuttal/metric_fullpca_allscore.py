#!/usr/bin/env python
"""All coordinate choices for the full-dimensional centroids (metric_fullpca_cent.py): every split and layer,
the sphere alone and each chart@F/c (3 frames x 12 cuts) and cylinder@F alone and joined with the sphere,
linmap held-out R^2 (metric_sensitivity.score), exactly as metric_gain_typed.py but on the full-dimensional
centroids (or, with --key, the k-dimensional / unreduced centroids of metric_kdim_cent.py). Writes <out>/<model>.npz in metric_gain_typed.py's format, so its summarize() makes the CSVs (picks by
all-layer mean, min / max over choices). Runs without astropy: the coordinate blocks come from --cand
(an npz of metric_sensitivity.candidates() made where astropy is available, --make-cand).
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from metric_sensitivity import LAMS, ridge, score


def linmap_fast(C, cent, y, tr_obj, iu, ju, pair_tr, fold):
    """metric_sensitivity.score(C, cent, ..., "linmap") with the squared distances y of the centroids passed in
    (computed once per split and layer) and the predicted distances computed in coordinate space:
    P = b0 + Cz B, so |P_i - P_j|^2 = (Cz_i - Cz_j) B B^T (Cz_i - Cz_j)^T (p x p, p <= 6) instead of over the
    centroid dimensions. Algebraically identical; same lambda grid, folds, tie-breaking and calibration."""
    m, s = C[tr_obj].mean(0), C[tr_obj].std(0) + 1e-12
    Cz = (C - m) / s
    tro = np.where(tr_obj)[0]
    best, lam = np.inf, LAMS[0]
    for la in LAMS:
        sse = 0.0
        for g in np.unique(fold):
            keep = fold != g
            tr, te = tro[keep[tro]], tro[~keep[tro]]
            b0, B = ridge(Cz[tr], cent[tr], la)
            sse += ((cent[te] - b0 - Cz[te] @ B) ** 2).sum()
        if sse < best:
            best, lam = sse, la
    _, B = ridge(Cz[tr_obj], cent[tr_obj], lam)
    D = Cz[iu] - Cz[ju]
    d = ((D @ (B @ B.T)) * D).sum(1)
    X = np.column_stack([np.ones(pair_tr.sum()), d[pair_tr]])
    a0, s0 = np.linalg.lstsq(X, y[pair_tr], rcond=None)[0]
    yt = y[~pair_tr]
    return 1.0 - ((yt - a0 - s0 * d[~pair_tr]) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()

MODELS = ["qwen235b", "mistrallarge123b", "llama33_70b", "qwen32b", "mixtral8x22b", "glm45air", "gptoss120b"]
_C = {}


def task(job):
    model, cent_dir, cand_path, s0, s1, key, fast = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if model not in _C:
        _C.clear()
        if key:                                     # metric_kdim_cent.py output, memory-mapped
            _C[model] = (np.load(os.path.join(cent_dir, f"{model}_{key}.npy"), mmap_mode="r"),
                         np.load(os.path.join(cent_dir, f"{model}_ktests.npy")))
        else:
            F = np.load(os.path.join(cent_dir, model + "_fullcent.npz"), allow_pickle=True)
            _C[model] = (F["cent_split"], F["tests"])
    CS, tests = _C[model]
    cand = np.load(cand_path, allow_pickle=True)
    names = [str(v) for v in cand["names"]]
    sph = cand["sphere@equatorial"]
    nl, nk = CS.shape[1], CS.shape[2]
    iu, ju = np.triu_indices(nk, 1)
    SPH = np.full((s1 - s0, nl), np.nan); AL = np.full((s1 - s0, nl, len(names)), np.nan); JO = AL.copy()
    for k, s in enumerate(range(s0, s1)):
        tr = np.ones(nk, bool); tr[tests[s]] = False
        pair_tr = tr[iu] & tr[ju]
        fold = np.random.default_rng(1000 + s).integers(0, 4, nk)
        for L in range(nl):
            c = np.asarray(CS[s, L], dtype=np.float64)
            if fast:
                G = c @ c.T
                y = np.diag(G)[iu] + np.diag(G)[ju] - 2 * G[iu, ju]
                sc = lambda C: linmap_fast(C, c, y, tr, iu, ju, pair_tr, fold)
            else:
                sc = lambda C: score(C, c, tr, iu, ju, pair_tr, fold, "linmap")
            SPH[k, L] = sc(sph)
            for j, n in enumerate(names):
                AL[k, L, j] = sc(cand[n])
                JO[k, L, j] = sc(np.column_stack([sph, cand[n]]))
    return model, s0, SPH, AL, JO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cent")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out")
    ap.add_argument("--objects", help="with --make-cand")
    ap.add_argument("--make-cand", action="store_true")
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--key", default="", help="k8 | k32 | k128 | kfull: read metric_kdim_cent.py's <model>_<key>.npy")
    ap.add_argument("--fast", action="store_true", help="linmap_fast (identical up to rounding; checked by --verify)")
    ap.add_argument("--verify", action="store_true", help="compare linmap_fast with score() on a sample, then exit")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    if a.make_cand:
        from metric_gain_typed import NAMES
        from metric_sensitivity import candidates
        rows = list(csv.DictReader(open(a.objects, newline="", encoding="utf-8")))
        c = candidates(np.array([float(r["l_ra_deg"]) for r in rows]), np.array([float(r["a_dec_deg"]) for r in rows]))
        np.savez(a.cand, names=np.array(NAMES), **{n: c[n] for n in NAMES + ["sphere@equatorial"]})
        return
    if a.verify:
        cand = np.load(a.cand, allow_pickle=True)
        names = [str(v) for v in cand["names"]]
        sph = cand["sphere@equatorial"]
        for m in a.models:
            CS = np.load(os.path.join(a.cent, f"{m}_{a.key}.npy"), mmap_mode="r")
            tests = np.load(os.path.join(a.cent, f"{m}_ktests.npy"))
            nl, nk = CS.shape[1], CS.shape[2]
            iu, ju = np.triu_indices(nk, 1)
            rng, worst, n = np.random.default_rng(0), 0.0, 0
            for s in rng.choice(len(tests), 3, replace=False):
                tr = np.ones(nk, bool); tr[tests[s]] = False
                pair_tr = tr[iu] & tr[ju]
                fold = np.random.default_rng(1000 + s).integers(0, 4, nk)
                for L in rng.choice(np.arange(1, nl), 4, replace=False):
                    c = np.asarray(CS[s, L], dtype=np.float64)
                    G = c @ c.T
                    y = np.diag(G)[iu] + np.diag(G)[ju] - 2 * G[iu, ju]
                    for nm in rng.choice(names, 12, replace=False):
                        for C in (cand[nm], np.column_stack([sph, cand[nm]])):
                            r0 = score(C, c, tr, iu, ju, pair_tr, fold, "linmap")
                            r1 = linmap_fast(C, c, y, tr, iu, ju, pair_tr, fold)
                            worst, n = max(worst, abs(r0 - r1)), n + 1
            print(f"verify {m} {a.key}: {n} scores, max |fast - score| = {worst:.2e}", flush=True)
        return
    os.makedirs(a.out, exist_ok=True)
    models = a.models or MODELS
    names = [str(v) for v in np.load(a.cand)["names"]]
    jobs = [(m, a.cent, a.cand, s, min(s + a.chunk, a.splits), a.key, a.fast) for m in models for s in range(0, a.splits, a.chunk)]
    parts = {m: [] for m in models}
    nch = len(range(0, a.splits, a.chunk))
    with Pool(a.procs) as p:
        for m, s0, S, A, J in p.imap_unordered(task, jobs):
            parts[m].append((s0, S, A, J))
            if len(parts[m]) == nch:
                parts[m].sort(key=lambda t: t[0])
                tests = (np.load(os.path.join(a.cent, f"{m}_ktests.npy")) if a.key else
                         np.load(os.path.join(a.cent, m + "_fullcent.npz"))["tests"])
                np.savez(os.path.join(a.out, m + ".npz"), sphere=np.concatenate([t[1] for t in parts[m]]),
                         alone=np.concatenate([t[2] for t in parts[m]]), joined=np.concatenate([t[3] for t in parts[m]]),
                         names=np.array(names), tests=tests)
                parts[m] = None
                print("done", m, flush=True)


if __name__ == "__main__":
    main()
