#!/usr/bin/env python
"""Full-dimensional version of the neural side of metric_gain_typed.py / vs_choice_intervals.py (cluster, GPU):
no common 128-PCA basis. Per layer, from the raw residual stream (n_rows, hidden):

  split s (100, the stratified splits of metric_gain_typed.py, rng 7000 + s): per-feature standardization
      fitted on the training objects' rows, each type centred on its training rows' mean, top-8 PCA of the
      training rows, scores of all rows scaled so PC1 has SD 1 on the training rows, template-averaged
      centroids of all 188 objects                                          -> cent_split (s, L, 188, 8)
  resample b (100, the within-type object bootstrap of vs_choice_intervals.py, rng 20261112): the same
      on the drawn objects' rows (a drawn-twice object enters twice), centroids per draw -> cent_boot (b, L, 188, 8)
  all objects: the same on all rows                                         -> cent_full (L, 188, 8)

Top-8 PCA by randomized SVD (torch.svd_lowrank, q=24, niter=6) of the centred training rows; checked against
an exact eigendecomposition on the first split of every 10th layer (printed).
Output <out>/<model>_fullcent.npz.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch


def splits(types, n, frac=0.2):
    out = []
    for s in range(n):
        rng = np.random.default_rng(7000 + s)
        out.append(np.sort(np.concatenate([rng.choice(np.where(types == t)[0], int(round(frac * (types == t).sum())), replace=False)
                                           for t in np.unique(types)])))
    return out


def all_draws(tid, nboot, seed):
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(np.where(tid == t)[0], (tid == t).sum()) for t in np.unique(tid)]) for _ in range(nboot)]


def top8(Z, rows_fit, rtype_fit, check=False):
    """Z: (n, d) float32 on GPU, the rows to fit on (already selected); returns scores of those rows (n, 8)
    after standardization and type-centring fitted on them, plus the transform to apply elsewhere"""
    mu, sd = Z.mean(0), Z.std(0, unbiased=False) + 1e-6
    X = (Z - mu) / sd
    tm = {}
    for t in torch.unique(rtype_fit):
        k = rtype_fit == t
        tm[int(t)] = X[k].mean(0)
        X[k] -= tm[int(t)]
    U, S, V = torch.svd_lowrank(X, q=24, niter=6, M=None)
    V8 = V[:, :8]
    if check:
        G = X @ X.T
        ev = torch.linalg.eigvalsh(G.double()).flip(0)[:8].float()
        Ue = torch.linalg.eigh(G.double())[1].flip(1)[:, :8].float()
        Ve = X.T @ Ue; Ve /= Ve.norm(dim=0, keepdim=True)
        ov = torch.linalg.svdvals(Ve.T @ V8).min().item()
        print(f"   check: max rel eig err {((S[:8] ** 2 - ev).abs() / ev).max().item():.2e}, min subspace cos {ov:.6f}", flush=True)
    P = X @ V8
    sc = P[:, 0].std(unbiased=False)
    return mu, sd, tm, V8, sc


def project(Z, rtype, mu, sd, tm, V8, sc):
    X = (Z - mu) / sd
    for t, m in tm.items():
        X[rtype == t] -= m
    return (X @ V8) / sc


def centroids(P, ob, n):
    C = torch.zeros(n, 8, device=P.device)
    C.index_add_(0, ob, P)
    return C / torch.bincount(ob, minlength=n)[:, None].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True, help="pca_repro/<model>_pca128.npz (obj_ids, types)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--nboot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    dev = "cuda"
    m = np.load(a.meta, allow_pickle=True)
    obj, types = m["obj_ids"], m["types"].astype(str)
    nk = len(types)
    tcode = {t: i for i, t in enumerate(np.unique(types))}
    tid_obj = np.array([tcode[t] for t in types])
    R = np.load(a.raw)                                      # whole file into RAM (node has TBs)
    assert R.shape[0] == len(obj), (R.shape, len(obj))
    nl = R.shape[1]
    S = splits(types, a.splits)
    D = all_draws(types, a.nboot, a.seed)
    obj_t = torch.tensor(obj, device=dev)
    rtype = torch.tensor(tid_obj[obj], device=dev)
    rows_of = [np.where(obj == o)[0] for o in range(nk)]
    cs = np.zeros((a.splits, nl, nk, 8), np.float32)
    cb = np.zeros((a.nboot, nl, nk, 8), np.float32)
    cf = np.zeros((nl, nk, 8), np.float32)
    t0 = time.time()
    for L in range(nl):
        Z = torch.tensor(np.ascontiguousarray(R[:, L, :]), device=dev, dtype=torch.float32)
        mu, sd, tm, V8, sc = top8(Z.clone(), None, rtype)
        cf[L] = centroids(project(Z, rtype, mu, sd, tm, V8, sc), obj_t, nk).cpu().numpy()
        for s, te in enumerate(S):
            tr = np.ones(nk, bool); tr[te] = False
            rm = torch.tensor(tr[obj], device=dev)
            mu, sd, tm, V8, sc = top8(Z[rm].clone(), None, rtype[rm], check=(s == 0 and L % 10 == 0))
            cs[s, L] = centroids(project(Z.clone(), rtype, mu, sd, tm, V8, sc), obj_t, nk).cpu().numpy()
        for b, dr in enumerate(D):
            rows = np.concatenate([rows_of[k] for k in dr])
            ro = torch.tensor(np.concatenate([np.full(len(rows_of[k]), j) for j, k in enumerate(dr)]), device=dev)
            ri = torch.tensor(rows, device=dev)
            Zb = Z[ri]
            mu, sd, tm, V8, sc = top8(Zb.clone(), None, rtype[ri])
            cb[b, L] = centroids(project(Zb, rtype[ri], mu, sd, tm, V8, sc), ro, nk).cpu().numpy()
        print(f"[{a.model}] layer {L + 1}/{nl}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"{a.model}_fullcent.npz"), cent_split=cs, cent_boot=cb, cent_full=cf,
             tests=np.array(S), draws=np.array(D), types=types, obj_ids=obj)
    print("DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
