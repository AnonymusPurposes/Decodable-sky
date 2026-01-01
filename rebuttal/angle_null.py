#!/usr/bin/env python
"""Null distribution of the median angular error on the real activations, at every layer.

Replaces the "protocol_null" rows of baseline_angle.csv, which used the true sky vectors as the
activations (3 regressors) and so did not reproduce the noise level of the real read-out.

Here the read-out is exactly the 80/20 one of split_pca.py (in-split standardization and PCA,
OLS with intercept on the top-k scores of individual prompts, test objects prompt-averaged), on
the first --n-splits splits. The object -> sky-vector assignment is permuted (whole objects, so
an object's prompts move together) and the test repeated. The prediction on test objects is a
fixed linear map of the training labels, so every permutation reuses it (as in split_perm.py).

Per layer it writes
  obs_ang, obs_r2           the observed median over splits (median angle over test objects)
  null_ang_p50/p05/p95      the same statistic under permutation: median and 5-95% of permutations
  null_r2_p50
  const_ang                 intercept-only prediction (the mean sky direction of the training
                            objects for every test object), median over splits: the floor that
                            the uneven placement of the objects alone gives

Usage: angle_null.py --model <m> --raw-dir <d> --split-dir <d> --out <d> [--n-splits 20 --n-perm 1000]
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from split_pca import load_sky, pca_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    tests = np.load(os.path.join(a.split_dir, a.model + ".npz"))["test_objects"][: a.n_splits]
    raw = np.load(os.path.join(a.raw_dir, a.model + "_raw.npy"), mmap_mode="r")
    n, nl, _ = raw.shape
    sky = torch.as_tensor(load_sky(a.objects), dtype=torch.float64, device=dev)
    no = sky.shape[0]
    obj = torch.as_tensor(np.tile(np.arange(no), n // no), device=dev)
    X = torch.from_numpy(np.array(raw)).to(dev)

    t0 = time.time()
    S = {}
    for s in range(a.n_splits):
        te = torch.as_tensor(tests[s], device=dev)
        te_mask = torch.isin(obj, te)
        tr = torch.tensor([o for o in range(no) if o not in set(tests[s].tolist())], device=dev)
        for L in range(nl):
            x = X[:, L].to(torch.float64)
            ztr, (zte,) = pca_fit(x[~te_mask], [x[te_mask]], a.k)
            d = torch.cat([torch.ones(len(ztr), 1, dtype=ztr.dtype, device=dev), ztr[:, :a.k]], 1)
            obj_tr = obj[~te_mask]
            Xsum = torch.stack([d[obj_tr == o].sum(0) for o in tr])
            zt = torch.stack([zte[obj[te_mask] == o][:, :a.k].mean(0) for o in te])
            q = torch.cat([torch.ones(len(te), 1, dtype=zt.dtype, device=dev), zt], 1)
            S[s, L] = (tr, te, q @ torch.linalg.solve(d.T @ d, Xsum.T))
    print(f"[{a.model}] prediction maps built {time.time() - t0:.0f}s", flush=True)

    def stats(Y):
        """Y (B, no, 3) -> (median-over-splits median test angle, median-over-splits R^2), each (B, nl)"""
        A = torch.zeros(Y.shape[0], nl, dtype=torch.float64, device=dev)
        R = torch.zeros_like(A)
        for L in range(nl):
            ang, r2 = [], []
            for s in range(a.n_splits):
                tr, te, W = S[s, L]
                pred = W @ Y[:, tr]
                t = Y[:, te]
                cos = (torch.nn.functional.normalize(pred, dim=2) * t).sum(2).clamp(-1, 1)
                ang.append(torch.rad2deg(torch.arccos(cos)).median(1).values)
                sse = ((t - pred) ** 2).sum(1)
                sst = ((t - t.mean(1, keepdim=True)) ** 2).sum(1)
                r2.append((1 - sse / sst).mean(1))
            A[:, L] = torch.stack(ang, 1).median(1).values
            R[:, L] = torch.stack(r2, 1).median(1).values
        return A, R

    oA, oR = (v[0].cpu().numpy() for v in stats(sky[None]))
    const = []
    for s in range(a.n_splits):
        tr, te, _ = S[s, 0]
        m = torch.nn.functional.normalize(sky[tr].mean(0), dim=0)
        const.append(float(torch.rad2deg(torch.arccos((sky[te] @ m).clamp(-1, 1))).median()))
    const = float(np.median(const))

    rng = np.random.default_rng(a.seed)
    nA, nR = [], []
    for done in range(0, a.n_perm, a.batch):
        b = min(a.batch, a.n_perm - done)
        perms = torch.as_tensor(np.stack([rng.permutation(no) for _ in range(b)]), device=dev)
        A, R = stats(sky[perms])
        nA.append(A.cpu().numpy()); nR.append(R.cpu().numpy())
    nA, nR = np.concatenate(nA), np.concatenate(nR)
    print(f"[{a.model}] {a.n_perm} permutations {time.time() - t0:.0f}s", flush=True)

    rows = [dict(model=a.model, layer=L, k=a.k, n_splits=a.n_splits, n_perm=a.n_perm,
                 obs_ang=round(float(oA[L]), 2), null_ang_p50=round(float(np.median(nA[:, L])), 2),
                 null_ang_p05=round(float(np.percentile(nA[:, L], 5)), 2),
                 null_ang_p95=round(float(np.percentile(nA[:, L], 95)), 2),
                 obs_r2=round(float(oR[L]), 4), null_r2_p50=round(float(np.median(nR[:, L])), 4),
                 const_ang=round(const, 2)) for L in range(nl)]
    with open(os.path.join(a.out, a.model + "_angle_null.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.model, flush=True)


if __name__ == "__main__":
    main()
