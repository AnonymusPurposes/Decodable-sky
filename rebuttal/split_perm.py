#!/usr/bin/env python
"""Empirical p-values for the 80/20 sky read-out with PCA fitted on training objects only.

Same control as correlations_split.py: the object -> sky-vector assignment is permuted and the
test repeated. PCA is unsupervised, so it does not depend on the labels: for each split and layer
the OLS prediction on the test objects is a fixed linear map of the training objects' labels, and
all permutations can reuse it.

The first --n-splits splits of split_pca.py are used (default 11, as in the paper).
  fixed layer  the statistic is the median test R^2 over splits at the layer reported in Table 5.
  selection    the observed statistic is the nested one (layer chosen inside the training objects,
               per split). The null uses, for each permutation, the best layer over ALL layers
               (max over layers of the median test R^2), which is at least as favourable to the
               null as any selection rule, so this p-value is conservative.
p = (1 + hits) / (1 + N), so the smallest reportable value is 1 / (1 + N).

Usage: split_perm.py --model <m> --raw-dir <d> --split-dir <d> --best-layer <L> --out <d>
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
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=11)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    sk = np.load(os.path.join(a.split_dir, a.model + ".npz"))
    if int(sk["done"]) < a.n_splits:
        raise SystemExit("split_pca has not reached the first splits yet")
    tests = sk["test_objects"][: a.n_splits]
    j = list(sk["ks"]).index(a.k)
    raw = np.load(os.path.join(a.raw_dir, a.model + "_raw.npy"), mmap_mode="r")
    n, nl, _ = raw.shape
    sky = torch.as_tensor(load_sky(a.objects), dtype=torch.float64, device=dev)
    no = sky.shape[0]
    obj = torch.as_tensor(np.tile(np.arange(no), n // no), device=dev)
    X = torch.from_numpy(np.array(raw)).to(dev)

    t0 = time.time()
    S = {}                                    # (split, layer) -> (train ids, test ids, weights n_test x n_train)
    for s in range(a.n_splits):
        te = torch.as_tensor(tests[s], device=dev)
        te_mask = torch.isin(obj, te)
        tr = torch.tensor([o for o in range(no) if o not in set(tests[s].tolist())], device=dev)
        for L in range(nl):
            x = X[:, L].to(torch.float64)
            ztr, (zte,) = pca_fit(x[~te_mask], [x[te_mask]], a.k)
            d = torch.cat([torch.ones(len(ztr), 1, dtype=ztr.dtype, device=dev), ztr[:, :a.k]], 1)
            G = d.T @ d
            obj_tr = obj[~te_mask]
            Xsum = torch.stack([d[obj_tr == o].sum(0) for o in tr])
            zt = torch.stack([zte[obj[te_mask] == o][:, :a.k].mean(0) for o in te])
            q = torch.cat([torch.ones(len(te), 1, dtype=zt.dtype, device=dev), zt], 1)
            S[s, L] = (tr, te, q @ torch.linalg.solve(G, Xsum.T))
    print(f"[{a.model}] prediction maps built {time.time() - t0:.0f}s", flush=True)

    def stats(Y):
        """Y (B, no, 3) -> median-over-splits test R^2 per layer, shape (B, nl)"""
        out = torch.zeros(Y.shape[0], nl, dtype=torch.float64, device=dev)
        for L in range(nl):
            r = []
            for s in range(a.n_splits):
                tr, te, W = S[s, L]
                pred = W @ Y[:, tr]                                   # (B, ntest, 3)
                t = Y[:, te]
                sse = ((t - pred) ** 2).sum(1)
                sst = ((t - t.mean(1, keepdim=True)) ** 2).sum(1)
                r.append((1 - sse / sst).mean(1))
            out[:, L] = torch.stack(r, 1).median(1).values
        return out

    obs = stats(sky[None])[0].cpu().numpy()
    r2nest = sk["r2_insplit"][: a.n_splits, :, j]
    sel = sk["inner_r2"][: a.n_splits, :, j].argmax(1)
    nested = float(np.median(r2nest[np.arange(a.n_splits), sel]))
    rng = np.random.default_rng(a.seed)
    hits_fixed = hits_sel = done = 0
    mx, fx = [], []
    while done < a.n_perm:
        b = min(a.batch, a.n_perm - done)
        perms = np.stack([rng.permutation(no) for _ in range(b)])
        st = stats(sky[torch.as_tensor(perms, device=dev)])
        f = st[:, a.best_layer].cpu().numpy()
        m = st.max(1).values.cpu().numpy()
        hits_fixed += int((f >= obs[a.best_layer]).sum())
        hits_sel += int((m >= nested).sum())
        fx.append(f); mx.append(m)
        done += b
        if done % 20000 == 0:
            print(f"  {done}/{a.n_perm}  {time.time() - t0:.0f}s", flush=True)
    fx, mx = np.concatenate(fx), np.concatenate(mx)
    row = dict(model=a.model, layer=a.best_layer, n_splits=a.n_splits, k=a.k, n_perm=a.n_perm,
               obs_r2_fixed_layer=round(float(obs[a.best_layer]), 4), null_p50=round(float(np.median(fx)), 4),
               null_p99=round(float(np.quantile(fx, 0.99)), 4), null_max=round(float(fx.max()), 4),
               p_fixed_layer=(hits_fixed + 1) / (a.n_perm + 1),
               obs_r2_nested=round(nested, 4), null_max_over_layers_p50=round(float(np.median(mx)), 4),
               null_max_over_layers_max=round(float(mx.max()), 4), p_with_selection=(hits_sel + 1) / (a.n_perm + 1))
    with open(os.path.join(a.out, a.model + "_perm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
