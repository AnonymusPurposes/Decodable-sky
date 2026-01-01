#!/usr/bin/env python
"""Repeated 80/20 object splits with everything fitted on the training objects only.

Answers two reviewer points at once:

  * PCA leakage. correlations_split.py reuses a PCA fitted on all objects. Here, for every
    split and layer, per-feature standardization and PCA are fitted on the training objects'
    per-prompt residual vectors alone; test objects are projected through that fit.
  * Layer selection. The old test took the layer with the best leave-one-out R^2 over all
    objects, so the layer choice had seen the test objects. Here the layer is chosen inside
    the training objects by k-fold cross-validation over objects, per split, and the test
    objects are scored once at that layer ("nested"). Every layer is also recorded.

For comparison, the same splits are also scored with the old global PCA ("global").

Protocol otherwise matches correlations_split.py: OLS with intercept from the top-k scores to
the unit sky vector, trained on individual prompts; test objects prompt-averaged; R^2 is
sklearn's uniform average over the three coordinates; angular error between the predicted
direction and the true one, median over test objects.

Output <out>/<model>.npz, written every --save-every splits and resumed on restart:
  r2_insplit, ang_insplit, r2_global, ang_global   (n_splits, n_layers, n_k)
  inner_r2                                         (n_splits, n_layers, n_k)
  done   number of finished splits    test_objects (n_splits, n_test)    ks

Usage: split_pca.py --model <m> --raw-dir <d> --objects data/astro_objects.csv --out <d>
"""
import argparse
import csv
import os
import time

import numpy as np
import torch


def load_sky(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])] for r in rows])


def pca_fit(train, others, kmax):
    """standardize on `train`, PCA on `train`; returns scores of train and of each array in others"""
    mu = train.mean(0)
    sd = train.std(0, correction=0) + 1e-6
    a = (train - mu) / sd
    if a.shape[1] <= a.shape[0]:
        _, v = torch.linalg.eigh(a.T @ a)
        comps = v[:, -kmax:].flip(1)
    else:
        w, u = torch.linalg.eigh(a @ a.T)
        u, w = u[:, -kmax:].flip(1), w[-kmax:].flip(0)
        comps = (a.T @ u) / w.clamp_min(1e-12).sqrt()
    return a @ comps, [((x - mu) / sd) @ comps for x in others]


def ols(z, y):
    """OLS with intercept: returns (beta, intercept)"""
    zm, ym = z.mean(0), y.mean(0)
    zc = z - zm
    beta = torch.linalg.solve(zc.T @ zc, zc.T @ (y - ym))
    return beta, ym - zm @ beta


def score(pred, truth):
    sse = ((truth - pred) ** 2).sum(0)
    sst = ((truth - truth.mean(0)) ** 2).sum(0)
    r2 = (1 - sse / sst).mean()
    cos = (torch.nn.functional.normalize(pred, dim=1) * torch.nn.functional.normalize(truth, dim=1)).sum(1)
    return float(r2), float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).median())


def obj_means(z, obj, ids):
    """average per-prompt rows of `z` over each object in `ids`"""
    return torch.stack([z[obj == i].mean(0) for i in ids])


def test_score(ztr, ytr, zte_obj, yte_obj, k):
    beta, b0 = ols(ztr[:, :k], ytr)
    return score(zte_obj[:, :k] @ beta + b0, yte_obj)


def inner_cv(ztr, ytr, obj_tr, ks, folds):
    """pooled cross-validated R^2 over training objects, on the training-only PCA scores"""
    out = []
    for k in ks:
        preds, truth = [], []
        for f in folds:
            te = torch.isin(obj_tr, f)
            beta, b0 = ols(ztr[~te, :k], ytr[~te])
            zo = obj_means(ztr[:, :k], obj_tr, f)
            preds.append(zo @ beta + b0)
            truth.append(torch.stack([ytr[obj_tr == i][0] for i in f]))
        out.append(score(torch.cat(preds), torch.cat(truth))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="same seed = same splits for every model")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    ks = [int(x) for x in a.ks.split(",")]
    kmax = max(ks)

    raw = np.load(os.path.join(a.raw_dir, a.model + "_raw.npy"), mmap_mode="r")
    n, nl, hidden = raw.shape
    sky = load_sky(a.objects)
    no = len(sky)
    assert n % no == 0, (n, no)
    np_ = n // no
    obj_np = np.tile(np.arange(no), np_)                 # order of extract_pca.py: prompts x objects
    obj = torch.as_tensor(obj_np, device=dev)
    y = torch.as_tensor(sky[obj_np], dtype=torch.float64, device=dev)
    print(f"[{a.model}] raw {raw.shape}, {np_} prompts x {no} objects", flush=True)

    ntest = max(1, int(round(no * a.test_frac)))
    rng = np.random.default_rng(a.seed)
    test_sets = np.stack([rng.permutation(no)[:ntest] for _ in range(a.n_splits)])
    frng = np.random.default_rng(a.seed + 1)

    shape = (a.n_splits, nl, len(ks))
    res = {k: np.full(shape, np.nan) for k in ("r2_insplit", "ang_insplit", "r2_global", "ang_global", "inner_r2")}
    path = os.path.join(a.out, a.model + ".npz")
    done = 0
    if os.path.exists(path):
        d = np.load(path)
        if (d["test_objects"] == test_sets).all() and list(d["ks"]) == ks:
            done = int(d["done"])
            for k in res:
                res[k] = d[k]
            print(f"resuming after {done} splits", flush=True)

    def save(nd):
        np.savez(path + ".tmp.npz", done=np.int64(nd), test_objects=test_sets, ks=np.array(ks), **res)
        os.replace(path + ".tmp.npz", path)

    # all layers on the device as float32 half-precision source, cast per layer
    t0 = time.time()
    X = torch.from_numpy(np.array(raw)).to(dev)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    global_scores = []
    for L in range(nl):
        x = X[:, L].to(torch.float64)
        z, _ = pca_fit(x, [], kmax)
        global_scores.append(z)
    print(f"global PCA done {time.time() - t0:.0f}s", flush=True)

    allo = torch.arange(no, device=dev)
    # fold assignments are drawn for all splits so that a resumed run gets the same ones
    fold_perms = [frng.permutation(no - ntest) for _ in range(a.n_splits)]
    for s in range(done, a.n_splits):
        ts = time.time()
        te_ids = torch.as_tensor(test_sets[s], device=dev)
        te_mask = torch.isin(obj, te_ids)
        tr_ids = allo[~torch.isin(allo, te_ids)]
        fp = torch.as_tensor(fold_perms[s], device=dev)
        folds = [tr_ids[fp[i::a.inner_folds]] for i in range(a.inner_folds)]
        ytr = y[~te_mask]
        obj_tr = obj[~te_mask]
        yte = torch.stack([y[obj == i][0] for i in te_ids])
        for L in range(nl):
            x = X[:, L].to(torch.float64)
            ztr, (zte,) = pca_fit(x[~te_mask], [x[te_mask]], kmax)
            zte_obj = obj_means(zte, obj[te_mask], te_ids)
            res["inner_r2"][s, L] = inner_cv(ztr, ytr, obj_tr, ks, folds)
            zg = global_scores[L]
            zg_obj = obj_means(zg[te_mask], obj[te_mask], te_ids)
            for j, k in enumerate(ks):
                res["r2_insplit"][s, L, j], res["ang_insplit"][s, L, j] = test_score(ztr, ytr, zte_obj, yte, k)
                res["r2_global"][s, L, j], res["ang_global"][s, L, j] = test_score(zg[~te_mask], ytr, zg_obj, yte, k)
        if (s + 1) % a.save_every == 0 or s + 1 == a.n_splits:
            save(s + 1)
        best = res["inner_r2"][s, :, ks.index(8) if 8 in ks else 0].argmax()
        print(f"split {s + 1}/{a.n_splits}  {time.time() - ts:.1f}s  inner-selected L{best}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
