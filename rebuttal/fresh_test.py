#!/usr/bin/env python
"""80/20 test on prompts that played no role in selecting the paper's prompt list (Issue 4).

rebuttal/prompts_fresh.csv has 25 new templates, written one for one against the paper's list (same line
order, same frame and final-word type) and committed before any activation was extracted from them. None
was dropped or edited after seeing a result.

The object splits are the ones of split_pca.py (same seed). Layers are decided before the new prompts are
looked at: the Table 5 layer ("fixed"), and the layer chosen inside the training objects on the paper's
prompts, already stored by split_pca.py ("nested"). All fits use training objects only.

  paper_paper  PCA and OLS from training objects on the paper's prompts, tested on held-out objects with
               the paper's prompts (reproduces split_pca.py)
  paper_fresh  the same fit, tested on held-out objects with the NEW prompts: neither the objects nor the
               prompts played any role in the fit
  fresh_fresh  PCA and OLS refitted from training objects on the new prompts, tested on held-out objects
               with the new prompts: the whole test re-done on the new list

Output: <out>/<model>_fresh.npz (per split) and <out>/<model>_fresh.csv (medians, 5-95% of splits, and the
paired difference to paper_paper).

Usage: fresh_test.py --model <m> --raw-dir <d> --fresh-dir <d> --split-dir <d> --layer <L> --out <d>
"""
import argparse
import csv
import os

import numpy as np
import torch

from split_pca import load_sky, obj_means, ols, pca_fit, score

TESTS = ("paper_paper", "paper_fresh", "fresh_fresh")
MODES = ("fixed", "nested")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--fresh-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="8,16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    ks = [int(x) for x in a.ks.split(",")]
    kmax = max(ks)
    sp = np.load(os.path.join(a.split_dir, a.model + ".npz"))
    n_splits = int(sp["done"])
    tests = sp["test_objects"][:n_splits]
    j8 = list(sp["ks"]).index(8)
    chosen = sp["inner_r2"][:n_splits, :, j8].argmax(1)

    rawp = np.load(os.path.join(a.raw_dir, a.model + "_raw.npy"), mmap_mode="r")
    rawf = np.load(os.path.join(a.fresh_dir, a.model + "_raw.npy"), mmap_mode="r")
    assert rawp.shape == rawf.shape, (rawp.shape, rawf.shape)
    sky = load_sky(a.objects)
    no = len(sky)
    nprompt = rawp.shape[0] // no
    obj_np = np.tile(np.arange(no), nprompt)
    obj = torch.as_tensor(obj_np, device=dev)
    y = torch.as_tensor(sky[obj_np], dtype=torch.float64, device=dev)

    cache = {}

    def layer(L):
        if L not in cache:
            cache[L] = (torch.from_numpy(np.array(rawp[:, L])).to(dev), torch.from_numpy(np.array(rawf[:, L])).to(dev))
        return [c.to(torch.float64) for c in cache[L]]

    r2 = np.full((n_splits, len(MODES), len(TESTS), len(ks)), np.nan)
    ang = r2.copy()
    allo = torch.arange(no, device=dev)
    for s in range(n_splits):
        te_ids = torch.as_tensor(tests[s], device=dev)
        te = torch.isin(obj, te_ids)
        yte = torch.stack([y[obj == i][0] for i in te_ids])
        for mi, L in enumerate((a.layer, int(chosen[s]))):
            xp, xf = layer(L)
            zp_tr, (zp_te, zf_te) = pca_fit(xp[~te], [xp[te], xf[te]], kmax)
            zr_tr, (zr_te,) = pca_fit(xf[~te], [xf[te]], kmax)
            for ti, (tr, ev) in enumerate(((zp_tr, zp_te), (zp_tr, zf_te), (zr_tr, zr_te))):
                zo = obj_means(ev, obj[te], te_ids)
                for ki, k in enumerate(ks):
                    b, b0 = ols(tr[:, :k], y[~te])
                    r2[s, mi, ti, ki], ang[s, mi, ti, ki] = score(zo[:, :k] @ b + b0, yte)
        if (s + 1) % 10 == 0:
            print(f"[{a.model}] {s + 1}/{n_splits}", flush=True)
    np.savez(os.path.join(a.out, a.model + "_fresh.npz"), r2=r2, ang=ang, tests=np.array(TESTS), modes=np.array(MODES),
             ks=np.array(ks), layer=a.layer, chosen=chosen, test_objects=tests)

    rows = []

    def rng(v, nd):
        return round(float(np.median(v)), nd), round(float(np.percentile(v, 5)), nd), round(float(np.percentile(v, 95)), nd)
    for mi, mode in enumerate(MODES):
        for ki, k in enumerate(ks):
            for ti, t in enumerate(TESTS):
                row = dict(model=a.model, mode=mode, k=k, test=t, layer=a.layer if mode == "fixed" else "nested", n_splits=n_splits)
                row["r2"], row["r2_p05"], row["r2_p95"] = rng(r2[:, mi, ti, ki], 3)
                row["ang"], row["ang_p05"], row["ang_p95"] = rng(ang[:, mi, ti, ki], 2)
                row["r2_minus_paper"], row["r2_minus_paper_p05"], row["r2_minus_paper_p95"] = rng(r2[:, mi, ti, ki] - r2[:, mi, 0, ki], 3)
                row["ang_minus_paper"], row["ang_minus_paper_p05"], row["ang_minus_paper_p95"] = rng(ang[:, mi, ti, ki] - ang[:, mi, 0, ki], 2)
                rows.append(row)
                print(row, flush=True)
    with open(os.path.join(a.out, a.model + "_fresh.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("DONE", a.model, flush=True)


if __name__ == "__main__":
    main()
