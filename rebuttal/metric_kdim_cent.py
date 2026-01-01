#!/usr/bin/env python
"""How much does the top-8 PCA limit matter for the linmap test? As metric_fullpca_cent.py (full hidden space,
per split: standardization, type centring and PCA fitted on the training objects' rows; the same 100 splits),
but the template-averaged centroids are kept in more dimensions:

  k8, k32, k128   top-k PCA of the training rows (randomized SVD, q = 192)
  full            no PCA at all: centroids in the whole standardized, type-centred hidden space, stored
                  exactly as their 188 x 188 SVD coordinates (U S; distances and the linmap ridge fit depend
                  on the centroids only through C C^T, so this is lossless)

The scale is irrelevant to the linmap R^2 (targets are squared distances, calibrated a + s d).
--full-only: only kfull. Output <out>/<model>_k<k>.npy (splits, layers, 188, dims) float32, <out>/<model>_ktests.npy.
"""
import argparse
import os
import time

import numpy as np
import torch

from metric_fullpca_cent import splits

KS = (8, 32, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--full-only", action="store_true", help="only the unreduced centroids (no PCA)")
    a = ap.parse_args()
    ks = () if a.full_only else KS
    dev = "cuda"
    m = np.load(a.meta, allow_pickle=True)
    obj, types = m["obj_ids"], m["types"].astype(str)
    nk = len(types)
    tcode = {t: i for i, t in enumerate(np.unique(types))}
    R = np.load(a.raw)
    nl = R.shape[1]
    S = splits(types, a.splits)
    obj_t = torch.tensor(obj, device=dev)
    rtype = torch.tensor(np.array([tcode[t] for t in types])[obj], device=dev)
    cnt = torch.bincount(obj_t, minlength=nk).float()[:, None]
    out = {k: np.zeros((a.splits, nl, nk, k), np.float32) for k in ks}
    out["full"] = np.zeros((a.splits, nl, nk, nk), np.float32)
    t0 = time.time()
    for L in range(nl):
        Z = torch.tensor(np.ascontiguousarray(R[:, L, :]), device=dev, dtype=torch.float32)
        for s, te in enumerate(S):
            tr = np.ones(nk, bool); tr[te] = False
            rm = torch.tensor(tr[obj], device=dev)
            Zt = Z[rm]
            mu, sd = Zt.mean(0), Zt.std(0, unbiased=False) + 1e-6
            X = (Z - mu) / sd
            for t in torch.unique(rtype):
                k = rtype == t
                X[k] -= X[k & rm].mean(0)
            if ks:
                Xt = X[rm]
                U, Sv, V = torch.svd_lowrank(Xt, q=192, niter=6, M=None)
            if ks and s == 0 and L % 10 == 0:
                ev = torch.linalg.eigvalsh((Xt @ Xt.T).double()).flip(0)[:128]
                err = ((Sv[:128].double() ** 2 - ev).abs() / ev)
                print(f"   L{L} check: rel eig err k8 {err[:8].max().item():.1e}, k32 {err[:32].max().item():.1e}, k128 {err.max().item():.1e}", flush=True)
            C = torch.zeros(nk, X.shape[1], device=dev).index_add_(0, obj_t, X) / cnt   # (188, d)
            for k in ks:
                out[k][s, L] = (C @ V[:, :k]).cpu().numpy()
            Cd = C.double()
            Uc, Sc, _ = torch.linalg.svd(Cd, full_matrices=False)
            out["full"][s, L] = (Uc * Sc).float().cpu().numpy()
        print(f"[{a.model}] layer {L + 1}/{nl}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for k, v in out.items():
        np.save(os.path.join(a.out, f"{a.model}_k{k}.npy"), v)
    np.save(os.path.join(a.out, f"{a.model}_ktests.npy"), np.array(S))
    print("DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
