#!/usr/bin/env python
"""Object centroids in the full standardized residual stream, in a form spectral/fit_models.py reads (Issue 0).

The neural RDM of appendix F.2 (fit_models.neural_rdm) is built from 25-template object centroids of the
top-128 PCA scores. This writes the same centroids without the truncation: per layer, every hidden dimension
is standardized exactly as before PCA (feat_mean / feat_std of the PCA128 file), the 25 prompt rows of each
object are averaged, and the 188 centroids are stored in an isometric 188-dimensional chart (Gram matrix of the
mean-centred centroids -> U sqrt(Lambda)). Every distance between centroids, before or after the type merge,
equals the distance in the full d-dimensional standardized space, so the downstream code runs unchanged with
obj_ids = 0..187 (one row per object).

Checks written alongside:
  check_rel   per layer, max |scores re-derived from the raw stream - stored top-128 scores| / max |stored|:
              confirms the row order and the standardization (float16 storage -> ~1e-3)
  frac128_all / frac128_const   share of the between-centroid sum of squares (after the type merge / among
              the 88 constellations) that lies in the top-128 PCA subspace

Usage: full_centroids.py --raw <dir with <model>_raw.npy> --meta <dir with <model>_pca128.npz> --out <dir> models...
"""
import argparse
import os
import time

import numpy as np


def merge_T(types):
    T = np.eye(len(types))
    for t in np.unique(types):
        m = types == t
        T[np.ix_(m, m)] -= 1.0 / m.sum()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for model in a.models:
        t0 = time.time()
        m = np.load(os.path.join(a.meta, model + "_pca128.npz"), allow_pickle=True)
        obj, types = m["obj_ids"], m["types"].astype(str)
        nk = len(types)
        R = np.load(os.path.join(a.raw, model + "_raw.npy"), mmap_mode="r")
        nl = R.shape[1]
        assert R.shape[0] == len(obj) and m["pca"].shape[1] == nl, (R.shape, m["pca"].shape)
        H = np.zeros((nk, len(obj)))
        H[obj, np.arange(len(obj))] = 1.0
        H /= H.sum(1, keepdims=True)
        T = merge_T(types)
        isc = types == "constellation"
        X = np.zeros((nk, nl, nk), np.float32)
        check, f_all, f_const = np.zeros(nl), np.zeros(nl), np.zeros(nl)
        fm, fs, pm, comp, sc = m["feat_mean"], m["feat_std"], m["pca_mean"], m["components"], m["pca"]
        for L in range(nl):
            Z = np.asarray(R[:, L, :], dtype=np.float32)
            S = (Z - fm[L]) / fs[L]
            ref = sc[:, L, :].astype(np.float32)
            check[L] = float(np.abs((S - pm[L]) @ comp[L].T - ref).max() / (np.abs(ref).max() + 1e-30))
            C = (H @ S.astype(np.float64))                          # 188 x d centroids
            Cc = C - C.mean(0)
            w, v = np.linalg.eigh(Cc @ Cc.T)
            o = np.argsort(-w)
            X[:, L, :] = (v[:, o] * np.sqrt(np.clip(w[o], 0, None))).astype(np.float32)
            P = comp[L].astype(np.float64)                           # 128 x d, orthonormal rows
            Ct = T @ C
            f_all[L] = float(((Ct @ P.T) ** 2).sum() / (Ct ** 2).sum())
            Cs = C[isc] - C[isc].mean(0)
            f_const[L] = float(((Cs @ P.T) ** 2).sum() / (Cs ** 2).sum())
        np.savez(os.path.join(a.out, model + "_pca128.npz"), pca=X, obj_ids=np.arange(nk), pr_ids=np.zeros(nk, int),
                 Yunit=m["Yunit"], types=m["types"], names=m["names"], check_rel=check,
                 frac128_all=f_all, frac128_const=f_const)
        print(model, "layers", nl, "d", R.shape[2], "check_rel max %.2e" % check.max(),
              "frac128_all median %.3f" % np.median(f_all), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
