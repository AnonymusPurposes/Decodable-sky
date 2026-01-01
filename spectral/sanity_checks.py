#!/usr/bin/env python
"""Sanity checks before the spectral-embedding fits (no fitting of decay parameters here).

Neural side: qwen32b, layer 63 (best LOO sky_r2 at k=16 in fixed_prompts/correlations),
top-128 PCA scores from fixed_prompts/PCA128/qwen32b_pca128.npz, averaged over the 25
location templates -> one 128-d point per object (188).

RDM convention (the planned loss): D_ij = ||e_i - e_j||^2, upper triangle as a vector,
centered and scaled to unit norm; loss(a, b) = ||r_a - r_b||^2 = 2 - 2 corr(D_a, D_b).

Checks
  1. how much of the neural RDM is object-TYPE block structure (star/other/constellation)
  2. RSA with the sphere RDM, all pairs vs within-type blocks, and vs the PCA truncation k
  3. split-half (template) reliability -> noise ceiling
  4. spherical-harmonic embedding: explicit Y_lm coordinates reproduce the zonal formula
        D(g) = sum_l f(l(l+1))^2 (2l+1)/(2 pi) (1 - P_l(cos g)),
     so every f gives an isotropic RDM; strong decay -> sphere RDM
  5. isotropic ceiling: best arbitrary function of angular separation (binned) -> upper
     bound on the SH family for any f
"""
import os, numpy as np, pandas as pd
import paths
from scipy.special import sph_harm_y, eval_legendre

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAYER = 63
rng = np.random.default_rng(0)

def upper(D):
    return D[np.triu_indices(len(D), 1)]

def rdm(E):
    g = (E * E).sum(1)
    return np.maximum(g[:, None] + g[None, :] - 2 * E @ E.T, 0)

def unit(v):
    v = v - v.mean(); return v / np.linalg.norm(v)

def corr(a, b):
    return float(unit(a) @ unit(b))

d = np.load(paths.pca128("qwen32b"), allow_pickle=True)
names, types, Y = d["names"], d["types"], d["Yunit"].astype(float)
S = d["pca"][:, LAYER, :].astype(np.float64)                 # (4700, 128)
obj, pr = d["obj_ids"], d["pr_ids"]
cat = pd.read_csv(paths.catalogue())
assert list(cat.name) == list(names)
Ycsv = cat[["x_sinA", "y_cosAsinL", "z_cosAcosL"]].to_numpy()
print("catalogue vs npz Yunit max |diff| = %.2e" % np.abs(Ycsv - Y).max())
n_obj = len(names); tmpl = np.unique(pr)
print("prompts %d, objects %d, templates %d, per-object counts %s"
      % (len(S), n_obj, len(tmpl), np.unique(np.bincount(obj))))

def centroids(mask=None, k=128):
    m = np.ones(len(S), bool) if mask is None else mask
    E = np.zeros((n_obj, k))
    for i in range(n_obj):
        E[i] = S[m & (obj == i), :k].mean(0)
    return E

E = centroids()
Dn = upper(rdm(E)); Ds = upper(rdm(Y))                        # sphere RDM = 2 - 2 cos g
iu = np.triu_indices(n_obj, 1)
ti, tj = types[iu[0]], types[iu[1]]
pair_type = np.array(["-".join(sorted(p)) for p in zip(ti, tj)])

# 1. type-block structure
r = unit(Dn); blocks = sorted(set(pair_type))
fit = np.zeros_like(r)
for b in blocks:
    fit[pair_type == b] = r[pair_type == b].mean()
print("\n[1] fraction of normalized neural RDM variance = type-block means: %.3f" % (fit @ fit))
for b in blocks:
    m = pair_type == b
    print("    %-27s n=%5d  mean sq-dist %8.1f" % (b, m.sum(), Dn[m].mean()))

# 2. RSA with sphere
def rsa_report(Dn_, tag):
    res = unit(Dn_) - fit_blocks(Dn_)
    out = {"all": corr(Dn_, Ds), "all|type-blocks removed": corr(res, Ds - blockmean(Ds))}
    for b in ["constellation-constellation", "star-star", "constellation-star"]:
        m = pair_type == b; out[b] = corr(Dn_[m], Ds[m])
    print("    %-10s " % tag + "  ".join("%s %.3f" % kv for kv in out.items()))
    return out

def blockmean(v):
    f = np.zeros_like(v)
    for b in blocks:
        f[pair_type == b] = v[pair_type == b].mean()
    return f

def fit_blocks(v):
    return blockmean(unit(v))

print("\n[2] Pearson RSA neural vs sphere (loss = 2 - 2r)")
for k in (8, 16, 32, 128):
    rsa_report(upper(rdm(centroids(k=k))), "top-%d" % k)

# 3. split-half template reliability (Spearman-Brown corrected)
print("\n[3] split-half reliability over templates (50 random 12/13 splits), top-128")
rel = {"all": [], "constellation-constellation": [], "star-star": []}
for _ in range(50):
    h = rng.permutation(tmpl)[:len(tmpl) // 2]
    ma = np.isin(pr, h)
    Da, Db = upper(rdm(centroids(ma))), upper(rdm(centroids(~ma)))
    rel["all"].append(corr(Da, Db))
    for b in ("constellation-constellation", "star-star"):
        m = pair_type == b; rel[b].append(corr(Da[m], Db[m]))
for kk, v in rel.items():
    rh = np.mean(v); print("    %-27s split-half r %.3f  -> Spearman-Brown %.3f" % (kk, rh, 2 * rh / (1 + rh)))

# 4. spherical harmonics: explicit embedding vs zonal formula
print("\n[4] SH embedding check (real orthonormal Y_lm, f(lambda)=exp(-beta*lambda), l<=12)")
dec = np.arcsin(Y[:, 0]); ra = np.arctan2(Y[:, 1], Y[:, 2])
theta, phi = np.pi / 2 - dec, ra                                # colatitude, azimuth
def sh_embed(fvals, lmax):
    cols = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Yc = sph_harm_y(l, abs(m), theta, phi)
            if m < 0:   c = np.sqrt(2) * (-1) ** m * Yc.imag
            elif m == 0: c = Yc.real
            else:       c = np.sqrt(2) * (-1) ** m * Yc.real
            cols.append(fvals[l] * c)
    return np.stack(cols, 1)
cosg = np.clip(Y @ Y.T, -1, 1)
def zonal(fvals, lmax):
    D = np.zeros_like(cosg)
    for l in range(1, lmax + 1):
        D += fvals[l] ** 2 * (2 * l + 1) / (2 * np.pi) * (1 - eval_legendre(l, cosg))
    return D
for beta in (0.3, 0.03, 0.003):
    fv = np.exp(-beta * np.arange(13) * np.arange(1, 14))
    De, Dz = rdm(sh_embed(fv, 12)), zonal(fv, 12)
    print("    beta %.3f: max|explicit - zonal| / max D = %.1e ; corr with sphere %.4f ; with neural %.3f"
          % (beta, np.abs(De - Dz).max() / Dz.max(), corr(upper(Dz), Ds), corr(upper(Dz), Dn)))

# 5. isotropic ceiling: arbitrary function of angle (quantile bins), in-sample
g = np.degrees(np.arccos(upper(cosg)))
print("\n[5] isotropic ceiling (neural RDM ~ binned function of angular separation)")
for nb in (10, 30, 60):
    q = np.quantile(g, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
    b = np.digitize(g, q[1:-1])
    pred = np.array([Dn[b == i].mean() for i in range(nb)])[b]
    predc = np.zeros_like(Dn)
    for blk in blocks:                                          # same, within each type block
        m = pair_type == blk
        for i in range(nb):
            mm = m & (b == i)
            if mm.any(): predc[mm] = Dn[mm].mean()
    print("    %2d bins: r(any f(angle)) %.3f ;  r(f(angle) per type block) %.3f ;  sphere %.3f"
          % (nb, corr(pred, Dn), corr(predc, Dn), corr(Ds, Dn)))
