#!/usr/bin/env python
"""Best-fit RSA of spectral embeddings against the residual stream -> fits/model_fits.csv

Two regimes, each with its own pair set and its own graphs:
  const  the 88 constellations (3828 pairs). Every graph is restricted to those nodes, so the
         constellation graph is the bare border graph -- stars and deep-sky objects take no
         part in it -- and the co-occurrence models are scored here.
  all    all 188 catalogue objects (17578 pairs), position models and the constellation graph
         only; nothing co-occurrence-related. Here the graph does carry the members, and a
         constellation->member edge has length d (weight 1/d), swept over d in (0, 1].

Type-centroid merge: in every embedding, neural and target alike, each type (constellation 88,
star 85, other 15) is shifted so that all three centroids coincide, i.e. the coordinates are
multiplied by T = I - sum_t 1_t 1_t' / n_t. Without it the three type blocks alone carry ~19%
of the neural RDM's variance in the `all` regime (sanity_checks.py) -- a difference in how the
model writes a constellation vs a star, which no positional model can or should explain. For a
target given as squared distances the same shift is applied through the Gram matrix,
G = T (-D/2) T, D' = diag(G) + diag(G)' - 2G, which needs no explicit coordinates. In the
`const` regime there is only one type, T is then the ordinary centering, and every distance is
left unchanged -- that regime's numbers are the same with the merge as without it, so the
ready-made RDM tables (which carry NaN for objects outside their vocabulary) are used as given.

Neural RDM: every layer whose LOO sky_r2 at k=16 is >= 80% of that model's maximum
(fixed_prompts/correlations/<model>_sky_r2_16.csv, all layers eligible). Per layer, the
squared distances between 25-template centroids of the top-128 PCA scores are centered and
scaled to unit norm over the analysed pairs, then averaged over the band -- without the
per-layer normalization the average would be dominated by the deepest layer, whose
residual-stream norms are largest.

Targets, each reported as the Pearson r of its RDM with the neural one:
  sphere       1 - cos(angular separation)
  harmonics    spherical harmonics, coordinate (l, m) = f(l(l+1)) Y_lm, l <= 2000:
               D(g) = sum_l f^2 (2l+1)(1 - P_l(cos g)), each degree merged separately
  const_graph  Laplacian eigenmap of the constellation graph (see the regimes above)
  cooc_graph   Laplacian eigenmap of the Wikipedia co-occurrence graph, and
  cooc_graph_nolist  the same with mentions in lists and tables dropped    [const only]
  <rdms/*.csv> ready-made RDM tables (e.g. the QWEM co-occurrence embedding from
               build_qwem_rdm.py); no free parameter                     [const only]
Graph coordinate n = f(lambda_n) * eigenvector n (zero mode dropped), with the combinatorial
Laplacian D - A (unit eigenvectors) or the normalized one I - D^-1/2 A D^-1/2 (eigenfunctions
D^-1/2 u); the better of the two is kept and recorded. The decay is f = exp(-k lambda) for
every target, k in [1e-3, 30] scanned on a log grid and refined with bounded Brent.

Usage: fit_models.py [model ...]      (default: every npz in fixed_prompts/PCA128)
"""
import os, sys, glob
import numpy as np, pandas as pd
import paths
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LMAX_SH = 2000
SH_CHUNK = 250
K_RANGE = (1e-3, 30.0)
NGRID = 401
D_RANGE = (1e-3, 1.0)
ND = 41
BAND_FRAC = 0.8


def band(model):
    c = pd.read_csv(paths.sky_r2(model))
    mx = c.r2.max()
    return c.layer[c.r2 >= BAND_FRAC * mx].tolist(), float(mx)


def load_pca(model, names):
    d = np.load(paths.pca128(model), allow_pickle=True)
    assert list(d["names"]) == list(names), model
    obj = d["obj_ids"]
    return d["pca"], obj, np.bincount(obj, minlength=len(names))[:, None]


def merge_T(groups):
    """T = I - sum_t 1_t 1_t' / n_t: subtract each type's mean, merging the type centroids"""
    T = np.eye(len(groups))
    for g in np.unique(groups):
        m = groups == g
        T[np.ix_(m, m)] -= 1.0 / m.sum()
    return T


def merge_D(D, T):
    """the same merge for a target given as squared distances, through its Gram matrix"""
    G = T @ (-0.5 * D) @ T
    d = np.diag(G)
    return d[:, None] + d[None, :] - 2 * G


def neural_rdm(pca, obj, cnt, layers, nodes, T, ii, jj):
    """mean over band layers of the unit-normalized centered RDM on the analysed pairs"""
    acc = np.zeros(len(ii))
    for L in layers:
        E = np.zeros((len(cnt), pca.shape[2]))
        np.add.at(E, obj, pca[:, L, :].astype(np.float64))
        E = T @ (E / cnt)[nodes]
        g = (E * E).sum(1)
        v = g[ii] + g[jj] - 2 * (E[ii] * E[jj]).sum(1)
        v = v - v.mean()
        acc += v / np.linalg.norm(v)
    return acc / len(layers)


def corr(X, y):
    X = np.atleast_2d(X).astype(np.float64)
    X = X - X.mean(1, keepdims=True); y = y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X @ y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y))


def sh_basis(C, T, ii, jj):
    """merged squared distances (2l+1)(1 - P_l(cos g)) of each degree l = 1..LMAX_SH"""
    B = np.empty((LMAX_SH, len(ii)), np.float32)
    buf = np.empty((SH_CHUNK, len(C), len(C)))
    p0, p1, rows = np.ones_like(C), C.copy(), []
    for l in range(1, LMAX_SH + 1):
        buf[len(rows)] = (2 * l + 1) * (1 - p1)
        rows.append(l - 1)
        if len(rows) == SH_CHUNK or l == LMAX_SH:
            G = np.matmul(T, np.matmul(-0.5 * buf[:len(rows)], T))
            dg = np.einsum("kii->ki", G)
            B[rows] = (dg[:, ii] + dg[:, jj] - 2 * G[:, ii, jj]).astype(np.float32)
            rows = []
        p0, p1 = p1, ((2 * l + 1) * C * p1 - l * p0) / (l + 1)
    lam = np.arange(1, LMAX_SH + 1) * np.arange(2, LMAX_SH + 2).astype(float)
    return lam, B


def graph_basis(A, kind, T, ii, jj):
    deg = A.sum(1)
    if kind == "comb":
        lam, V = np.linalg.eigh(np.diag(deg) - A)
    else:
        dm = 1 / np.sqrt(deg)
        lam, U = np.linalg.eigh(np.eye(len(A)) - dm[:, None] * A * dm[None, :])
        V = U * dm[:, None]
    keep = lam > 1e-9 * max(1.0, lam.max())
    assert (~keep).sum() == 1, "graph not connected"
    lam, V = lam[keep], T @ V[:, keep]
    return lam, ((V[ii] - V[jj]) ** 2).T.copy()


def const_graph_A(base, const_of, nodes, d):
    """border edges weight 1, constellation->member edges weight 1/d (edge length d);
    restricting to the constellation nodes drops the member edges altogether"""
    A = base.astype(float).copy()
    m = const_of >= 0
    A[np.where(m)[0], const_of[m]] = A[const_of[m], np.where(m)[0]] = 1.0 / d
    return A[np.ix_(nodes, nodes)]


def target_rdm(basis, k):
    lam, B = basis
    k = np.atleast_1d(k).astype(float)
    lw = -2 * k[:, None] * lam[None, :]                       # log f^2, f = exp(-k lambda)
    W = np.exp(lw - lw.max(1, keepdims=True)).astype(B.dtype)
    return W @ B


def best_decay(basis, y):
    """-> (r, k) maximising the correlation over the exponential decay rate"""
    grid = np.logspace(np.log10(K_RANGE[0]), np.log10(K_RANGE[1]), NGRID)
    r = np.concatenate([corr(target_rdm(basis, g), y) for g in np.array_split(grid, 4)])
    i = int(np.nanargmax(r))
    res = minimize_scalar(lambda lk: -corr(target_rdm(basis, np.exp(lk)), y)[0],
                          bounds=(np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, NGRID - 1)])),
                          method="bounded", options=dict(xatol=1e-4))
    return (-res.fun, float(np.exp(res.x))) if -res.fun >= r[i] else (r[i], float(grid[i]))


def best_graph(A_of_d, y, T, ii, jj, sweep_d):
    """-> dict(r, laplacian, param, d); d is swept only where member edges exist"""
    def at(d):
        return max((best_decay(graph_basis(A_of_d(d), kind, T, ii, jj), y) + (kind,)
                    for kind in ("comb", "norm")), key=lambda t: t[0])
    if not sweep_d:
        r, k, kind = at(1.0)
        return dict(r=float(r), laplacian=kind, param=k, d=np.nan)
    ds = np.logspace(np.log10(D_RANGE[0]), np.log10(D_RANGE[1]), ND)
    vals = [at(d) for d in ds]
    i = int(np.argmax([v[0] for v in vals]))
    res = minimize_scalar(lambda ld: -at(float(np.exp(ld)))[0],
                          bounds=(np.log(ds[max(i - 1, 0)]), np.log(ds[min(i + 1, ND - 1)])),
                          method="bounded", options=dict(xatol=1e-3))
    d_opt = float(np.exp(res.x)) if -res.fun >= vals[i][0] else float(ds[i])
    r, k, kind = at(d_opt)
    return dict(r=float(r), laplacian=kind, param=k, d=d_opt)


def setup(regime, cat, gc, go):
    """pair set, type-centroid merge and targets of one regime"""
    names, types = cat.name.tolist(), cat.type.to_numpy()
    nodes = (np.where(types == "constellation")[0] if regime == "const"
             else np.arange(len(names)))
    ii, jj = np.triu_indices(len(nodes), 1)
    T = merge_T(types[nodes])
    Y = cat[["x_sinA", "y_cosAsinL", "z_cosAcosL"]].to_numpy()[nodes]
    C = np.clip(Y @ Y.T, -1, 1)
    graphs = [("const_graph", lambda d: const_graph_A(gc["A"], gc["const_of"], nodes, d),
               regime == "all")]
    tables = {}
    if regime == "const":
        graphs.append(("cooc_graph", lambda d: go["A"].astype(float)[np.ix_(nodes, nodes)], False))
        graphs.append(("cooc_graph_nolist",
                       lambda d: go["A_nolist"].astype(float)[np.ix_(nodes, nodes)], False))
        for f in sorted(glob.glob(os.path.join(HERE, "rdms", "*.csv"))):
            t = pd.read_csv(f, index_col=0)
            assert list(t.index) == names, f
            tables[os.path.basename(f)[:-4]] = t.to_numpy(float)[nodes[ii], nodes[jj]]
    return dict(regime=regime, nodes=nodes, T=T, ii=ii, jj=jj,
                sphere=merge_D(1 - C, T)[ii, jj], sh=sh_basis(C, T, ii, jj),
                graphs=graphs, tables=tables)


def main():
    models = sys.argv[1:] or paths.models()
    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist()
    gc = np.load(os.path.join(HERE, "graphs", "graph_constellation.npz"))
    go = np.load(os.path.join(HERE, "graphs", "graph_cooccurrence.npz"))
    assert list(gc["names"]) == names and list(go["names"]) == names
    regimes = [setup(r, cat, gc, go) for r in ("const", "all")]

    rows = []
    for model in models:
        layers, r2max = band(model)
        pca, obj, cnt = load_pca(model, names)
        for R in regimes:
            y = neural_rdm(pca, obj, cnt, layers, R["nodes"], R["T"], R["ii"], R["jj"])
            row = dict(model=model, regime=R["regime"], n_pairs=len(R["ii"]),
                       n_layers_band=len(layers), band_first=min(layers), band_last=max(layers),
                       r2_max=r2max, sphere_r=float(corr(R["sphere"], y)[0]))
            r, k = best_decay(R["sh"], y)
            row.update(harmonics_r=r, harmonics_k=k)
            for gname, A_of_d, sweep in R["graphs"]:
                b = best_graph(A_of_d, y, R["T"], R["ii"], R["jj"], sweep)
                row.update({gname + "_r": b["r"], gname + "_laplacian": b["laplacian"],
                            gname + "_k": b["param"]})
                if sweep:
                    row[gname + "_d"] = b["d"]
            for tname, v in R["tables"].items():
                ok = ~np.isnan(v)
                row[tname + "_r"] = float(corr(v[ok], y[ok])[0])
                row[tname + "_n_pairs"] = int(ok.sum())
            rows.append(row)
            print("%-20s %-5s %5d pairs | band L%d-%d (%d) | sphere %.3f | harmonics %.3f (k %.3g) | %s%s"
                  % (model, R["regime"], row["n_pairs"], row["band_first"], row["band_last"],
                     row["n_layers_band"], row["sphere_r"], row["harmonics_r"], row["harmonics_k"],
                     " | ".join("%s %.3f (%s k %.3g%s)"
                                % (g, row[g + "_r"], row[g + "_laplacian"], row[g + "_k"],
                                   ", d %.3g" % row[g + "_d"] if g + "_d" in row else "")
                                for g, _, _ in R["graphs"]),
                     "".join(" | %s %.3f" % (t, row[t + "_r"]) for t in R["tables"])), flush=True)

    os.makedirs(os.path.join(HERE, "fits"), exist_ok=True)
    out = os.path.join(HERE, "fits", "model_fits.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print("saved", os.path.relpath(out, HERE))   # abs path may be un-encodable in the console codepage


if __name__ == "__main__":
    main()
