#!/usr/bin/env python
"""Bootstrap intervals for the correlations of appendix F.2 (Issue 0).

Same targets, neural RDM and decay family as spectral/fit_models.py, imported from it. Objects are
resampled with replacement, within each type, and a pair of copies of the same object is dropped. The
decay rate of the harmonic and graph targets is refitted in every resample on a 41-point grid, so the
interval includes the choice of that free parameter. The type-centroid merge is fixed at its full-data
value. Paired differences to the sphere use the same resamples. The member-edge length d of the constellation
graph in the `all` regime is held at each model's own best value from fits/model_fits.csv (--graph-d own,
default), or at the median over the models (--graph-d median, the earlier setting).

Output, one row per model, regime and target: r on the full data (same grid), the 2.5 and 97.5
percentiles, and target minus sphere with its percentiles.

Usage: spectral_intervals.py --spectral <dir with fit_models.py, graphs/, rdms/> --out <csv> [--nboot 500]
"""
import argparse
import csv
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--graph-d", choices=["own", "median"], default="own")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    sys.path.insert(0, a.spectral)
    import pandas as pd
    import fit_models as F
    import paths

    GRID = np.logspace(np.log10(F.K_RANGE[0]), np.log10(F.K_RANGE[1]), 41)

    def fit(bases, y, sel):
        """best r over the grid, over the given bases, on the pairs sel"""
        best = -2.0
        for lam, B in bases:
            r = F.corr(F.target_rdm((lam, B[:, sel]), GRID), y[sel])
            if np.any(np.isfinite(r)):
                best = max(best, float(np.nanmax(r)))
        return best

    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist()
    types = cat.type.to_numpy()
    gc = np.load(os.path.join(a.spectral, "graphs", "graph_constellation.npz"))
    go = np.load(os.path.join(a.spectral, "graphs", "graph_cooccurrence.npz"))
    regimes = [F.setup(r, cat, gc, go) for r in ("const", "all")]
    models = a.models or paths.models()
    rng = np.random.default_rng(a.seed)
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for R in regimes:
        n = len(R["nodes"])
        pid = np.full((n, n), -1)
        pid[R["ii"], R["jj"]] = pid[R["jj"], R["ii"]] = np.arange(len(R["ii"]))
        ty = types[R["nodes"]]
        boots = []
        for _ in range(a.nboot):
            b = np.concatenate([rng.choice(np.where(ty == t)[0], (ty == t).sum()) for t in np.unique(ty)])
            s = pid[b[R["ii"]], b[R["jj"]]]
            boots.append(s[s >= 0])
        R["boots"] = boots
        R["gcache"] = {}

    fits = pd.read_csv(os.path.join(a.spectral, "fits", "model_fits.csv"))

    def graph_bases(R, model):
        """graph bases of this regime; a swept member-edge length d is taken as found on the full data"""
        out = {}
        for gname, A_of_d, sweep in R["graphs"]:
            d = 1.0
            if sweep:
                t = fits[fits.regime == R["regime"]]
                d = float(t[gname + "_d"].median() if a.graph_d == "median" else t[t.model == model][gname + "_d"].iloc[0])
            key = (gname, round(d, 9))
            if key not in R["gcache"]:
                R["gcache"][key] = [F.graph_basis(A_of_d(d), kind, R["T"], R["ii"], R["jj"]) for kind in ("comb", "norm")]
            out[gname] = R["gcache"][key]
        return out

    for model in models:
        layers, _ = F.band(model)
        pca, obj, cnt = F.load_pca(model, names)
        for R in regimes:
            y = F.neural_rdm(pca, obj, cnt, layers, R["nodes"], R["T"], R["ii"], R["jj"])
            tg = {"sphere": lambda s: float(F.corr(R["sphere"][s], y[s])[0]),
                  "harmonics": lambda s: fit([R["sh"]], y, s)}
            for g, bs in graph_bases(R, model).items():
                tg[g] = (lambda bs: (lambda s: fit(bs, y, s)))(bs)
            for t, v in R["tables"].items():
                def tab(v):
                    def f(s):
                        s2 = s[~np.isnan(v[s])]
                        return float(F.corr(v[s2], y[s2])[0])
                    return f
                tg[t] = tab(v)
            full = np.arange(len(y))
            res = {k: np.array([f(s) for s in R["boots"]]) for k, f in tg.items()}
            pts = {k: f(full) for k, f in tg.items()}
            for k in tg:
                d = res[k] - res["sphere"]
                rows.append(dict(model=model, regime=R["regime"], target=k, r=round(pts[k], 4),
                                 r_lo=round(float(np.percentile(res[k], 2.5)), 4),
                                 r_hi=round(float(np.percentile(res[k], 97.5)), 4),
                                 minus_sphere=round(pts[k] - pts["sphere"], 4),
                                 minus_sphere_lo=round(float(np.percentile(d, 2.5)), 4),
                                 minus_sphere_hi=round(float(np.percentile(d, 97.5)), 4), nboot=a.nboot))
            print(model, R["regime"], {k: round(v, 3) for k, v in pts.items()}, flush=True)
        with open(a.out, "w", newline="") as fh:              # after each model: partial results are usable
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
