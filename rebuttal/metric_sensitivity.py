#!/usr/bin/env python
"""Appendix F.1 checks requested in review: (1) a full metric with cross terms, (2) sensitivity to the
coordinate frame and to where the flat chart is cut.

Same objects (non-constellation), same 100 train/test object splits and same layers as
metric_heldout.py (read from its output; bands longer than --max-layers are subsampled evenly). Neural side: the per-prompt rows of the global 128-PCA file
(fixed_prompts/PCA128), top-8 PCA refitted on the training objects' rows only, scaled so PC-1 has
variance 1, squared distances between template-averaged centroids. (metric_heldout.py starts from the
raw activations instead; the 128-D basis here saw all objects, which is checked by rerunning its OLS
fits on the same splits: column check_ols vs heldout_ols in the output.)

Candidates: coordinates of one map in one frame.
  sphere@F    unit vector                         (frame-free for the psd and linmap fitters)
  cylinder@F  sin(lat), cos(lon), sin(lon)        axis = frame's pole
  chart@F/c   lat, (lon - c) mod 360               flat map cut at longitude c, c = 0, 30, ..., 330
  F = equatorial (the paper), ecliptic, galactic.
  dec, ra/c   the equatorial chart's two coordinates separately (latitude angle; RA cut at c)
Each candidate is scored alone and joined with the equatorial sphere.

Fitters, all fitted on train objects and scored by R^2 on held-out pairs (at least one test object):
  diag     OLS on one squared-difference RDM per coordinate (the paper's F.1 regression)
  psd      squared Mahalanobis distance (dc)^T G (dc) + b0 with all cross terms, G constrained to be
           positive semidefinite (projected gradient), ridge penalty chosen by inner CV over object folds
  linmap   ridge linear map from coordinates to the 8 PCs (inner CV over object folds); predicted
           squared distances calibrated by a + s*d on train pairs

Output <out>/<model>.npz: r2 (splits, layers, candidates, joined 0/1, fitters) and names; and a summary
CSV over models (metric_sensitivity_summary.csv) written by --summarize.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh

FITTERS = ("diag", "psd", "linmap")
LAMS = np.logspace(-4, 2, 7)
CUTS = list(range(0, 360, 30))
FRAMES = ("equatorial", "ecliptic", "galactic")


def frames(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
    import astropy.units as u
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    e = c.transform_to(BarycentricTrueEcliptic())
    g = c.galactic
    return {"equatorial": (np.radians(dec_deg), np.radians(ra_deg)),
            "ecliptic": (e.lat.radian, e.lon.radian),
            "galactic": (g.b.radian, g.l.radian)}


def candidates(ra, dec):
    fr = frames(ra, dec)
    cand = {}
    for f in FRAMES:
        lat, lon = fr[f]
        cand[f"sphere@{f}"] = np.column_stack([np.sin(lat), np.cos(lat) * np.sin(lon), np.cos(lat) * np.cos(lon)])
        cand[f"cylinder@{f}"] = np.column_stack([np.sin(lat), np.cos(lon), np.sin(lon)])
        for c in CUTS:
            cand[f"chart@{f}/{c}"] = np.column_stack([lat, np.mod(lon - np.radians(c), 2 * np.pi)])
    lat, lon = fr["equatorial"]                     # the chart's two coordinates separately
    cand["dec"] = lat[:, None]
    for c in CUTS:
        cand[f"ra/{c}"] = np.mod(lon - np.radians(c), 2 * np.pi)[:, None]
    return cand


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ridge(X, Y, lam):
    """ridge with free intercept; lam scaled by n"""
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    B = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ (Y - ym))
    return ym - xm @ B, B


def psd_proj(g, p):
    G = g.reshape(p, p)
    w, V = eigh((G + G.T) / 2)
    return ((V * np.clip(w, 0, None)) @ V.T).ravel()


def psd_fit(F, y, lam, p, iters=400):
    """min mean (y - b0 - F g)^2 + lam |g|^2 over symmetric PSD G = g.reshape(p, p); FISTA"""
    fm, ym = F.mean(0), y.mean()
    Fc = F - fm
    Q, r = Fc.T @ Fc / len(F), Fc.T @ (y - ym) / len(F)
    I = np.eye(len(Q))
    L = 2 * (np.linalg.eigvalsh(Q)[-1] + lam)
    g = psd_proj(np.linalg.lstsq(Q + lam * I, r, rcond=None)[0], p)
    z, t = g.copy(), 1.0
    for _ in range(iters):
        gn = psd_proj(z - (2 * (Q @ z - r) + 2 * lam * z) / L, p)
        t1 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = gn + (t - 1) / t1 * (gn - g)
        g, t = gn, t1
    return ym - fm @ g, g


def cv_lam(fit_pred, fold):
    """pick lambda by leave-one-fold-of-objects-out; fit_pred(lam, train_obj_mask) -> held-out SSE"""
    best, bl = np.inf, LAMS[0]
    for lam in LAMS:
        sse = sum(fit_pred(lam, fold != g) for g in np.unique(fold))
        if sse < best:
            best, bl = sse, lam
    return bl


def pair_feats(C, iu, ju, cross):
    D = C[iu] - C[ju]
    if not cross:
        return D ** 2
    p = C.shape[1]
    a, b = np.triu_indices(p)
    return D[:, a] * D[:, b] * np.where(a == b, 1.0, 2.0)


def score(C, cent, tr_obj, iu, ju, pair_tr, fold, how):
    """held-out-pair R^2 for one coordinate block C (n_obj, p)"""
    y = ((cent[iu] - cent[ju]) ** 2).sum(1)
    m, s = C[tr_obj].mean(0), C[tr_obj].std(0) + 1e-12
    Cz = (C - m) / s
    pair_te = ~pair_tr
    if how == "diag":
        F = pair_feats(Cz, iu, ju, False)
        fm, fs = F[pair_tr].mean(0), F[pair_tr].std(0) + 1e-12
        F = (F - fm) / fs
        X = np.column_stack([np.ones(pair_tr.sum()), F[pair_tr]])
        b = np.linalg.lstsq(X, y[pair_tr], rcond=None)[0]
        return r2(y[pair_te], b[0] + F[pair_te] @ b[1:])
    if how == "psd":
        p = Cz.shape[1]
        D = Cz[iu] - Cz[ju]
        F = (D[:, :, None] * D[:, None, :]).reshape(len(D), p * p)   # <D D^T, G> = F . vec(G)
        fi, fj = fold[iu], fold[ju]
        tri = np.where(pair_tr)[0]

        def fp(lam, keep):                                            # lambda chosen without the cone
            tr = tri[keep[fi[tri]] & keep[fj[tri]]]
            te = tri[~keep[fi[tri]] | ~keep[fj[tri]]]
            b0, B = ridge(F[tr], y[tr], lam)
            return ((y[te] - b0 - F[te] @ B) ** 2).sum()
        b0, g = psd_fit(F[pair_tr], y[pair_tr], cv_lam(fp, fold), p)
        return r2(y[pair_te], b0 + F[pair_te] @ g)
    # linmap: coordinates -> centroids
    tro = np.where(tr_obj)[0]

    def fo(lam, keep):
        tr, te = tro[keep[tro]], tro[~keep[tro]]
        b0, B = ridge(Cz[tr], cent[tr], lam)
        return ((cent[te] - b0 - Cz[te] @ B) ** 2).sum()
    lam = cv_lam(fo, fold)
    b0, B = ridge(Cz[tr_obj], cent[tr_obj], lam)
    P = b0 + Cz @ B
    d = ((P[iu] - P[ju]) ** 2).sum(1)
    X = np.column_stack([np.ones(pair_tr.sum()), d[pair_tr]])
    a0, s0 = np.linalg.lstsq(X, y[pair_tr], rcond=None)[0]
    return r2(y[pair_te], a0 + s0 * d[pair_te])


def run(job):
    model, indir, objects, heldout_dir, out, nsplits, max_layers = job
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    path = os.path.join(out, model + ".npz")
    if os.path.exists(path):
        return model
    H = np.load(os.path.join(heldout_dir, model + ".npz"), allow_pickle=True)
    tests, layers = H["tests"][:nsplits], [int(x) for x in H["layers"]]
    lsel = np.unique(np.round(np.linspace(0, len(layers) - 1, min(len(layers), max_layers))).astype(int))
    layers = [layers[i] for i in lsel]
    ref_ols = H["heldout_r2"][:nsplits][:, lsel][:, :, :, list(H["fitters"]).index("ols")]
    ref_names = [str(x) for x in H["candidates"]]
    d = np.load(os.path.join(indir, model + "_pca128.npz"), allow_pickle=True)
    rows = list(csv.DictReader(open(objects, newline="", encoding="utf-8")))
    names = d["names"].astype(str)
    assert [r["name"] for r in rows] == list(names)
    keep = np.array([i for i, r in enumerate(rows) if r["type"] != "constellation"])
    nk = len(keep)
    remap = -np.ones(len(rows), int); remap[keep] = np.arange(nk)
    sel = np.isin(d["obj_ids"], keep)
    obj = remap[d["obj_ids"][sel]]
    X = d["pca"][sel][:, layers].astype(np.float32)
    ra = np.array([float(rows[i]["l_ra_deg"]) for i in keep])
    dec = np.array([float(rows[i]["a_dec_deg"]) for i in keep])
    cand = candidates(ra, dec)
    cnames = list(cand)
    sph = cand["sphere@equatorial"]
    # the paper's own coordinate blocks, to check against metric_heldout.py (diag == its OLS)
    rr, dd = np.radians(ra), np.radians(dec)
    own = {"sphere": sph, "cylinder": np.column_stack([np.sin(dd), np.cos(rr), np.sin(rr)]),
           "chart": np.column_stack([dd, rr])}
    iu, ju = np.triu_indices(nk, 1)
    res = np.full((len(tests), len(layers), len(cnames), 2, len(FITTERS)), np.nan)
    chk = np.full((len(tests), len(layers), 3), np.nan)
    for s, te in enumerate(tests):
        is_te = np.zeros(nk, bool); is_te[te] = True
        tr_obj = ~is_te
        pair_tr = tr_obj[iu] & tr_obj[ju]
        fold = np.random.default_rng(1000 + s).integers(0, 4, nk)
        rmask = tr_obj[obj]
        for li in range(len(layers)):
            Z = X[:, li].astype(np.float64)
            zt = Z[rmask]
            mu = zt.mean(0)
            _, V = eigh((zt - mu).T @ (zt - mu) / len(zt))
            P = (Z - mu) @ V[:, ::-1][:, :8]
            P /= P[rmask, 0].std()
            cent = np.zeros((nk, 8))
            np.add.at(cent, obj, P)
            cent /= np.bincount(obj, minlength=nk)[:, None]
            for k, (nm, C) in enumerate(own.items()):
                chk[s, li, k] = score(C, cent, tr_obj, iu, ju, pair_tr, fold, "diag")
            for ci, nm in enumerate(cnames):
                for j, C in enumerate((cand[nm], np.column_stack([sph, cand[nm]]))):
                    if j == 1 and nm == "sphere@equatorial":
                        continue
                    for fi, how in enumerate(FITTERS):
                        res[s, li, ci, j, fi] = score(C, cent, tr_obj, iu, ju, pair_tr, fold, how)
        if (s + 1) % 10 == 0:
            print(f"[{model}] split {s + 1}/{len(tests)}", flush=True)
    np.savez(path, r2=res, names=np.array(cnames), fitters=np.array(FITTERS), layers=np.array(layers),
             check_ols=chk, heldout_ols=ref_ols[:, :, [ref_names.index(n) for n in own]])
    return model


def summarize(out, models):
    rows = []
    for m in models:
        d = np.load(os.path.join(out, m + ".npz"), allow_pickle=True)
        R, names = d["r2"], [str(x) for x in d["names"]]
        sph = R[:, :, names.index("sphere@equatorial"), 0, :]            # (split, layer, fitter)
        chk = np.nanmean(np.abs(d["check_ols"] - d["heldout_ols"]))
        for ci, nm in enumerate(names):
            for fi, how in enumerate(FITTERS):
                alone = R[:, :, ci, 0, fi].mean(1)                         # mean over band layers, per split
                joined = R[:, :, ci, 1, fi].mean(1)
                s = sph[:, :, fi].mean(1)
                row = dict(model=m, candidate=nm, fitter=how, sphere=np.median(s), alone=np.median(alone))
                if not np.isnan(joined).all():
                    add_c = joined - s                                      # candidate's coordinates on top of the sphere
                    add_s = joined - alone                                  # sphere on top of the candidate
                    for k, v in (("add_to_sphere", add_c), ("sphere_adds", add_s)):
                        row[k] = np.median(v)
                        row[k + "_p05"], row[k + "_p95"] = np.percentile(v, 5), np.percentile(v, 95)
                        row[k + "_frac_pos"] = (v > 0).mean()
                row["check_vs_heldout_ols"] = chk
                rows.append({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "metric_sensitivity_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--heldout", required=True, help="metric_heldout.py output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--max-layers", type=int, default=14, help="evenly spaced subset of the band")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    models = a.models or ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b", "mixtral8x22b", "mistrallarge123b", "glm45air"]
    os.makedirs(a.out, exist_ok=True)
    if not a.summarize:
        with Pool(a.procs) as p:
            for m in p.imap_unordered(run, [(m, a.indir, a.objects, a.heldout, a.out, a.n_splits, a.max_layers) for m in models]):
                print("done", m, flush=True)
    summarize(a.out, models)


if __name__ == "__main__":
    main()
