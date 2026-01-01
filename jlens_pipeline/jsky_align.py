#!/usr/bin/env python
"""Is the fitted 3-dim sky subspace among the directions the Jacobian lens singles out?

Conventions, verified against the jlens source:
  * J_l maps a SOURCE residual into the final-layer basis: h_final ~= J_l @ h_l.
    Rows are target dims, columns source dims, so the operator that contracts the
    output index and lives on the source side is J^T J.
  * jlens hooks record BLOCK OUTPUTS, so jlens source index j is hidden_states[j+1],
    i.e. PCA layer j+1; jlens target L-1 is PCA layer L.  We report PCA layers
    1..L-1, which excludes the embedding (PCA layer 0) and the final layer.
  * PCA scores are c = ((h - feat_mean)/feat_std - pca_mean) @ components.T, so
        coord displacement dc -> raw  dh = feat_std * (components.T @ dc)
        coord covector    a  -> raw  w  = (components.T @ a) / feat_std
    J acts on raw unstandardized residuals, so the back-map matters.

Per layer, for the ENCODING subspace (image of the sphere, coords ~ Y) and the
DECODING subspace (probe covectors, Y ~ coords):
    g_i / sigma_1   normalized transport gains of the 3-dim subspace
    rank_g1         number of singular values of J at least g1, and /d
    A_l             (||J V||_F^2 / 3) / (||J||_F^2 / d); expectation 1 under a
                    Haar-random 3-subspace of R^d
    T_l             ||P_final J V||_F^2 / ||J V||_F^2, sky-to-sky transport

Control: random 3-subspaces built the same way from the same top-k PCA span.
"""
import argparse, csv, json, os, time
import numpy as np
import torch

PAIRS = [
    ("glm-4.5-air", "glm45air_pca128.npz"),
    ("gpt-oss-120b", "gptoss120b_pca128.npz"),
    ("llama3.3-70b-it", "llama33_70b_pca128.npz"),
    ("mistral-large-2411", "mistrallarge123b_pca128.npz"),
    ("mixtral-8x22b-it", "mixtral8x22b_pca128.npz"),
    ("qwen3-235b-a22b-thinking", "qwen235b_pca128.npz"),
    ("qwen3-32b", "qwen32b_pca128.npz"),
]
KS = [8, 16, 32]

COLS = ["model", "slug", "pca_file", "d_model", "n_layers", "layer_pca", "layer_jlens",
        "depth_frac", "subspace", "k", "n_null",
        "sky_r2_loo", "sky_r2_in", "ang_err_med_deg",
        "sigma1", "fro_norm", "eff_dim_pr",
        "g1", "g2", "g3", "g1_s1", "g2_s1", "g3_s1",
        "rank_g1", "rank_pct_g1", "A_l", "T_l",
        "A_null_med", "A_null_p95", "A_p",
        "T_null_med", "T_null_p95", "T_p",
        "g1_s1_null_med", "g1_s1_null_p95", "g1_s1_p",
        "rank_pct_null_med", "rank_pct_null_p05", "rank_pct_p"]


def log(*a):
    print("[" + time.strftime("%H:%M:%S") + "]", *a, flush=True)


# ------------------------------------------------------------------ sky fits
def object_means(pca, obj_ids, n_obj):
    """Average the per-prompt PCA scores within each object -> (n_obj, n_pca_layers, K)."""
    out = np.zeros((n_obj, pca.shape[1], pca.shape[2]), np.float64)
    for o in range(n_obj):
        out[o] = pca[obj_ids == o].astype(np.float64).mean(0)
    return out


def loo_stats(X, Y):
    """Leave-one-out R^2 of the sky fit plus median angular error, via the hat matrix."""
    n = X.shape[0]
    Xd = np.concatenate([np.ones((n, 1)), X], 1)
    H = Xd @ np.linalg.pinv(Xd.T @ Xd) @ Xd.T
    h = np.clip(np.diag(H), 0.0, 1.0 - 1e-9)
    resid = Y - H @ Y
    loo_resid = resid / (1.0 - h)[:, None]
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    yloo = Y - loo_resid
    nrm = np.linalg.norm(yloo, axis=1, keepdims=True)
    cos = np.clip((yloo / np.maximum(nrm, 1e-12) * Y).sum(1), -1.0, 1.0)
    return (1 - (loo_resid ** 2).sum() / ss_tot,
            1 - (resid ** 2).sum() / ss_tot,
            float(np.median(np.degrees(np.arccos(cos)))))


def sky_bases(Xk, Y, Ck, s):
    """Raw-space (d,3) bases for the encoding and decoding sky subspaces at one layer."""
    Xc, Yc = Xk - Xk.mean(0), Y - Y.mean(0)
    G = np.linalg.lstsq(Yc, Xc, rcond=None)[0]        # (3,k)  coords ~ Y
    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0]        # (k,3)  Y ~ coords
    enc = s[:, None] * (Ck.T @ G.T)                   # (d,3)
    dec = (Ck.T @ A) / s[:, None]                     # (d,3)
    return enc, dec


def null_bases(rng, n_null, k, Ck, s, kind):
    """n_null random 3-subspaces drawn inside the same top-k PCA span, as (d, 3*n_null)."""
    if kind == "enc":
        Gn = rng.standard_normal((n_null, 3, k))                       # (n,3,k)
        M = np.transpose(Gn, (2, 0, 1)).reshape(k, 3 * n_null)
        return s[:, None] * (Ck.T @ M)
    An = rng.standard_normal((n_null, k, 3))                           # (n,k,3)
    M = np.transpose(An, (1, 0, 2)).reshape(k, 3 * n_null)
    return (Ck.T @ M) / s[:, None]


# ------------------------------------------------------------------ gpu stats
def batch_stats(J, Vf, P_all, n, fro2, d, sigma_asc):
    """Statistics for n subspaces whose (non-orthonormal) bases are stacked in P_all (d,3n).

    Orthonormalisation is folded in analytically: with V = P (P^T P)^{-1/2},
    ||J V||_F^2 = tr((P^T P)^{-1} P^T J^T J P), and likewise for the projected part.
    """
    Y1 = J @ P_all                                     # (d,3n)
    Y2 = Vf.T @ Y1                                     # (3,3n)
    P3 = P_all.T.reshape(n, 3, d)                      # P3[i] = P_i^T
    Sp = torch.bmm(P3, P3.transpose(1, 2))             # P^T P
    Y1m = Y1.T.reshape(n, 3, d)
    Gm = torch.bmm(Y1m, Y1m.transpose(1, 2))           # (J P)^T (J P)
    Y2m = Y2.T.reshape(n, 3, 3)
    Q2 = torch.bmm(Y2m, Y2m.transpose(1, 2))           # (Vf^T J P)^T (Vf^T J P)

    e, U = torch.linalg.eigh(Sp)
    e = e.clamp_min(1e-30)
    Sinv = U @ torch.diag_embed(1.0 / e) @ U.transpose(1, 2)
    Sisq = U @ torch.diag_embed(e.rsqrt()) @ U.transpose(1, 2)

    tr_g = torch.diagonal(Sinv @ Gm, dim1=1, dim2=2).sum(-1)   # ||J V||_F^2
    tr_q = torch.diagonal(Sinv @ Q2, dim1=1, dim2=2).sum(-1)
    lam = torch.linalg.eigvalsh(Sisq @ Gm @ Sisq).clamp_min(0).flip(-1)   # descending
    g = lam.sqrt()
    rank = d - torch.searchsorted(sigma_asc, g[:, 0].contiguous())
    return dict(A=(tr_g / 3.0) / (fro2 / d),
                T=tr_q / tr_g.clamp_min(1e-30),
                g=g, rank=rank.double())


def run_model(model_slug, pca_file, n_null, seed, device, lens_dir, pca_dir, out_dir):
    lens_path = os.path.join(lens_dir, model_slug + "_jacobian_lens.pt")
    meta = json.load(open(os.path.join(lens_dir, model_slug + "_meta.json")))
    L, d = meta["n_layers"], meta["d_model"]

    z = np.load(os.path.join(pca_dir, pca_file), allow_pickle=True)
    assert z["components"].shape == (L + 1, 128, d), (z["components"].shape, L, d)
    Y = z["Yunit"].astype(np.float64)
    n_obj = Y.shape[0]
    Cobj = object_means(z["pca"], z["obj_ids"], n_obj)
    comps, fstd = z["components"], z["feat_std"]
    log(model_slug + ": L=%d d=%d objects=%d pca_layers=%d" % (L, d, n_obj, comps.shape[0]))

    log(model_slug + ": loading lens %.1f GiB" % (os.path.getsize(lens_path) / 2 ** 30))
    Jd = torch.load(lens_path, map_location="cpu", weights_only=True)["J"]
    assert sorted(Jd) == list(range(L - 1)), (min(Jd), max(Jd), len(Jd))

    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    # final-layer sky subspaces (PCA layer L = the lens target), orthonormalised once
    Vf = {}
    for k in KS:
        enc, dec = sky_bases(Cobj[:, L, :k], Y, comps[L, :k].astype(np.float64),
                             fstd[L].astype(np.float64))
        for kind, B in (("enc", enc), ("dec", dec)):
            Vf[(k, kind)] = torch.linalg.qr(
                torch.as_tensor(B, dtype=torch.float32, device=dev))[0]

    rows = []
    for li in range(1, L):                       # PCA layers 1..L-1
        j = li - 1                               # jlens source index
        J = Jd[j].to(dev, torch.float32)
        Gram = J.T @ J
        evals = torch.linalg.eigvalsh(Gram).clamp_min(0)      # ascending
        del Gram
        sigma_asc = evals.sqrt().contiguous()
        fro2 = float(evals.sum())
        sigma1 = float(sigma_asc[-1])
        eff_dim = fro2 ** 2 / float((evals ** 2).sum())
        del evals
        torch.cuda.empty_cache()

        Ck_full = comps[li].astype(np.float64)
        s = fstd[li].astype(np.float64)
        for k in KS:
            Ck = Ck_full[:k]
            r2_loo, r2_in, ang = loo_stats(Cobj[:, li, :k], Y)
            enc, dec = sky_bases(Cobj[:, li, :k], Y, Ck, s)
            for kind, B in (("enc", enc), ("dec", dec)):
                Po = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float32, device=dev)
                obs = batch_stats(J, Vf[(k, kind)], Po, 1, fro2, d, sigma_asc)
                Pn = torch.as_tensor(np.ascontiguousarray(
                    null_bases(rng, n_null, k, Ck, s, kind)),
                    dtype=torch.float32, device=dev)
                nul = batch_stats(J, Vf[(k, kind)], Pn, n_null, fro2, d, sigma_asc)

                A, T = float(obs["A"][0]), float(obs["T"][0])
                g = obs["g"][0].tolist()
                g1s1 = g[0] / sigma1
                rank = float(obs["rank"][0])
                nA = nul["A"].cpu().numpy()
                nT = nul["T"].cpu().numpy()
                nG = (nul["g"][:, 0] / sigma1).cpu().numpy()
                nR = (nul["rank"] / d).cpu().numpy()
                rows.append(dict(
                    model=meta["model"], slug=model_slug, pca_file=pca_file,
                    d_model=d, n_layers=L, layer_pca=li, layer_jlens=j,
                    depth_frac=j / max(L - 2, 1), subspace=kind, k=k, n_null=n_null,
                    sky_r2_loo=r2_loo, sky_r2_in=r2_in, ang_err_med_deg=ang,
                    sigma1=sigma1, fro_norm=fro2 ** 0.5, eff_dim_pr=eff_dim,
                    g1=g[0], g2=g[1], g3=g[2],
                    g1_s1=g1s1, g2_s1=g[1] / sigma1, g3_s1=g[2] / sigma1,
                    rank_g1=rank, rank_pct_g1=rank / d, A_l=A, T_l=T,
                    A_null_med=float(np.median(nA)), A_null_p95=float(np.quantile(nA, .95)),
                    A_p=float((nA >= A).mean()),
                    T_null_med=float(np.median(nT)), T_null_p95=float(np.quantile(nT, .95)),
                    T_p=float((nT >= T).mean()),
                    g1_s1_null_med=float(np.median(nG)),
                    g1_s1_null_p95=float(np.quantile(nG, .95)),
                    g1_s1_p=float((nG >= g1s1).mean()),
                    rank_pct_null_med=float(np.median(nR)),
                    rank_pct_null_p05=float(np.quantile(nR, .05)),
                    rank_pct_p=float((nR <= rank / d).mean()),
                ))
                del Po, Pn
        del J
        torch.cuda.empty_cache()
        if li % 10 == 0 or li == L - 1:
            log(model_slug + ": layer %d/%d" % (li, L - 1))

    out = os.path.join(out_dir, model_slug + "_jsky.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    log(model_slug + ": wrote " + out + " (%d rows)" % len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=[s for s, _ in PAIRS],
                    help="lens slug to run; one model per process")
    ap.add_argument("--lensdir", default=os.environ.get("LENS_DIR", "lenses"),
                    help="holds <slug>_jacobian_lens.pt and <slug>_meta.json "
                         "(default: $LENS_DIR, else ./lenses)")
    ap.add_argument("--pcadir", default=os.environ.get("PCA_DIR", "pca"),
                    help="holds the <name>_pca128.npz extractions "
                         "(default: $PCA_DIR, else ./pca)")
    ap.add_argument("--outdir", default=os.environ.get("JSKY_DIR", "jsky"),
                    help="where <slug>_jsky.csv is written (default: $JSKY_DIR, else ./jsky)")
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.outdir, exist_ok=True)
    run_model(a.only, dict(PAIRS)[a.only], a.n_null, a.seed, a.device,
              a.lensdir, a.pcadir, a.outdir)


if __name__ == "__main__":
    main()
