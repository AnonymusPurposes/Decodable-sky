#!/usr/bin/env python
"""Object-level co-occurrence graph from graphs/cooc_counts_wikipedia.npz.

Two graphs are exported: `A` from variant "art" (astronomical paragraph + astronomy-dense
article) and `A_nolist` from variant "artnl", which additionally drops mentions in lists and
tables. Both: two objects are connected iff they co-occur within 10 words at least once,
surface forms "all" (catalogue name + Bayer designation + curated aliases), count >= 1.
All variants are stored alongside, as symmetric 188x188 counts:
  C<W>_<variant>_<policy>   pair events at distance <= W   (W in 10, 16)
  C10_artlist_<policy>      the part of C10_art inside name-dense stretches (lists)
  D_<variant>_<policy>      articles mentioning both
  u_<variant>_<policy>      mentions per object
variant in raw / para / art / artnl;  policy in all / canonical.

Also prints diagnostics: coverage, edge density by type block, isolated nodes, and sky
signal -- angular separation of edges vs non-edges, share of constellation-constellation
edges that are bordering pairs, share of star-constellation edges that are membership.

Outputs: graphs/graph_cooccurrence.npz, graphs/graph_cooccurrence_adjacency.csv,
         graphs/graph_cooccurrence_summary.csv
"""
import os
import numpy as np, pandas as pd
import paths
import scipy.sparse.csgraph as cg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G = os.path.join(HERE, "graphs")
VARIANTS = ("raw", "para", "art", "artnl")


def main():
    d = np.load(os.path.join(G, "cooc_counts_wikipedia.npz"))
    forms = [str(f) for f in d["forms"]]; S = len(forms); fidx = {f: i for i, f in enumerate(forms)}
    sf = pd.read_csv(os.path.join(G, "surface_forms.csv"), keep_default_na=False)
    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist(); types = cat.type.to_numpy(); N = len(names); oidx = {n: i for i, n in enumerate(names)}
    Y = cat[["x_sinA", "y_cosAsinL", "z_cosAcosL"]].to_numpy()
    ang = np.degrees(np.arccos(np.clip(Y @ Y.T, -1, 1)))
    cg_npz = np.load(os.path.join(G, "graph_constellation.npz"))
    border = cg_npz["border_deg"] > 0; const_of = cg_npz["const_of"]

    M = {}
    for pol, kinds in (("all", {"canonical", "bayer", "alias"}), ("canonical", {"canonical"})):
        m = np.zeros((S, N))
        for r in sf.itertuples():
            if r.kind in kinds:
                m[fidx[r.form], oidx[r.object]] = 1
        M[pol] = m

    def obj(Cff, pol, sym=True):
        C = Cff + Cff.T if sym else Cff
        O = M[pol].T @ C @ M[pol]
        np.fill_diagonal(O, 0)
        return O

    out = dict(names=np.array(names), types=types.astype(str))
    for pol in M:
        for v in VARIANTS:
            Cd = d["Cd_" + v].astype(float)
            for W in (10, 16):
                out["C%d_%s_%s" % (W, v, pol)] = obj(Cd[:, :, :W].sum(2), pol)
            out["D_%s_%s" % (v, pol)] = obj(d["D_" + v].astype(float), pol)
            out["u_%s_%s" % (v, pol)] = M[pol].T @ d["u_" + v]
        out["C10_artlist_%s" % pol] = obj(d["Cd_art_list"][:, :, :10].sum(2).astype(float), pol)
    A = (out["C10_art_all"] >= 1).astype(np.int8)
    out["A"] = A
    out["A_nolist"] = (out["C10_artnl_all"] >= 1).astype(np.int8)

    iu = np.triu_indices(N, 1)
    ti, tj = types[iu[0]], types[iu[1]]
    blocks = {"const-const": (ti == "constellation") & (tj == "constellation"),
              "star-star": (ti == "star") & (tj == "star"),
              "star-const": ((ti == "star") & (tj == "constellation")) | ((ti == "constellation") & (tj == "star")),
              "other-*": (ti == "other") | (tj == "other")}
    member = (const_of[iu[0]] == iu[1]) | (const_of[iu[1]] == iu[0])
    print("corpus: %d articles >= 200 tok, %.2f B tokens; articles with a mention %d, astronomy-dense %d"
          % (d["n_art"], d["n_tok"] / 1e9, d["n_art_mention"], d["n_art_dense"]))
    print("mentions raw %d / para %d / art %d; confirm-word rejections %d\n"
          % (d["n_mention_raw"], d["n_mention_para"], d["n_mention_art"], d["n_confirm_fail"]))
    rows = []
    for pol in M:
        for v in VARIANTS:
            for thr in (1, 2, 5):
                C = out["C10_%s_%s" % (v, pol)]; E = C[iu] >= thr
                u = out["u_%s_%s" % (v, pol)]
                Ab = (C >= thr).astype(int)
                ncomp, lab = cg.connected_components(Ab, directed=False)
                deg = Ab.sum(1)
                cb = np.where(types == "constellation")[0]          # the 88-node block that
                Ac = Ab[np.ix_(cb, cb)]                             # fit_models actually scores
                ncomp_c, lab_c = cg.connected_components(Ac, directed=False)
                rec = dict(policy=pol, variant=v, window=10, threshold=thr,
                           pair_events=int(C[iu].sum()), edges=int(E.sum()), density=E.mean(),
                           isolated=int((deg == 0).sum()), components=ncomp,
                           largest_component=int(np.bincount(lab).max()),
                           const_edges=int(Ac[np.triu_indices(len(cb), 1)].sum()),
                           const_isolated=int((Ac.sum(1) == 0).sum()), const_components=ncomp_c,
                           const_largest_component=int(np.bincount(lab_c).max()),
                           objects_never_mentioned=int((u == 0).sum()),
                           ang_edge_median=np.median(ang[iu][E]) if E.any() else np.nan,
                           ang_nonedge_median=np.median(ang[iu][~E]),
                           cc_edges_bordering=(E & blocks["const-const"] & border[iu]).sum() / max(1, (E & blocks["const-const"]).sum()),
                           cc_bordering_base_rate=border[iu][blocks["const-const"]].mean(),
                           sc_edges_membership=(E & blocks["star-const"] & member).sum() / max(1, (E & blocks["star-const"]).sum()),
                           sc_membership_base_rate=member[blocks["star-const"]].mean(),
                           list_share_of_events=(out["C10_artlist_%s" % pol][iu].sum() / max(1, C[iu].sum())) if v == "art" else np.nan)
                for b, mask in blocks.items():
                    rec["density_" + b] = E[mask].mean()
                rows.append(rec)
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = ["policy", "variant", "threshold", "pair_events", "edges", "density", "isolated", "components",
            "const_edges", "const_isolated", "const_components", "objects_never_mentioned",
            "ang_edge_median", "ang_nonedge_median", "cc_edges_bordering",
            "cc_bordering_base_rate", "sc_edges_membership", "sc_membership_base_rate", "list_share_of_events"]
    print(summ[show].round(3).to_string(index=False))
    print("\ndensity by block:")
    print(summ[["policy", "variant", "threshold"] + ["density_" + b for b in blocks]].round(3).to_string(index=False))

    u = out["u_art_all"]; deg = A.sum(1)
    order = np.argsort(u)
    print("\ndefault graph (art, W=10, all forms, >=1): %d edges; least-mentioned objects:" % (A[iu].sum()))
    print("   " + "; ".join("%s u=%d deg=%d" % (names[i].replace(" constellation", ""), u[i], deg[i]) for i in order[:25]))
    print("isolated:", [names[i] for i in np.where(deg == 0)[0]])
    C = out["C10_art_all"]; top = np.argsort(-C[iu])[:25]
    print("top pairs:", "; ".join("%s-%s %d (%.0f deg)" % (names[iu[0][k]].replace(" constellation", ""),
                                                          names[iu[1][k]].replace(" constellation", ""),
                                                          C[iu][top][n], ang[iu][k]) for n, k in enumerate(top)))

    An = out["A_nolist"]; cb = np.where(types == "constellation")[0]
    for tag, Ax in (("art", A), ("artnl", An)):
        sub = Ax[np.ix_(cb, cb)]
        nc, _ = cg.connected_components(sub, directed=False)
        print("%-6s graph: %d edges (188), %d edges among the 88 constellations, %d component(s) there, "
              "%d isolated constellations" % (tag, Ax[iu].sum(), sub[np.triu_indices(len(cb), 1)].sum(),
                                              nc, int((sub.sum(1) == 0).sum())))
    np.savez_compressed(os.path.join(G, "graph_cooccurrence.npz"), **out)
    pd.DataFrame(A, index=names, columns=names).to_csv(os.path.join(G, "graph_cooccurrence_adjacency.csv"))
    summ.to_csv(os.path.join(G, "graph_cooccurrence_summary.csv"), index=False)
    print("saved graph_cooccurrence.npz / _adjacency.csv / _summary.csv")


if __name__ == "__main__":
    main()
