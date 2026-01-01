#!/usr/bin/env python
"""QWEM co-occurrence embedding -> an RDM table for fit_models.py.

Construction follows cooccur_rdm/cooccur_rdm.py: from the co-occurrence matrix

    M*_ij = 2 (c_ij - e_ij) / (c_ij + e_ij),   e_ij = 2 S_f u_i u_j / N

(count form of the closed-form QWEM solution; S_f = sum_d f(d), f(d) = 17 - d, L = 16),
take the block of words kept, symmetrize, eigendecompose, and read the embedding off the
positive spectrum: W = V_+ sqrt(Lambda_+). The RDM is the squared Euclidean distance
between rows of W. Words with fewer than --min-u corpus occurrences are dropped (as there),
and the embedding is a JOINT factorization: every kept word takes part, the table is written
for the catalogue objects among them, so two objects that never co-occur directly are still
placed by what they share.

Two input paths:
  --counts <cooc_counts_wikipedia.npz>   counts of the 188 catalogue objects made by
        cooc_count.py (surface forms merged per object; --variant raw|para|art|artnl picks
        the context gate: raw = the plain corpus counting that M* assumes, art = astronomical
        paragraph in an astronomy-dense article with the confirm-word test for polysemous
        names, artnl = art minus lists and tables). N in e_ij is the token count of the
        contexts that variant counted over (n_tok_<variant>), not the whole corpus, or the
        gated counts would look far above chance everywhere.
  --mstar <csv>   a ready M* matrix in the format written by build_cooc_mstar.py
        (word, kind, unigram, ...) or by cooccur/cooccur_mstar.py (name, unigram, ...)

Output: <out>.csv, a 188 x 188 table of squared distances indexed by catalogue name, NaN
where the object is not covered by that vocabulary.

Usage: build_qwem_rdm.py (--counts <npz> | --mstar <csv>) --out rdms/<name>.csv [--min-u 40]
       [--variant raw] [--forms all|canonical]
"""
import os, re, csv, argparse, unicodedata
import numpy as np, pandas as pd
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LW = [0.0] + [17.0 - d for d in range(1, 17)]        # f(d) = 17 - d over L = 16


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def mstar_from_counts(npz, variant, forms_policy, names):
    """-> words (catalogue names), unigram, M* over the objects that were counted"""
    d = np.load(npz)
    sf = pd.read_csv(os.path.join(HERE, "graphs", "surface_forms.csv"), keep_default_na=False)
    kinds = {"all": {"canonical", "bayer", "alias"}, "canonical": {"canonical"}}[forms_policy]
    forms = [str(f) for f in d["forms"]]; fidx = {f: i for i, f in enumerate(forms)}
    Mmap = np.zeros((len(forms), len(names)))
    oidx = {n: i for i, n in enumerate(names)}
    for r in sf.itertuples():
        if r.kind in kinds:
            Mmap[fidx[r.form], oidx[r.object]] = 1
    C = np.tensordot(d["Cd_" + variant].astype(float), np.array(LW[1:]), axes=([2], [0]))
    C = Mmap.T @ (C + C.T) @ Mmap                    # ordered pairs -> symmetric object counts
    np.fill_diagonal(C, 0.0)
    u = Mmap.T @ d["u_" + variant].astype(float)
    ntok = "n_tok" if variant == "raw" else "n_tok_" + variant
    assert ntok in d.files, "%s missing: N must be the tokens the gated u were counted over" % ntok
    return u, C, float(d[ntok])


def mstar_from_csv(path, names):
    df = pd.read_csv(path)
    key = "word" if "word" in df.columns else "name"
    words = [norm(str(w).replace("_", " ").replace(" constellation", "")) for w in df[key]]
    M = df[df.columns[3 if key == "word" else 2:]].to_numpy(float)
    return words, df.unigram.to_numpy(float), 0.5 * (M + M.T)


def embed(M, keep):
    Ms = 0.5 * (M[np.ix_(keep, keep)] + M[np.ix_(keep, keep)].T)
    w, V = np.linalg.eigh(Ms)
    pos = w > 1e-9
    return V[:, pos] * np.sqrt(w[pos])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts"); ap.add_argument("--mstar")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-u", type=float, default=40)
    ap.add_argument("--variant", default="art")
    ap.add_argument("--forms", default="all")
    a = ap.parse_args()

    cat = pd.read_csv(paths.catalogue())
    names = cat.name.tolist()
    obj_key = [norm(n.replace(" constellation", "")) for n in names]

    if a.counts:
        u, C, N = mstar_from_counts(a.counts, a.variant, a.forms, names)
        e = 2.0 * sum(LW) * np.outer(u, u) / N
        M = 2.0 * (C - e) / (C + e + 1e-12)
        words = obj_key
    else:
        words, u, M = mstar_from_csv(a.mstar, names)
    np.fill_diagonal(M, 0.0)

    keep = [i for i in range(len(words)) if u[i] >= a.min_u]
    W = embed(M, keep)
    row_of = {words[k]: r for r, k in enumerate(keep)}
    have = [i for i, k in enumerate(obj_key) if k in row_of]
    X = W[[row_of[obj_key[i]] for i in have]]
    g = (X * X).sum(1)
    D = np.full((len(names), len(names)), np.nan)
    D[np.ix_(have, have)] = np.maximum(g[:, None] + g[None, :] - 2 * X @ X.T, 0)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(D, index=names, columns=names).to_csv(a.out, float_format="%.8g")
    ntyp = cat.type[have].value_counts().to_dict()
    print("%s: %d words in vocabulary, %d kept (u >= %g), embedding dim %d (positive eigenvalues)"
          % (a.out, len(words), len(keep), a.min_u, W.shape[1]))
    print("  covers %d/%d catalogue objects %s; %d constellation pairs"
          % (len(have), len(names), ntyp, int((cat.type[have] == "constellation").sum() *
                                              ((cat.type[have] == "constellation").sum() - 1) / 2)))
    miss = [names[i] for i in range(len(names)) if i not in have and cat.type[i] == "constellation"]
    if miss:
        print("  constellations NOT covered:", miss)


if __name__ == "__main__":
    main()
