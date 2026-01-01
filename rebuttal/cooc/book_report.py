#!/usr/bin/env python
"""Score every harvested book -> books/books_catalog.csv (+ the book sets used downstream)

Per book (from its own cooc_stream.py part):
  tokens, astro-gated tokens, mentions (pure / gated), distinct constellations named,
  constellation mentions per 1k tokens, and the sky-guide score of Storan et al. 2026
  (arXiv:2610.00647, App. G): D = corr(M*_ij, cos theta_ij) over pairs of objects named
  >= 5 times in the book, M* from flat-window (d <= 16) counts with additive smoothing
  alpha = 0.25, one-sided object-permutation p (500 shuffles). Their window was 30 tokens;
  our parts stop at 16.
Duplicates (the same work scanned several times, or in Gutenberg and IA): key = first six
content words of the title + first author token; Gutenberg is preferred, else the copy with
the most mentions. Duplicates are listed but excluded from every set.
Tiers:  sky_guide   D > 0, p < 0.05, >= 15 well-counted objects   (the paper's criterion)
        catalogue   would be sky_guide/astro_dense, but the title says catalogue / almanac /
                    navigation / annals / observations (RA-ordered tables are sky-structured
                    without being sky-tour prose); kept apart so their effect can be tested
        astro_dense not a sky_guide, but >= 1 constellation mention / 1k tokens and >= 15
                    distinct constellations
        other       the rest (general astronomy, almanacs, fiction that matched a title...)

Usage: book_report.py <books_dir> <counts_books_dir> <catalogue_csv> <forms_csv>
"""
import sys, os, csv, re, glob
import numpy as np

ALPHA, MIN_MENT, NPERM = 0.25, 5, 500
RNG = np.random.default_rng(0)
CATALOGUE_RE = re.compile(r"catalog|almanac|navigat|navigator|ephemeris|annals|zones|durchmusterung|"
                          r"observations made|tables|reduction of|results of|epitome|nautical|"
                          r"report of|proceedings|transactions|memoirs of|monthly notices|bulletin", re.I)
STOP = {"the", "a", "an", "of", "and", "or", "with", "in", "on", "to", "for", "by", "its", "their"}


def title_key(title, author):
    w = [x for x in re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower()).split() if x not in STOP][:6]
    a = re.sub(r"[^a-z ]+", " ", (author or "").lower()).split()
    return " ".join(w) + " | " + (a[0] if a else "")


def main():
    bdir, cdir, catcsv, formscsv = sys.argv[1:5]
    cat = list(csv.DictReader(open(catcsv, encoding="utf-8")))
    names = [r["name"] for r in cat]
    xyz = np.array([[float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])] for r in cat])
    ctype = np.array([r["type"] for r in cat])
    fr = list(csv.DictReader(open(formscsv, encoding="utf-8")))
    cand = {}
    for fn in ("candidates.csv", "candidates_rev.csv"):     # two harvesters, one from each end
        fp = os.path.join(bdir, fn)
        if os.path.exists(fp):
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                cand.setdefault("%s_%s" % (r["source"], re.sub(r"[^A-Za-z0-9._-]", "_", r["id"])), r)

    rows, prof = [], {}
    Mmap = None
    for p in sorted(glob.glob(os.path.join(cdir, "part_*.npz"))):
        tag = os.path.basename(p)[5:-4]
        d = np.load(p)
        if Mmap is None:
            forms = [str(f) for f in d["forms"]]
            fobj = {}
            for r in fr:
                fobj.setdefault(r["form"], r["object"])
            oi = {n: i for i, n in enumerate(names)}
            Mmap = np.zeros((len(forms), len(names)))
            for i, f in enumerate(forms):
                Mmap[i, oi[fobj[f]]] = 1
        u = Mmap.T @ d["u_raw"].astype(float)
        ug = Mmap.T @ d["u_art"].astype(float)
        C = Mmap.T @ d["Cd_raw"].sum(2).astype(float) @ Mmap
        C = C + C.T
        np.fill_diagonal(C, 0)
        ntok = int(d["n_tok"])
        isc = ctype == "constellation"
        row = dict(tag=tag, tokens=ntok, tokens_astro=int(d["n_tok_art"]),
                   mentions_pure=int(u.sum()), mentions_gated=int(ug.sum()),
                   const_mentions=int(u[isc].sum()), distinct_const=int((u[isc] > 0).sum()),
                   const_per_1k=round(1000 * u[isc].sum() / max(ntok, 1), 3),
                   well_counted=int((u >= MIN_MENT).sum()), D="", D_p="")
        k = np.where(u >= MIN_MENT)[0]
        if len(k) >= 10 and ntok > 0:
            F = 16.0
            e = 2 * F * np.outer(u[k], u[k]) / ntok
            c = C[np.ix_(k, k)] + ALPHA
            M = 2 * (c - e) / (c + e)
            ii, jj = np.triu_indices(len(k), 1)
            cos = (xyz[k] @ xyz[k].T)[ii, jj]
            m = M[ii, jj]
            if m.std() > 0:
                D = float(np.corrcoef(m, cos)[0, 1])
                cnt = 0
                for _ in range(NPERM):
                    pk = RNG.permutation(len(k))
                    cnt += np.corrcoef(m, (xyz[k][pk] @ xyz[k][pk].T)[ii, jj])[0, 1] >= D
                row.update(D=round(D, 4), D_p=round((cnt + 1) / (NPERM + 1), 4))
        c0 = cand.get(tag, {})
        row.update(source=c0.get("source", tag.split("_")[0]), id=c0.get("id", ""),
                   title=c0.get("title", ""), author=c0.get("author", ""), year=c0.get("year", ""),
                   subjects=c0.get("subjects", ""), url=c0.get("url", ""))
        rows.append(row)
        prof[tag] = u

    # duplicates: (a) same title key, or (b) same content -- two scans of one book have nearly the
    # same object-mention profile and length even when the catalogue metadata disagree
    # ("Olcott, William Tyler" vs "William Tyler Olcott", OCR'd titles like "Spendours")
    parent = {r["tag"]: r["tag"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    bykey = {}
    for r in rows:
        bykey.setdefault(title_key(r["title"], r["author"]), []).append(r["tag"])
    for tags in bykey.values():
        for t in tags[1:]:
            parent[find(t)] = find(tags[0])
    big = [r for r in rows if r["mentions_pure"] >= 100]
    P = np.array([prof[r["tag"]] for r in big]); P = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = P @ P.T
    tok = np.array([r["tokens"] for r in big], float)
    for a, b in zip(*np.where(np.triu(S, 1) >= 0.97)):
        if 0.75 <= tok[a] / tok[b] <= 1.33:
            parent[find(big[a]["tag"])] = find(big[b]["tag"])
    # (c) same TEXT: MinHash over 5-word shingles (64 hashes) of every book with >= 50 mentions;
    # catches reprints and editions whose OCR noise breaks the profile test (Smyth's Cycle x3)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import zlib
    from cooc_stream import paragraphs, strip_gutenberg
    rng = np.random.default_rng(1)
    A_ = rng.integers(1, 2**31 - 1, 64, dtype=np.int64); B_ = rng.integers(0, 2**31 - 1, 64, dtype=np.int64)
    PR = np.int64(2**31 - 1)
    mh_rows, sigs = [], []
    for r in rows:
        if r["mentions_pure"] < 50:
            continue
        c0 = cand.get(r["tag"], {})
        path = c0.get("path") or os.path.join(bdir, "raw", r["tag"] + ".txt")
        if not os.path.exists(path):
            continue
        txt = open(path, "rb").read().decode("utf-8", "replace")
        toks = [w for p in paragraphs(strip_gutenberg(txt), True) for w in p]
        h = np.array([zlib.crc32(" ".join(toks[i:i + 5]).encode()) for i in range(0, max(len(toks) - 4, 0))],
                     np.int64)
        if len(h) < 100:
            continue
        sigs.append(((np.outer(h, A_) + B_) % PR).min(0)); mh_rows.append(r)
    if sigs:
        SG = np.array(sigs)
        for a in range(len(SG)):
            jac = (SG[a + 1:] == SG[a]).mean(1)
            for b in np.where(jac >= 0.25)[0]:
                parent[find(mh_rows[a + 1 + b]["tag"])] = find(mh_rows[a]["tag"])
    best = {}
    for r in rows:
        g = find(r["tag"]); r["dup_key"] = g
        rank = (r["source"] == "gutenberg", r["mentions_pure"])
        if g not in best or rank > best[g][0]:
            best[g] = (rank, r["tag"])
    for r in rows:
        keep = best[r["dup_key"]][1]
        r["dup_of"] = "" if keep == r["tag"] else keep
        sg = r["D"] != "" and r["D"] > 0 and r["D_p"] < 0.05 and r["well_counted"] >= 15
        dense = r["const_per_1k"] >= 1.0 and r["distinct_const"] >= 15
        cata = bool(CATALOGUE_RE.search(r["title"]))
        r["tier"] = ("duplicate" if r["dup_of"] else ("catalogue" if cata and (sg or dense) else
                     ("sky_guide" if sg else ("astro_dense" if dense else "other"))))

    rows.sort(key=lambda r: (["sky_guide", "catalogue", "astro_dense", "other", "duplicate"].index(r["tier"]),
                             -(r["D"] if r["D"] != "" else -9)))
    cols = ["tag", "tier", "source", "id", "title", "author", "year", "tokens", "tokens_astro",
            "mentions_pure", "mentions_gated", "const_mentions", "distinct_const", "const_per_1k",
            "well_counted", "D", "D_p", "dup_of", "subjects", "url"]
    with open(os.path.join(bdir, "books_catalog.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    for tier_set, tiers in (("sky_guide", {"sky_guide"}), ("catalogue", {"catalogue"}),
                            ("guides_dense", {"sky_guide", "astro_dense"}),
                            ("all_books", {"sky_guide", "catalogue", "astro_dense", "other"})):
        with open(os.path.join(bdir, "set_%s.txt" % tier_set), "w") as fh:
            fh.write("\n".join(os.path.join(cdir, "part_%s.npz" % r["tag"]) for r in rows if r["tier"] in tiers) + "\n")
    from collections import Counter
    print(Counter(r["tier"] for r in rows))
    for t in ("sky_guide", "catalogue", "astro_dense", "other"):
        sel = [r for r in rows if r["tier"] == t]
        print(t, "books %d  tokens %.1fM  pure mentions %d  gated %d" % (
            len(sel), sum(r["tokens"] for r in sel) / 1e6, sum(r["mentions_pure"] for r in sel),
            sum(r["mentions_gated"] for r in sel)))


if __name__ == "__main__":
    main()
