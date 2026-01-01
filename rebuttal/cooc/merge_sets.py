#!/usr/bin/env python
"""Merge cooc_stream.py parts into the corpus sets the pilot evaluates -> sets/<name>.npz

Web parts are taken in task order (tasks_web.txt interleaves C4 and FineWeb by position),
so a prefix is a mixed sample and the learning curve is nested: web_f1_16 c web_f1_8 c ...
Halves (even / odd position) give split-half reliability. Books come from book_report.py's
set_*.txt lists; the four guides of Storan et al. 2026 are matched by Gutenberg title.

Only the variants the evaluation reads are kept (raw = pure, art = gated, artnl for reference).
Usage: merge_sets.py [only_name ...]
"""
import sys, os, glob, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = ("Cd_raw", "Cd_art", "Cd_artnl", "u_raw", "u_art", "u_artnl", "n_tok", "n_tok_art", "n_tok_artnl",
        "n_art", "n_mention_raw", "n_mention_art")
PAPER4 = ("field book of the stars", "round the year with the stars", "astronomy with an opera-glass",
          "astronomy for young australians")


def merge(paths, out):
    tot, forms, n = {}, None, 0
    for p in paths:
        if not os.path.exists(p):
            continue
        d = np.load(p)
        if forms is None:
            forms = d["forms"]
        for k in KEEP:
            tot[k] = tot.get(k, 0) + (d[k].astype(np.int64) if d[k].ndim else int(d[k]))
        n += 1
    if not n:
        print("empty:", out); return
    np.savez_compressed(out, forms=forms, n_parts=np.int64(n), **{k: (np.int64(v) if np.isscalar(v) else v)
                                                                   for k, v in tot.items()})
    print("%-28s %4d parts  %6.2fB tokens  pure %8d  gated %7d" % (
        os.path.basename(out), n, tot["n_tok"] / 1e9, tot["n_mention_raw"], tot["n_mention_art"]), flush=True)


def main():
    only = set(sys.argv[1:])
    os.makedirs(os.path.join(HERE, "sets"), exist_ok=True)
    sets = {}
    wiki = sorted(glob.glob(os.path.join(HERE, "counts_wiki", "part_wiki_*.npz")))
    sets["wiki"] = wiki
    sets["wiki_h0"], sets["wiki_h1"] = wiki[0::2], wiki[1::2]
    order = [l.split()[2] for l in open(os.path.join(HERE, "tasks_web.txt")) if not l.split()[2].startswith("wiki_")]
    done = [t for t in order if os.path.exists(os.path.join(HERE, "counts_web", "part_%s.npz" % t))]
    # nested prefix: the first k tasks in order that are ALL done (no holes)
    pref = []
    for t in order:
        if t not in done:
            break
        pref.append(os.path.join(HERE, "counts_web", "part_%s.npz" % t))
    for num, den in ((1, 16), (1, 8), (1, 4), (1, 2), (1, 1)):
        k = len(order) * num // den
        if k <= len(pref):
            sets["web_f%d_%d" % (num, den)] = pref[:k]
    sets["web_h0"], sets["web_h1"] = pref[0::2], pref[1::2]
    sets["c4"] = [p for p in pref if "part_c4_" in p]
    sets["fineweb"] = [p for p in pref if "part_fw_" in p]
    bdir = os.path.join(HERE, "books")
    for s in ("sky_guide", "catalogue", "guides_dense", "all_books"):
        f = os.path.join(bdir, "set_%s.txt" % s)
        if os.path.exists(f):
            ps = [l.strip() for l in open(f) if l.strip()]
            sets["books_" + s] = ps
            sets["books_%s_h0" % s], sets["books_%s_h1" % s] = ps[0::2], ps[1::2]
    cat = os.path.join(bdir, "books_catalog.csv")
    if os.path.exists(cat):
        rows = list(csv.DictReader(open(cat, encoding="utf-8")))
        sets["books_paper4"] = [os.path.join(HERE, "counts_books", "part_%s.npz" % r["tag"]) for r in rows
                                if r["source"] == "gutenberg" and any(t in r["title"].lower() for t in PAPER4)]
    for name, ps in sets.items():
        if only and name not in only:
            continue
        merge(ps, os.path.join(HERE, "sets", name + ".npz"))
    with open(os.path.join(HERE, "sets", "web_prefix_done.txt"), "w") as fh:
        fh.write("%d of %d web tasks done in order\n" % (len(pref), len(order)))


if __name__ == "__main__":
    main()
