#!/usr/bin/env python
"""Split halves of every nested corpus stage -> <out>/<stage>_h0.npz, <stage>_h1.npz

Same stages as merge_sets.py (web_f1_16 ... web_f1_1, the pilot web prefix) and merge_ext.py
(ext_f1_16 ... ext_f1_2, ext_all = pilot web + a prefix of the extended parts). Half 0 / 1 holds the
even / odd positions of the stage's parts, counting the pilot web parts and the extended parts
separately, exactly as merge_ext.py builds ext_h0 / ext_h1 (so ext_all_h0 == ext_h0). One pass.

Usage: merge_stage_halves.py --pilot <cooc_pilot dir> --ext <counts_ext dir> --tasks tasks_ext.txt --out <dir>
"""
import argparse
import os

from merge_ext import Acc


def main():
    ap = argparse.ArgumentParser()
    for k in ("pilot", "ext", "tasks", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    web = [l.split()[2] for l in open(os.path.join(a.pilot, "tasks_web.txt")) if l.strip() and not l.split()[2].startswith("wiki_")]
    web = [os.path.join(a.pilot, "counts_web", "part_%s.npz" % t) for t in web]
    miss = [p for p in web if not os.path.exists(p)]
    if miss:
        raise SystemExit("%d web parts missing (stages would not be nested), first %s" % (len(miss), miss[0]))
    ext = [os.path.join(a.ext, "part_%s.npz" % l.split()[2]) for l in open(a.tasks) if l.strip()]
    h = [Acc(), Acc()]

    def snap(name):
        for j in (0, 1):
            h[j].save(os.path.join(a.out, "%s_h%d.npz" % (name, j)))

    wst = {len(web) * num // den: "web_f%d_%d" % (num, den) for num, den in ((1, 16), (1, 8), (1, 4), (1, 2), (1, 1))}
    for i, p in enumerate(web):
        h[i % 2].add(p)
        if i + 1 in wst:
            snap(wst[i + 1])
    est = {len(ext) * n // 16: "ext_f1_%d" % (16 // n) for n in (1, 2, 4, 8)}
    est[len(ext)] = "ext_all"
    for i, p in enumerate(ext):
        h[i % 2].add(p)
        if i + 1 in est:
            snap(est[i + 1])


if __name__ == "__main__":
    main()
