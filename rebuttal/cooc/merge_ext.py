#!/usr/bin/env python
"""Merge the extended FineWeb counts into nested corpus stages -> <out>/<name>.npz

Saturation check for Issue 3. Every stage contains the whole pilot web set (C4 + FineWeb sample) and a
growing prefix of the extended parts, taken in task order (the file list was shuffled before the run):
  ext_f1_16 ... ext_f1_2, ext_all   pilot web + 1/16 ... 1/2, all of the extended parts
  ext_only                          the extended parts alone
  ext_h0, ext_h1                    pilot web and extended parts, even and odd positions (split-half)
  ext_everything                    ext_all + Wikipedia + all harvested books

Usage: merge_ext.py --pilot <cooc_pilot dir> --ext <counts_ext dir> --tasks tasks_ext.txt --out <dir>
"""
import argparse
import glob
import os

import numpy as np

KEEP = ("Cd_raw", "Cd_art", "Cd_artnl", "u_raw", "u_art", "u_artnl", "n_tok", "n_tok_art", "n_tok_artnl",
        "n_art", "n_mention_raw", "n_mention_art")


class Acc:
    def __init__(self):
        self.t, self.forms, self.n = {}, None, 0

    def add(self, p):
        d = np.load(p)
        f = [str(x) for x in d["forms"]]
        if self.forms is None:
            self.forms = f
        elif f != self.forms:
            raise SystemExit("surface forms differ in " + p)
        for k in KEEP:
            v = d[k].astype(np.int64)
            self.t[k] = self.t[k] + v if k in self.t else v
        self.n += 1

    def merge(self, o):
        for k in KEEP:
            self.t[k] = self.t[k] + o.t[k] if k in self.t else o.t[k].copy()
        self.forms = self.forms or o.forms
        self.n += o.n

    def copy(self):
        a = Acc()
        a.merge(self)
        return a

    def save(self, out):
        np.savez_compressed(out, forms=np.array(self.forms), n_parts=np.int64(self.n), **self.t)
        print("%-20s %5d parts  %8.1fB tokens  pure %10d  gated %9d" % (
            os.path.basename(out), self.n, self.t["n_tok"] / 1e9, self.t["n_mention_raw"], self.t["n_mention_art"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    order = [l.split()[2] for l in open(a.tasks) if l.strip()]
    ext = [os.path.join(a.ext, "part_%s.npz" % t) for t in order]
    miss = [p for p in ext if not os.path.exists(p)]
    if miss:
        raise SystemExit("%d extended parts are missing, first %s" % (len(miss), miss[0]))
    web = [l.split()[2] for l in open(os.path.join(a.pilot, "tasks_web.txt")) if not l.split()[2].startswith("wiki_")]
    web = [os.path.join(a.pilot, "counts_web", "part_%s.npz" % t) for t in web]

    base = Acc()
    for p in web:
        base.add(p)
    h = [Acc(), Acc()]
    for i, p in enumerate(web):                      # halves of the pilot web, by position
        h[i % 2].add(p)

    stages = {len(ext) * n // 16: "ext_f1_%d" % (16 // n) for n in (1, 2, 4, 8)}
    only = Acc()
    for i, p in enumerate(ext, 1):
        only.add(p)
        h[(i - 1) % 2].add(p)
        if i in stages:
            s = base.copy()
            s.merge(only)
            s.save(os.path.join(a.out, stages[i] + ".npz"))
    only.save(os.path.join(a.out, "ext_only.npz"))
    full = base.copy()
    full.merge(only)
    full.save(os.path.join(a.out, "ext_all.npz"))
    h[0].save(os.path.join(a.out, "ext_h0.npz"))
    h[1].save(os.path.join(a.out, "ext_h1.npz"))
    for p in sorted(glob.glob(os.path.join(a.pilot, "counts_wiki", "part_wiki_*.npz"))) + \
            sorted(glob.glob(os.path.join(a.pilot, "counts_books", "part_*.npz"))):
        full.add(p)
    full.save(os.path.join(a.out, "ext_everything.npz"))


if __name__ == "__main__":
    main()
