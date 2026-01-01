#!/usr/bin/env python
"""Sum the per-shard outputs of cooc_count.py into one form-level file.

Usage: merge_cooc.py <counts_dir> <out.npz> <out_examples.jsonl>

The npz keeps everything at surface-form level (Cd_* by distance, D_*, u_*, statistics);
export_cooc_graph.py turns it into object-level graphs. Examples are re-sampled across
shards in proportion to each shard's number of occurrences.
"""
import sys, os, glob, json, random
import numpy as np


def main():
    cdir, out_npz, out_ex = sys.argv[1:4]
    parts = sorted(glob.glob(os.path.join(cdir, "part_*.npz")))
    tot, forms, scalars = {}, None, {}
    for p in parts:
        d = np.load(p)
        if forms is None:
            forms = d["forms"]
        assert (d["forms"] == forms).all(), p
        for k in d.files:
            if k in ("forms", "src"):
                continue
            if d[k].ndim == 0:
                if k in ("LMAX", "ART_DENS"):
                    scalars[k] = d[k]
                else:
                    tot[k] = tot.get(k, 0) + int(d[k])
            else:
                tot[k] = tot.get(k, 0) + d[k]
    np.savez_compressed(out_npz, forms=forms, n_parts=np.int64(len(parts)),
                        **{k: (np.int64(v) if np.isscalar(v) else v) for k, v in tot.items()}, **scalars)
    print("%d parts -> %s" % (len(parts), out_npz))
    print({k: v for k, v in tot.items() if np.isscalar(v)})

    pool = {}
    for p in sorted(glob.glob(os.path.join(cdir, "examples_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            e = pool.setdefault((r["form"], r["level"]), dict(n=0, items=[]))
            e["n"] += r["n"]
            w = r["n"] / len(r["contexts"])
            e["items"] += [(w, c) for c in r["contexts"]]
    rnd = random.Random(0)
    with open(out_ex, "w", encoding="utf-8") as fh:
        for (form, level), e in sorted(pool.items()):
            items = list(e["items"]); pick = []
            while items and len(pick) < 8:            # weighted sampling without replacement
                i = rnd.choices(range(len(items)), weights=[w for w, _ in items])[0]
                pick.append(items.pop(i)[1])
            fh.write(json.dumps(dict(form=form, level=level, n=e["n"], contexts=pick)) + "\n")
    print("examples ->", out_ex)


if __name__ == "__main__":
    main()
