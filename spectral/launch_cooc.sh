#!/bin/bash
# Count object co-occurrences in all 41 Wikipedia shards, one process per shard.
# Usage: launch_cooc.sh [out_dir]   (default counts_wiki)
umask 000
cd /data/work/decodable_sky/spectral || exit 1
export PYTHONPATH=/data/work/pylibs
P=/data/work/astro_venv/bin/python
OUT=${1:-counts_wiki}
mkdir -p "$OUT" logs
for f in corpus/wikipedia_20231101_en/train-*-of-00041.parquet; do
  t=$(basename "$f" .parquet)
  [ -s "$OUT/part_$t.npz" ] && continue
  nohup $P cooc_count.py "$f" graphs/surface_forms.csv "$OUT" > "logs/cooc_$t.log" 2>&1 &
done
wait
echo ALLDONE
