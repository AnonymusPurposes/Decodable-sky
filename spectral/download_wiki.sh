#!/bin/bash
# One-time download of HF wikimedia/wikipedia 20231101.en (41 parquet shards), 8 in parallel.
umask 000
cd /data/work/decodable_sky/spectral/corpus/wikipedia_20231101_en || exit 1
seq 0 40 | xargs -P 8 -I{} bash -c 'f=$(printf "train-%05d-of-00041.parquet" {}); [ -s "$f" ] && exit 0; curl -sSL --retry 5 --max-time 3600 -o "$f.part" "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/$f" && mv "$f.part" "$f" && echo "ok $f"'
echo DONE; ls | wc -l; du -sh .
