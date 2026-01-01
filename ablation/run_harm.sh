#!/bin/bash
# One model of the l<=2 harmonic sky-subspace ablation.
#
# Usage: ./run_harm.sh <gpus> <model> <outdir> [extra ablate_harm.py args]
#   ./run_harm.sh 0   qwen32b      results_harm_frac80
#   ./run_harm.sh 4,5 mixtral8x22b results_harm_frac80 --device-map auto
#
# Inputs resolve relative to this directory; override any of them in the environment:
#   PCA_DIR   full 128-PC archives, <model>_pca128.npz          (default ../pca)
#   R2_DIR    sky_r2 tables, <model>_sky_r2_16.csv              (default ../steering/rotation/harmonic/r2_tables)
#   PROMPTS   prompt templates                                  (default ../data/astro_prompts_location.csv)
#   PYTHON    interpreter                                       (default python)
#
# PCA_DIR must hold a FULL extract_pca.py run, not the cropped top-16 archive bundled
# in PCA128/: the projection needs components, feat_mean, feat_std and pca_mean.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PCA_DIR="${PCA_DIR:-$HERE/../pca}"
R2_DIR="${R2_DIR:-$HERE/../steering/rotation/harmonic/r2_tables}"
PROMPTS="${PROMPTS:-$HERE/../data/astro_prompts_location.csv}"
PYTHON="${PYTHON:-python}"

if [ $# -lt 3 ]; then
  echo "usage: $0 <gpus> <model> <outdir> [extra ablate_harm.py args]" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$1"
export TOKENIZERS_PARALLELISM=false
MODEL="$2"
OUT="$3"
shift 3

exec "$PYTHON" "$HERE/ablate_harm.py" \
  --model "$MODEL" \
  --npz "$PCA_DIR/${MODEL}_pca128.npz" \
  --r2-csv "$R2_DIR/${MODEL}_sky_r2_16.csv" \
  --prompts "$PROMPTS" \
  --out "$OUT/$MODEL" "$@"
