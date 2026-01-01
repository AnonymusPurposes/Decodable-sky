#!/bin/bash
# usage: run_fullcent.sh <gpu> <splits> <nboot> <outdir> <models...>
G=$1; NS=$2; NB=$3; O=$4; shift 4
W=/data/llm_games/decodable_sky_rebuttal
export PYTHONPATH=/data/llm_games/pylibs OMP_NUM_THREADS=8
for m in "$@"; do
  CUDA_VISIBLE_DEVICES=$G /data/llm_games/vllm_venv/bin/python $W/repo/rebuttal/metric_fullpca_cent.py --model $m \
    --raw $W/raw/${m}_raw.npy --meta $W/pca_repro/${m}_pca128.npz --out $O --splits $NS --nboot $NB
done
