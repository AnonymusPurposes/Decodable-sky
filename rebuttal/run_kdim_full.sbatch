#!/bin/bash
#SBATCH -J reb-kfull
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 112
#SBATCH --mem=400G
#SBATCH -t 120
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/kfull_%j.log
# Linmap in the full hidden space (no PCA) for the remaining models; GPUs 4 and 5 only (named explicitly).
# 1) fast scorer checked against score() (sample + whole qwen32b rescoring vs job 14135), 2) centroids, 3) scores.
for g in 4 5; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt 2000 ] && { echo "refusing: GPU $g already holds $m MiB"; exit 1; }; done
W=/data/llm_games/decodable_sky_rebuttal; O=$W/results/kdim; P=/data/llm_games/toolchain/bin/python3.12; V=/data/llm_games/vllm_venv/bin/python
export PYTHONPATH=/data/llm_games/pylibs
cd $W/repo/rebuttal
lane() { G=$1; shift; for M in "$@"; do OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G $V metric_kdim_cent.py --model $M --full-only \
  --raw $W/raw/${M}_raw.npy --meta $W/pca_repro/${M}_pca128.npz --out $O; done; }
lane 4 mistrallarge123b glm45air > $W/logs/kfull_A.log 2>&1 &
lane 5 llama33_70b qwen235b mixtral8x22b > $W/logs/kfull_B.log 2>&1 &
SC="--cent $O --key kfull --cand $W/results/fullcent/cand188.npz"
OMP_NUM_THREADS=1 $P metric_fullpca_allscore.py $SC --verify qwen32b
OMP_NUM_THREADS=1 $P metric_fullpca_allscore.py $SC --fast --out $O/score_kfull_fast --procs 100 --chunk 1 qwen32b
OMP_NUM_THREADS=1 $P -c "
import numpy as np
a, b = np.load('$O/score_kfull/qwen32b.npz'), np.load('$O/score_kfull_fast/qwen32b.npz')
for k in ('sphere', 'alone', 'joined'):
    print('rescore qwen32b', k, 'max |fast - slow| per score %.2e, per median %.2e' % (np.nanmax(np.abs(a[k] - b[k])), np.nanmax(np.abs(np.median(a[k], 0) - np.median(b[k], 0)))))"
wait; grep -h -E "DONE|Error|Traceback" $W/logs/kfull_[AB].log
OMP_NUM_THREADS=1 $P metric_fullpca_allscore.py $SC --fast --out $O/score_kfull --procs 110 --chunk 1 mistrallarge123b llama33_70b qwen235b mixtral8x22b glm45air
echo ALLDONE
