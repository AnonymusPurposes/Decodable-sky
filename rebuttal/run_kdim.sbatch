#!/bin/bash
#SBATCH -J reb-kdim
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 112
#SBATCH --mem=300G
#SBATCH -t 90
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/kdim_%j.log
# Linmap without the top-8 limit (k = 8, 32, 128, full), one model; GPU 4 only (named explicitly), then CPU scoring.
M=${1:-qwen32b}; G=4
m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$m" -gt 2000 ] && { echo "refusing: GPU $G already holds $m MiB"; exit 1; }
W=/data/llm_games/decodable_sky_rebuttal; O=$W/results/kdim; P=/data/llm_games/toolchain/bin/python3.12
export PYTHONPATH=/data/llm_games/pylibs
cd $W/repo/rebuttal
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G /data/llm_games/vllm_venv/bin/python metric_kdim_cent.py --model $M \
  --raw $W/raw/${M}_raw.npy --meta $W/pca_repro/${M}_pca128.npz --out $O || exit 1
for k in k8 kfull k32 k128; do
  OMP_NUM_THREADS=1 $P metric_fullpca_allscore.py --cent $O --key $k --cand $W/results/fullcent/cand188.npz \
    --out $O/score_$k --procs 100 --chunk 1 $M
done
echo ALLDONE
