#!/bin/bash
#SBATCH -J reb-fullcentroid
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 180
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/full_centroids_%j.log
# Full standardized-space object centroids for the neural RDM of Figures 10/20 (CPU only, no GPU).
W=/data/llm_games/decodable_sky_rebuttal
export PYTHONPATH=/data/llm_games/pylibs OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
/data/llm_games/vllm_venv/bin/python $W/repo/rebuttal/full_centroids.py --raw $W/raw --meta $W/pca_repro \
  --out $W/results/full_centroids qwen32b gptoss120b gptoss120b_harmony glm45air mixtral8x22b qwen235b llama33_70b mistrallarge123b
echo ALLDONE
