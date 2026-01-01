#!/bin/bash
#SBATCH -J reb-fullcent
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 32
#SBATCH --mem=400G
#SBATCH -t 120
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/fullcent_%j.log
# Full-dimensional top-8 PCA centroids (no common 128 basis), GPUs 4 and 5 only (named explicitly).
for g in 4 5; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt 2000 ] && { echo "refusing: GPU $g already holds $m MiB"; exit 1; }; done
W=/data/llm_games/decodable_sky_rebuttal; O=$W/results/fullcent
bash $W/repo/rebuttal/run_fullcent.sh 4 100 100 $O mistrallarge123b qwen235b gptoss120b > $W/logs/fullcent_A.log 2>&1 &
bash $W/repo/rebuttal/run_fullcent.sh 5 100 100 $O llama33_70b mixtral8x22b qwen32b glm45air > $W/logs/fullcent_B.log 2>&1 &
wait; grep -h DONE $W/logs/fullcent_[AB].log; echo ALLDONE
