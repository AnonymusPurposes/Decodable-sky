#!/bin/bash
#SBATCH -J reb-fullscore
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 112
#SBATCH --mem=200G
#SBATCH -t 120
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/fullscore_%j.log
# CPU only: linmap scores of all 79 coordinate choices on the full-dimensional centroids.
W=/data/llm_games/decodable_sky_rebuttal
cd $W/repo/rebuttal && OMP_NUM_THREADS=1 PYTHONPATH=/data/llm_games/pylibs /data/llm_games/toolchain/bin/python3.12 metric_fullpca_allscore.py \
  --cent $W/results/fullcent --cand $W/results/fullcent/cand188.npz --out $W/results/fullscore --procs 110
echo ALLDONE
