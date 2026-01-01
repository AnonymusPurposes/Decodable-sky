#!/bin/bash
#SBATCH -J reb-splits
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 24
#SBATCH --mem=400G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/splits_%j.log
# In-split PCA (Issues 0, 1, 4) and held-out metric comparison (Issues 0, 2) for every model.
# GPUs 4 and 5 only. Each script checkpoints and resumes, so this can be resubmitted freely.
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4
export PYTHONPATH=/data/llm_games/pylibs:/data/llm_games/decodable_sky_rebuttal/repo/rebuttal
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1; PY="$VBIN/python"
mkdir -p "$W/results/split_pca" "$W/results/metric_heldout" "$W/logs/an"
one () { CUDA_VISIBLE_DEVICES=$1 $PY rebuttal/split_pca.py --model $2 --raw-dir "$W/raw" --out "$W/results/split_pca" > "$W/logs/an/split_$2.log" 2>&1 || echo "FAIL split $2"
  CUDA_VISIBLE_DEVICES=$1 $PY rebuttal/metric_heldout.py --model $2 --raw-dir "$W/raw" --r2-table spectral/r2_tables/$2_sky_r2_16.csv --out "$W/results/metric_heldout" > "$W/logs/an/metric_$2.log" 2>&1 || echo "FAIL metric $2"; echo "finished $2 $(date)"; }
( for m in mistrallarge123b qwen235b mixtral8x22b gptoss120b_harmony; do one 4 $m; done ) &
( for m in llama33_70b qwen32b glm45air gptoss120b; do one 5 $m; done ) &
wait; echo ALLDONE
