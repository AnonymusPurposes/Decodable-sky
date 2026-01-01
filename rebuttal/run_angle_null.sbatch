#!/bin/bash
#SBATCH -J reb-angnull
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/angnull_%j.log
# Permutation null of the median angular error on the real activations, every layer. GPUs 4 and 5 only.
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4
export PYTHONPATH=/data/llm_games/pylibs:/data/llm_games/decodable_sky_rebuttal/repo/rebuttal
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1; PY="$VBIN/python"; mkdir -p "$W/results/angle_null"
one () { CUDA_VISIBLE_DEVICES=$1 $PY rebuttal/angle_null.py --model $2 --raw-dir "$W/raw" --split-dir "$W/results/split_pca" --out "$W/results/angle_null" > "$W/logs/an/angnull_$2.log" 2>&1 && echo "OK $2 $(date)" || echo "FAIL $2 $(date)"; }
( one 4 gptoss120b; one 4 mistrallarge123b; one 4 qwen235b; one 4 gptoss120b_harmony ) &
( one 5 qwen32b; one 5 llama33_70b; one 5 glm45air; one 5 mixtral8x22b ) &
wait; echo ALLDONE
