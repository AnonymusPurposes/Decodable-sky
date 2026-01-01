#!/bin/bash
#SBATCH -J reb-perm
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/perm_%j.log
# Empirical p-values with PCA fitted on training objects (10^5 permutations). GPUs 4 and 5 only.
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4
export PYTHONPATH=/data/llm_games/pylibs:/data/llm_games/decodable_sky_rebuttal/repo/rebuttal
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1; PY="$VBIN/python"; mkdir -p "$W/results/split_perm"
one () { CUDA_VISIBLE_DEVICES=$1 $PY rebuttal/split_perm.py --model $2 --best-layer $3 --raw-dir "$W/raw" --split-dir "$W/results/split_pca" --out "$W/results/split_perm" > "$W/logs/an/perm_$2.log" 2>&1 && echo "OK $2 $(date)" || echo "FAIL $2 $(date)"; }
( one 4 mistrallarge123b 72; one 4 qwen235b 88; one 4 mixtral8x22b 52; one 4 gptoss120b_harmony 29 ) &
( one 5 llama33_70b 72; one 5 qwen32b 63; one 5 glm45air 46; one 5 gptoss120b 29 ) &
wait; echo ALLDONE
