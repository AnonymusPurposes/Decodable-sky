#!/bin/bash
#SBATCH -J reb-k32
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/k32_%j.log
# gpt-oss at top-16/top-32 (Figure 12 plots gpt-oss with these): 80/20 splits and angle null. GPUs 4 and 5 only.
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4
export PYTHONPATH=/data/llm_games/pylibs:/data/llm_games/decodable_sky_rebuttal/repo/rebuttal
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1; PY="$VBIN/python"
mkdir -p "$W/results/split_pca_k32" "$W/results/angle_null_k16" "$W/results/angle_null_k32"
one () {
  G=$1; M=$2
  CUDA_VISIBLE_DEVICES=$G $PY rebuttal/split_pca.py --model $M --raw-dir "$W/raw" --out "$W/results/split_pca_k32" --ks 8,16,32 > "$W/logs/an/split32_$M.log" 2>&1 && echo "OK split $M $(date)" || echo "FAIL split $M"
  for K in 16 32; do
    CUDA_VISIBLE_DEVICES=$G $PY rebuttal/angle_null.py --model $M --k $K --raw-dir "$W/raw" --split-dir "$W/results/split_pca" --out "$W/results/angle_null_k$K" > "$W/logs/an/angnull${K}_$M.log" 2>&1 && echo "OK null$K $M $(date)" || echo "FAIL null$K $M"
  done
}
one 4 gptoss120b & one 5 gptoss120b_harmony & wait; echo ALLDONE
