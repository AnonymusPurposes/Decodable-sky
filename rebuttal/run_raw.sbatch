#!/bin/bash
#SBATCH -J reb-raw
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 32
#SBATCH --mem=1200G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/raw_%j.log
# Re-extract the residual stream with the paper's 25 prompts x 188 objects and keep the raw
# activations, so standardization and PCA can be refitted inside training splits.
# GPUs 4 and 5 ONLY: Slurm's --gpus takes the lowest free ids, so this job requests CPUs and
# memory and names the two GPUs itself. Resumable: a model whose raw file exists is skipped.
for g in 4 5; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt 2000 ] && { echo "refusing: GPU $g already holds $m MiB"; exit 1; }; done
umask 000
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" CUDA_HOME="$TC" CC="$TC/bin/cc" CXX="$TC/bin/c++" TRITON_CC="$TC/bin/cc"
export C_INCLUDE_PATH="$TC/include/python3.12:$TC/include:$TC/targets/x86_64-linux/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH" CPATH="$TC/targets/x86_64-linux/include"
export LIBRARY_PATH="$TC/lib:$TC/lib/stubs:/usr/lib/x86_64-linux-gnu" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH"
export HF_HOME=/data/llm_games/hf_cache FLASHINFER_WORKSPACE_BASE=/data/llm_games/flashinfer_cache
export PYTHONPATH=/data/llm_games/pylibs TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1
PY="$VBIN/python"; P=data/astro_prompts_location.csv; O="$W/pca_repro"; R="$W/raw"; mkdir -p "$O" "$R" "$W/logs"
std () { [ -f "$R/$2_raw.npy" ] && { echo "SKIP $2"; return; }; echo "START $2 $(date)"
  CUDA_VISIBLE_DEVICES=$1 $PY extract_pca.py --model "$2" --prompts $P --out "$O" --raw-out "$R" --batch 32 && echo "OK $2 $(date)" || echo "FAIL $2 $(date)"; }
har () { [ -f "$R/gptoss120b_harmony_raw.npy" ] && { echo "SKIP harmony"; return; }; echo "START harmony $(date)"
  CUDA_VISIBLE_DEVICES=$1 $PY extract_pca_gptoss_harmony.py --prompts $P --out "$O" --raw-out "$R" --batch 32 && echo "OK harmony $(date)" || echo "FAIL harmony $(date)"; }
# the two largest need both GPUs, then the rest in two single-GPU lanes
std 4,5 mistrallarge123b > "$W/logs/raw_A.log" 2>&1
std 4,5 mixtral8x22b >> "$W/logs/raw_A.log" 2>&1
( std 4 qwen235b; std 4 qwen32b ) > "$W/logs/raw_B.log" 2>&1 &
( std 5 llama33_70b; std 5 glm45air; std 5 gptoss120b; har 5 ) > "$W/logs/raw_C.log" 2>&1 &
wait; grep -h -E "^(OK|FAIL|SKIP)" "$W"/logs/raw_[ABC].log; echo ALLDONE
