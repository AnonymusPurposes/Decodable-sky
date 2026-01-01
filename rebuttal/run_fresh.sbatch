#!/bin/bash
#SBATCH -J reb-fresh
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 32
#SBATCH --mem=1200G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/fresh_%j.log
# New prompts of the same pattern: extract raw activations, then the 80/20 test on them.
# GPUs 4 and 5 ONLY, named explicitly. A model whose raw file exists is not extracted again.
for g in 4 5; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt 2000 ] && { echo "refusing: GPU $g already holds $m MiB"; exit 1; }; done
umask 000
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" CUDA_HOME="$TC" CC="$TC/bin/cc" CXX="$TC/bin/c++" TRITON_CC="$TC/bin/cc"
export C_INCLUDE_PATH="$TC/include/python3.12:$TC/include:$TC/targets/x86_64-linux/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH" CPATH="$TC/targets/x86_64-linux/include"
export LIBRARY_PATH="$TC/lib:$TC/lib/stubs:/usr/lib/x86_64-linux-gnu" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH"
export HF_HOME=/data/llm_games/hf_cache FLASHINFER_WORKSPACE_BASE=/data/llm_games/flashinfer_cache
export PYTHONPATH=/data/llm_games/pylibs TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1
PY="$VBIN/python"; P=rebuttal/prompts_fresh.csv; O="$W/pca_fresh"; R="$W/raw_fresh"; T="$W/results/fresh"; mkdir -p "$O" "$R" "$T" "$W/logs/fr"
run () { # gpus model layer
  if [ ! -f "$R/$2_raw.npy" ]; then
    echo "START $2 $(date)"
    CUDA_VISIBLE_DEVICES=$1 $PY extract_pca.py --model "$2" --prompts $P --out "$O" --raw-out "$R" --batch 32 || { echo "FAIL extract $2 $(date)"; return; }
  fi
  G=${1%%,*}
  CUDA_VISIBLE_DEVICES=$G $PY rebuttal/fresh_test.py --model "$2" --raw-dir "$W/raw" --fresh-dir "$R" --split-dir "$W/results/split_pca" --layer "$3" --out "$T" && echo "OK $2 $(date)" || echo "FAIL test $2 $(date)"
}
# the two largest need both GPUs, then the rest in two single-GPU lanes
run 4,5 mistrallarge123b 72 > "$W/logs/fr/A.log" 2>&1
run 4,5 mixtral8x22b 52 >> "$W/logs/fr/A.log" 2>&1
( run 4 qwen235b 88; run 4 qwen32b 63 ) > "$W/logs/fr/B.log" 2>&1 &
( run 5 llama33_70b 72; run 5 glm45air 46; run 5 gptoss120b 29 ) > "$W/logs/fr/C.log" 2>&1 &
wait; grep -h -E "^(OK|FAIL)" "$W"/logs/fr/[ABC].log; echo ALLDONE
