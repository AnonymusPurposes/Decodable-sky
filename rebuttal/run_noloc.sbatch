#!/bin/bash
#SBATCH -J reb-noloc
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 32
#SBATCH --mem=1200G
#SBATCH -t 300
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/noloc_%j.log
# Generic (non-location) prompts, data/astro_prompts_no_location.csv (= the paper's table), scored exactly as the
# location prompts in Figure 2: raw activations kept, then split_pca.py (80/20 object splits, standardization and
# PCA fitted on the training objects), k = 8 and 32, 20 splits (seed 0: the first 20 test sets of the location run).
# No gpt-oss. GPUs 4 and 5 ONLY, named explicitly. Resumable: existing raw files are not extracted again.
for g in 4 5; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt 2000 ] && { echo "refusing: GPU $g already holds $m MiB"; exit 1; }; done
umask 000
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" CUDA_HOME="$TC" CC="$TC/bin/cc" CXX="$TC/bin/c++" TRITON_CC="$TC/bin/cc"
export C_INCLUDE_PATH="$TC/include/python3.12:$TC/include:$TC/targets/x86_64-linux/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH" CPATH="$TC/targets/x86_64-linux/include"
export LIBRARY_PATH="$TC/lib:$TC/lib/stubs:/usr/lib/x86_64-linux-gnu" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH"
export HF_HOME=/data/llm_games/hf_cache FLASHINFER_WORKSPACE_BASE=/data/llm_games/flashinfer_cache
export PYTHONPATH=/data/llm_games/pylibs:/data/llm_games/decodable_sky_rebuttal/repo/rebuttal TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo" || exit 1
PY="$VBIN/python"; P=data/astro_prompts_no_location.csv; O="$W/pca_noloc"; R="$W/raw_noloc"; S="$W/results/split_pca_noloc"
mkdir -p "$O" "$R" "$S" "$W/logs/nl"
ext () { [ -f "$R/$2_raw.npy" ] && { echo "SKIP $2"; return; }; echo "START $2 $(date)"
  CUDA_VISIBLE_DEVICES=$1 $PY extract_pca.py --model "$2" --prompts $P --out "$O" --raw-out "$R" --batch 32 && echo "OK extract $2 $(date)" || echo "FAIL extract $2 $(date)"; }
spl () { CUDA_VISIBLE_DEVICES=$1 $PY rebuttal/split_pca.py --model $2 --raw-dir "$R" --out "$S" --ks 8,32 --n-splits 20 \
  > "$W/logs/nl/split_$2.log" 2>&1 && echo "OK split $2 $(date)" || echo "FAIL split $2 $(date)"; }
# the two largest are extracted on both GPUs, then two single-GPU lanes interleave extraction and splits
ext 4,5 mistrallarge123b > "$W/logs/nl/A.log" 2>&1
ext 4,5 mixtral8x22b >> "$W/logs/nl/A.log" 2>&1
( ext 4 qwen235b; spl 4 mistrallarge123b; ext 4 qwen32b; spl 4 qwen235b; spl 4 qwen32b ) > "$W/logs/nl/B.log" 2>&1 &
( ext 5 llama33_70b; spl 5 mixtral8x22b; ext 5 glm45air; spl 5 llama33_70b; spl 5 glm45air ) > "$W/logs/nl/C.log" 2>&1 &
wait; grep -h -E "^(OK|FAIL|SKIP)" "$W"/logs/nl/[ABC].log; echo ALLDONE
