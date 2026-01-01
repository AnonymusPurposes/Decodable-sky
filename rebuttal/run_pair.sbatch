#!/bin/bash
#SBATCH -J reb-pair
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 12
#SBATCH --mem=500G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/pair_%j.log
# Steering with a map fitted without each pair's A and B. GPUs 4 and 5 only.
# Usage: sbatch run_pair.sbatch "<gpus>" <model> [<model> ...]   Degree 2 first, then degree 1.
GPUS="$1"; shift
case ",$GPUS," in *,0,*|*,1,*|*,2,*|*,3,*|*,6,*|*,7,*) echo "refusing GPUs $GPUS"; exit 1;; esac
TC=/data/llm_games/toolchain; VBIN=/data/llm_games/vllm_venv/bin
export PATH="$VBIN:$TC/bin:$PATH" CUDA_HOME="$TC" CC="$TC/bin/cc" CXX="$TC/bin/c++" TRITON_CC="$TC/bin/cc"
export C_INCLUDE_PATH="$TC/include/python3.12:$TC/include:$TC/targets/x86_64-linux/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH" CPATH="$TC/targets/x86_64-linux/include"
export LIBRARY_PATH="$TC/lib:$TC/lib/stubs:/usr/lib/x86_64-linux-gnu" LD_LIBRARY_PATH="$TC/lib:$LD_LIBRARY_PATH"
export HF_HOME=/data/llm_games/hf_cache FLASHINFER_WORKSPACE_BASE=/data/llm_games/flashinfer_cache
export PYTHONPATH=/data/llm_games/pylibs TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPUS"
W=/data/llm_games/decodable_sky_rebuttal; S=/data/llm_games/harmonic_multimodel_fixed_20260918; OUT="$W/steering_pair"; mkdir -p "$OUT"
cd "$W/repo/steering/rotation/harmonic" || exit 1; PY="$VBIN/python"
for M in "$@"; do
  export HF_HUB_OFFLINE=$($PY -c "import models_multi as m; print(0 if m.MODELS['$M'].get('online_kernels') else 1)")
  for DEG in 2 1; do
    T=""; [ $DEG = 1 ] && T="_l1"
    B=$(ls -d "$S"/prepared_${M}_heldout_typed${T}_6[0-9][0-9][0-9] | head -1)
    P="$OUT/prepared_${M}${T}"; R="$OUT/run_${M}${T}"; A="$OUT/results${T}"
    [ -f "$A/trials_${M}.csv" ] && { echo "SKIP $M deg$DEG"; continue; }
    rm -rf "$P" "$R"
    echo "START $M deg$DEG $(date)"
    $PY prepare_pair_holdout.py --bundle "$B" --out "$P" && \
    $PY run_pair_holdout.py --prepared "$P" --out "$R" --token-policy sustained && \
    $PY analyze_multi.py --prepared "$P" --run "$R" --out "$A" && echo "OK $M deg$DEG $(date)" || echo "FAIL $M deg$DEG $(date)"
  done
done
echo LANEDONE
