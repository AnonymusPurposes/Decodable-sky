#!/bin/bash
#SBATCH -J reb-tmpl
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 12
#SBATCH --mem=500G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/tmpl_%j.log
# Pair-held-out steering (as run_pair.sbatch) on the three extra templates of
# rebuttal/steering_templates_extra.csv. GPUs 4 and 5 only.
# Usage: sbatch run_templates.sbatch "<gpus>" <model> [<model> ...]   Degree 2 first, then degree 1.
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
W=/data/llm_games/decodable_sky_rebuttal; S=/data/llm_games/harmonic_multimodel_fixed_20260918; OUT="$W/steering_templates"; mkdir -p "$OUT"
EXTRA="$W/repo/rebuttal/steering_templates_extra.csv"
cd "$W/repo/steering/rotation/harmonic" || exit 1; PY="$VBIN/python"
# Other users' processes share GPUs 4-5: wait until every assigned GPU is (nearly) empty.
wait_free () {
  for i in $(seq 1 720); do
    busy=0
    for g in ${GPUS//,/ }; do
      u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$u" -gt 3000 ] && busy=1
    done
    [ $busy = 0 ] && return 0
    [ $((i % 10)) = 1 ] && echo "waiting for GPUs $GPUS $(date)"
    sleep 60
  done
  echo "GPUs $GPUS still busy after 12 h"; exit 1
}
for M in "$@"; do
  export HF_HUB_OFFLINE=$($PY -c "import models_multi as m; print(0 if m.MODELS['$M'].get('online_kernels') else 1)")
  for DEG in 2 1; do
    T=""; [ $DEG = 1 ] && T="_l1"
    B=$(ls -d "$S"/prepared_${M}_heldout_typed${T}_6[0-9][0-9][0-9] | head -1)
    P0="$OUT/battery_${M}${T}"; P="$OUT/prepared_${M}${T}"; R="$OUT/run_${M}${T}"; A="$OUT/results${T}"
    [ -f "$A/trials_${M}.csv" ] && { echo "SKIP $M deg$DEG"; continue; }
    rm -rf "$P0" "$P" "$R"
    wait_free
    echo "START $M deg$DEG $(date)"
    $PY "$W/repo/rebuttal/prepare_template_battery.py" --bundle "$B" --templates "$EXTRA" --out "$P0" && \
    $PY prepare_pair_holdout.py --bundle "$P0" --out "$P" && \
    $PY run_pair_holdout.py --prepared "$P" --out "$R" --token-policy sustained && \
    $PY analyze_multi.py --prepared "$P" --run "$R" --out "$A" && echo "OK $M deg$DEG $(date)" || echo "FAIL $M deg$DEG $(date)"
  done
done
echo LANEDONE
