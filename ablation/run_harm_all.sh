#!/bin/bash
# All-model l<=2 harmonic ablation at frac 0.8, producing results_harm_frac80/.
#
# Five lanes over six GPUs, models sequential within a lane. Resumable: a model whose
# diagnostics file already exists is skipped, so a failed lane can simply be rerun.
#
# Usage: ./run_harm_all.sh [outdir]
#
# Lane assignment assumes eight roughly 200 GB GPUs. On a smaller machine, run the
# lanes one at a time, or call run_harm.sh per model with your own device list.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/results_harm_frac80}"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

run () {   # $1=gpus $2=model [extra ablate_harm.py args]
  local g=$1 m=$2
  shift 2
  if [ -f "$OUT/$m/diagnostics_$m.json" ]; then echo "### SKIP $m"; return; fi
  echo "### [$m] start $(date) GPUS=$g"
  if "$HERE/run_harm.sh" "$g" "$m" "$OUT" "$@" > "$LOG/$m.log" 2>&1; then
    echo "### [$m] OK $(date)"
  else
    echo "### FAIL $m $(date)"
  fi
}

echo "### BATCH START $(date)"
( run 0 qwen235b ) > "$LOG/laneA.log" 2>&1 &
( run 1 glm45air ) > "$LOG/laneB.log" 2>&1 &
( run 2 llama33_70b; run 2 qwen32b ) > "$LOG/laneC.log" 2>&1 &
( run 3 gptoss120b; run 3 gptoss120b_harmony --min-score-corr 0.98 ) > "$LOG/laneD.log" 2>&1 &
( run 4,5 mixtral8x22b --device-map auto; run 4,5 mistrallarge123b --device-map auto ) > "$LOG/laneE.log" 2>&1 &
wait
echo "### ALL LANES DONE $(date)"
