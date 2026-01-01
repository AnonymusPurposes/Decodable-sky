#!/bin/bash
# Usage: sbatch fitvariants.sbatch <model_key> <variant> [degree]
#
# Compares how the harmonic map is FITTED, holding everything else fixed --
# same layers, same pairs, same sustained policy, same evaluation templates.
#
#   base         every template, constellations only (the battery's default)
#   tmplheldout  every template EXCEPT the two evaluated on, constellations only
#   typed        every template, ALL objects with one intercept per type
#                (constellation / star / other); the chart keeps the
#                constellation intercept
#   heldout_typed  both of the above at once
#
# Results go to results_fit_<variant>/, with _l1 appended for degree 1.
#
# Mixtral-8x22B is 141B parameters in BF16 (~282 GB) and does not fit one 275 GB
# B300: submit it as `sbatch --gpus=2 fitvariants.sbatch mixtral8x22b ...`.
#SBATCH --job-name=harm-fitvar
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.log
set -euo pipefail
umask 000
MODEL="${1:?supply a model key}"
VARIANT="${2:?supply base|tmplheldout|typed}"
DEG="${3:-2}"
case "$VARIANT" in
  base)          FLAGS="" ;;
  tmplheldout)   FLAGS="--fit-templates exclude-eval" ;;
  typed)         FLAGS="--fit-scope typed" ;;
  heldout_typed) FLAGS="--fit-templates exclude-eval --fit-scope typed" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
TAG=""
[ "$DEG" = "1" ] && TAG="_l1"
export PATH=/data/work/vllm_venv/bin:/data/work/toolchain/bin:$PATH
export PYTHONPATH=/data/work/pylibs
export HF_HOME=/data/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

export HF_HUB_OFFLINE=$(python -c "import models_multi as m; print(0 if m.MODELS['$MODEL'].get('online_kernels') else 1)")
echo "$MODEL: variant=$VARIANT degree=$DEG flags='$FLAGS' HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

prep="prepared_${MODEL}_${VARIANT}${TAG}_$SLURM_JOB_ID"
run="run_${MODEL}_${VARIANT}${TAG}_$SLURM_JOB_ID"

echo "=============== $MODEL / $VARIANT : fit ==============="
python prepare_multi.py --model "$MODEL" --degree "$DEG" --out "$prep" $FLAGS

echo "=============== $MODEL / $VARIANT : sustained steering ==============="
python run.py --prepared "$prep" --out "$run" --token-policy sustained

echo "=============== $MODEL / $VARIANT : scoring ==============="
python analyze_multi.py --prepared "$prep" --run "$run" --out "results_fit_${VARIANT}${TAG}"

echo "=============== $MODEL / $VARIANT DONE ==============="
