#!/bin/bash
#SBATCH -J reb-coocx
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 64
#SBATCH --mem=240G
#SBATCH -o /data/llm_games/decodable_sky_rebuttal/logs/coocx_%j.log
# Extend the co-occurrence count with random whole-FineWeb files (CPU only). One part per file;
# a finished part is never recounted, so the job can be cancelled and resubmitted freely.
W=/data/llm_games/decodable_sky_rebuttal; cd "$W/repo/rebuttal/cooc" || exit 1
export PYTHONPATH=/data/llm_games/pylibs COOC_TMP="$W/cooc/tmp" OMP_NUM_THREADS=1
P=/data/llm_games/astro_venv/bin/python; mkdir -p "$W/cooc/counts_ext" "$W/cooc/logs" "$COOC_TMP"
date
xargs -P 60 -L 1 bash -c "$P -u cooc_stream.py surface_forms_v2.csv $W/cooc/counts_ext \"\$0\" \"\$1\" \"\$2\" > $W/cooc/logs/\$2.log 2>&1 || echo FAIL \$2" < tasks_ext.txt
date; echo ALLDONE
