#!/bin/bash
#SBATCH -J cooc-pilot
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 64
#SBATCH --mem=240G
#SBATCH -o /data/llm_games/decodable_sky/cooc_pilot/logs/pilot.log
# Web + Wikipedia pilot: one cooc_stream.py process per file, 60 at a time.
cd /data/llm_games/decodable_sky/cooc_pilot
export PYTHONPATH=/data/llm_games/pylibs COOC_TMP=/data/llm_games/decodable_sky/cooc_pilot/tmp OMP_NUM_THREADS=1
P=/data/llm_games/astro_venv/bin/python
mkdir -p counts_wiki counts_web logs/w tmp
date
xargs -P 60 -L 1 bash -c 'case "$2" in wiki_*) o=counts_wiki;; *) o=counts_web;; esac; '"$P"' -u cooc_stream.py surface_forms_v2.csv $o "$0" "$1" "$2" > logs/w/$2.log 2>&1 || echo "FAIL $2"' < tasks_web.txt
date; echo ALLDONE
