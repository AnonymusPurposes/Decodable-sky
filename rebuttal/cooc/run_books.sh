#!/bin/bash
#SBATCH -J cooc-books
#SBATCH -w b300-nsu4-0-37x9xx
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -o /data/llm_games/decodable_sky/cooc_pilot/logs/books.log
# Count every harvested book (one part per book), then score them.
cd /data/llm_games/decodable_sky/cooc_pilot
export PYTHONPATH=/data/llm_games/pylibs OMP_NUM_THREADS=1
P=/data/llm_games/astro_venv/bin/python
mkdir -p counts_books logs/b
ls books/raw/*.txt > books/all_files.txt
split -n r/16 -d books/all_files.txt books/chunk_
ls books/chunk_* | xargs -P 16 -I{} bash -c "$P -u cooc_stream.py surface_forms_v2.csv counts_books books {} > logs/b/\$(basename {}).log 2>&1 || echo FAIL {}"
rm -f books/chunk_*
$P -u book_report.py books counts_books /data/llm_games/decodable_sky/data/astro_objects.csv surface_forms_v2.csv
echo BOOKSDONE
