# Revision analyses

Code, data and result tables for the analyses added in response to the reviews. Each script's docstring
describes its inputs, method and output columns. The `run_*.sbatch` / `run_*.sh` files are the Slurm
drivers we used; their paths are specific to our cluster.

## Data

The raw residual-stream activations (`<model>_raw.npy`: 4700 prompts × layers × hidden size, float16, about
30 GB for all models) are not included; `run_raw.sbatch` re-extracts them with the paper's prompts
(`extract_pca.py --raw-out`). Everything downstream of them that a table or figure needs is in `data/`:

| Folder | Content | Used by |
|---|---|---|
| `data/fullstd/<model>_pca128.npz` | object centroids in the full standardized residual stream, every hidden dimension, as an isometric 188-d chart per layer (`full_centroids.py`); same npz layout as `../PCA128` | neural RDM of Figures 10, 21–23 and 24, via `../spectral/paths.py` |
| `data/cooc/sets*/` | merged co-occurrence counts of every corpus stage and split half (`cooc/merge_*.py`; corpora in `cooc/SOURCES.md`) | Q_text: `cooc/qtext_*.py`, `figures/qtext_variant_rdms.py` |
| `data/steering/<run>/results{,_l1}_rich_v3/` | every steering trial with its completion and parsed category, l=1,2 and l=1 maps; `steering_pair` = templates 0 and 18, `steering_templates` = the three extra templates | `steering_paired.py`, `steering_templates.py`, `compare_pair.py` |
| `data/ablation/<model>/` | every ablation completion with its scored answer (`completions_<model>_scored.csv.gz`) and run diagnostics | `intervals.py` (reads the `.gz` directly) |

`../spectral/graphs/*.npz` (constellation graph, Wikipedia counts) are shipped for the same reason.

## Review points

| Review point | Paper | Script(s) | Result table(s) |
|---|---|---|---|
| PCA fitted on all objects before the split; layer selection; variability across splits | §2.1, App. D.1, Fig. 2 | `split_pca.py`, `split_summary.py`, `split_layer_csv.py` | `results/split_summary.csv`, `results/split_by_layer.csv` |
| Empirical p-value with PCA fitted in the split, with and without layer selection | App. D.1 | `split_perm.py` | `results/split_perm.csv` |
| Prompts selected for performance: a new prompt list fixed before extraction | App. D.1 | `fresh_test.py`, `prompts_fresh.csv` | `results/fresh_prompts.csv` |
| Empirical baseline for the angle (shuffled labels, same read-out) | Fig. 2 | `angle_null.py` | `results/angle_null_summary.csv`, `results/angle_null/` |
| Empirical chance level for ablation (answers shuffled across query objects) | Figs. 5, 16, App. E.1 | `intervals.py` | `results/baseline_angle.csv` |
| Ablation: intervals and removed variance per condition | App. E.1, Figs. 16–17 | `intervals.py` | `results/ablation_intervals.csv` (`var_frac_top16_removed`) |
| Why the strict (l=2)⊥ ablation matters in some models (response only) | — | `harmonic_alignment.py`, `l2_overlap.py`, `l2_mixing.py` | `results/harmonic_alignment_l2_in_l1*.csv`, `results/l2_overlap.csv`, `results/l2_mixing.csv` |
| Steering: A and B held out of the fit; paired uncertainty; success without naming B; usable denominators | §3.3, App. E.2, Tables 6–8, Figs. 7, 19 | `make_run_pair.py`, `compare_pair.py`, `steering_templates.py`, `steering_paired.py` | `results/steering_paired.csv`, `results/steering_paired_l12_vs_l1.csv`, `results/steering_templates*.csv`, `results/steering_pair_holdout.csv` |
| Flat chart / cylinder vs sphere: held-out comparison, frames and cuts, uncertainty | §3.4, App. F.1, Figs. 8, 20 | `metric_kdim_cent.py`, `metric_fullpca_allscore.py` (full hidden space, no PCA: the figures); `metric_sensitivity.py` (fitters, scoring), `metric_gain_typed.py`, `metric_fullpca_cent.py`; checks: `kdim_compare.py`, `metric_heldout.py`, `metric_l2_control.py`, `metric_harmonic_parity.py`, `pca128_bound.py` | `results/kdim/full/` (figures), `results/kdim/full_vs_8pc.txt`, `results/vs_linmap_typed/` (top-8 PCs), `results/metric_*_summary.csv`, `results/pca128_bound.csv` |
| Correlation with candidate geometries: intervals | App. F.2, Figs. 10, 21–23 | `spectral_intervals.py`, `full_centroids.py`, `figures/qtext_variant_rdms.py` | `results/spectral_intervals.csv`, `results/qtext_variant_intervals.csv`, `results/model_fits_full.csv` |
| Co-occurrence corpus; negative eigenvalues of M*; factorizations; saturation with corpus size | App. F.3, Table 10, Fig. 24 | `cooc/` (see `cooc/README.md`, `cooc/SOURCES.md`) | `results/qtext_*.csv` |
| Generic (non-locational) prompts on the 80/20 scorer | App. D.4, Fig. 15 | `run_noloc.sbatch` (extraction, then `split_pca.py`, k = 8 and 32, 20 splits) | `figures/data/noloc/` |
| R² of the top-16 PCA explained by l=1 and l=2 harmonics; linear decodability of stellar distance (response only) | — | `curiosity.py` | `results/harmonics_r2.csv`, `results/distance_decode.csv` |

`figures/` holds the Mathematica code and the tables that draw the revised figures, and the commands that
rebuild the tables from `data/` (`figures/README.md`).

Conventions: unless a table says otherwise, "5–95%" is the spread over 100 random 80/20 object splits
(the same splits for every model); it shows how much a held-out score depends on which objects are held
out and is not a confidence interval. "95% interval" is a bootstrap interval that resamples objects (query
objects for ablation, source constellations for steering, objects within each type for correlations).
"The six models" are all tested models except gpt-oss-120b, in which the sky representation is weak.
