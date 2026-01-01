# Figures of the revised paper

`figures.wl` (Mathematica) draws the figures that the review response changed or added, from the tables in
`data/` only. Set `figDir` to this folder and `Get` the file, or paste its sections (one per
`(* ==== title ==== *)` marker) into a notebook and run Setup first. PDFs go to `pdf/` (not tracked).
`rebuttal_figures_1.nb` is the notebook several of the paper's PDFs were exported from; it holds the same
sections, with some styling edits, except the generic-prompt, no-PCA linmap and per-model ablation sections,
which are only in `figures.wl`.

| Section | Paper figure | Output | Data and source |
|---|---|---|---|
| Figure 2 | 2 | `fig2_r2.pdf`, `fig2_ang.pdf` | `split/`, `null/`: 80/20 object splits with standardization and PCA fitted on the training objects (`../split_pca.py`, `../split_layer_csv.py`), median and 5–95% over 100 splits; angle null from shuffled labels (`../angle_null.py`) |
| Figures 3 and 12 | 3, 12 | `<model>_dump.pdf` | the same curves (thick top-4, thin top-8 with bands; gpt-oss top-16 / top-32) and the PCA strips (`strips/`, `../../correlations.py`) |
| Figures 5, 16, 17 | 5, 16, 17 | `ablation.pdf`, `ablation_1-4.pdf`, `ablation_5-7.pdf` | `ablation/`: medians with 95% intervals over query objects, shuffled-answer chance level (`../intervals.py` on `../data/ablation`) |
| Figure 18 | 18 | `ablation_parsed.pdf` | `ablation/all_models_summary.csv` |
| Figures 7 and 19 | 7, 19 | `steering.pdf` | `steering/`: map refitted without A and B, five evaluation templates (`../steering_templates.py` on `../data/steering`) |
| Figures 8 and 20 | 8, 20 | `vs_chart_linmap_nopca.pdf`, `vs_cylinder_linmap_nopca.pdf`, per model `<model>_vs_<name>.pdf` | `vs/*_nopca.csv`: held-out R² gain of adding one map's coordinates to the other's, ridge map onto the object centroids in the whole standardized, type-centred hidden space (`../metric_kdim_cent.py --full-only`, `../metric_fullpca_allscore.py --fast`; `../results/kdim/full`), 5–95% over 100 splits, thin lines the range over frames and cuts |
| Figures 10, 21–23 | 10, 21–23 | `models_rdm.pdf`, `models_rdm_<const\|all>_<1-3\|4-6>.pdf` | `rdm/`: correlation with candidate geometries, neural RDM in the full standardized residual stream (`../full_centroids.py`), 95% intervals from resampling objects (`../spectral_intervals.py`); Q_text columns at 975 B tokens |
| Figure 24 | 24 | `qtext_saturation.pdf` | `rdm/qtext_saturation.csv` (`../cooc/qtext_saturation.py` on `../data/cooc`) |
| Figure 15 | 15 | `noloc.pdf`, `<model>_noloc.pdf` | `noloc/`: generic prompts on the same scorer as Figure 2, 20 splits, top-8 and top-32 (`../run_noloc.sbatch`) |

Rebuilding the tables from the shipped data (run from the repository root):

```
python rebuttal/intervals.py --ablation rebuttal/data/ablation --out rebuttal/results
python rebuttal/steering_paired.py --old rebuttal/data/steering/steering_pair --new rebuttal/data/steering/steering_templates --out rebuttal/results
python rebuttal/figures/qtext_variant_rdms.py --sets rebuttal/data/cooc/sets_ext/ext_everything.npz rebuttal/data/cooc/sets/wiki.npz \
       --forms rebuttal/cooc/surface_forms_v2.csv --objects data/astro_objects.csv --out spectral/rdms
python rebuttal/spectral_intervals.py --spectral spectral --out rebuttal/results/spectral_intervals.csv
```

`spectral_intervals.py` scores every table in `spectral/rdms/` against the neural RDM read through
`spectral/paths.py` (full-space centroids in `rebuttal/data/fullstd`; `SKY_NEURAL=pca128` switches to the
bundled top-128 PCA scores). The tables of Figures 2, 3, 8, 12, 15 and 20 need the raw
activations (not shipped); their outputs are in `data/` and `../results/`. `prepare_data.py` is the script
that collected `data/` on the authors' machine.
