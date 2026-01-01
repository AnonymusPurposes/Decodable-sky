# Low-frequency sky-subspace ablation

Decodability says a representation is present. Ablation asks whether removing it costs
the model anything. Here the degree-1 and degree-2 harmonic image of the sky direction
is fitted in a model's top-16 PCA space and projected out of the residual stream at
every token position, prompt and generated alike, while the model answers location
questions about 188 objects across 25 prompt templates.

This directory holds the minimum needed to reproduce `results_harm_frac80/`.

## Conditions

The projection is applied at every hidden-states level whose sky R² reaches 80% of that
model's maximum, so the ablated band follows each model's own depth profile.

| condition | removed |
|---|---|
| `none` | nothing, baseline |
| `sky3` | the degree-1 span, the dipole alone |
| `harm8` | the degree-1 and degree-2 span together |
| `comp8` | the complementary 8 dimensions of the top-16 space, a control |
| `l2pure` | the degree-2 part orthogonal to degree-1 |
| `l2raw` | the degree-2 fit span as fitted, which overlaps degree-1 |

The projection is affine, onto a subspace through the data centroid rather than the
origin, so removed coordinates land on their mean instead of zero. `ablate_harm.py`
verifies idempotence and centroid fixity against the model's own hidden states before
generating.

## Inputs

| input | where from | default location |
|---|---|---|
| `<model>_pca128.npz` | `extract_pca.py`, a full run | `../pca` |
| `<model>_sky_r2_16.csv` | `correlations.py` | `../steering/rotation/harmonic/r2_tables` |
| `astro_prompts_location.csv` | bundled | `../data` |
| `astro_objects.csv` | bundled | `../data` |
| `bright_stars_v4.csv` | `build_bright_stars.py`, from SIMBAD | here |

The PCA archive has to be a full `extract_pca.py` run. The projection needs
`components`, `feat_mean`, `feat_std` and `pca_mean` to move between PCA coordinates
and raw hidden states, and the cropped top-16 archive bundled in `PCA128/` does not
carry them.

## Running it

```
./run_harm_all.sh                       # every model, five lanes, resumable
./run_harm.sh 0 qwen32b results_harm_frac80    # one model
python collect_ablation.py results_harm_frac80 --objects ../data/astro_objects.csv
```

`run_harm.sh` resolves its inputs from `PCA_DIR`, `R2_DIR`, `PROMPTS` and `PYTHON`,
each falling back to the defaults in the table above. Models that do not fit on one
device take `--device-map auto`.

Generation writes `completions_<model>.csv` and `diagnostics_<model>.json` per model.
`collect_ablation.py` then calls `score_ablation.py` on each, producing
`completions_<model>_scored.csv` and `completions_<model>_summary.csv`, and merges the
per-condition summaries with the run diagnostics into `all_models_summary.csv`.

## Scoring

`score_ablation.py` reads what the model actually named and how far that is from the
object asked about, as a great-circle angle. The parser is ported from the steering
experiment's and adapted to a catalogue that also holds stars and deep-sky objects. It
flattens line breaks before reading, folds accents so `Boötes` matches `Bootes`,
strips the prompt's own phrasing so an echo is not read as an answer, and accepts
Latin genitives, aliases and English names. Enumerated answers are scored at the mean
angle of their members. An opening reference to the celestial equator, the ecliptic or
a celestial pole is scored as a landmark, at the distance to that circle or point.
Naming the prompt object itself is counted separately from naming nothing.

Reported medians carry a bootstrap confidence interval that resamples prompt objects as
clusters, so it is an interval on the median rather than the spread of the angles.
