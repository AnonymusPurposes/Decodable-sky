# Decodable Sky

Probing whether large language models internally represent **astronomical objects by
their position on the sky**. Each object (a star, constellation, or deep-sky object)
is labelled by a unit direction vector `(sin Dec, cos Dec sin RA, cos Dec cos RA)`, and
we ask how linearly decodable that direction is from a model's residual stream.

The pipeline has two steps:

* **`extract_pca.py`** — for a chosen model, record the top-K PCA components of the
  last-token residual stream (per layer, K set with `--npca`, default 128), in pure
  autocomplete mode, over a grid of location-style prompts × objects. Produces one
  compact `<model>_pca128.npz` per model.
* **`correlations.py`** — turn those `.npz` files into per-layer, per-direction
  correlation / density tables that show *where* (which PCA directions, which layers)
  the sky-direction (or object-type) signal lives.
* **`pc_table.py`** — small utility: for one model/layer/principal-component, dump an
  objects × prompts table of that PC's value (handy for eyeballing what a single
  direction encodes).
* **`extract_pca_gptoss_harmony.py`** — a gpt-oss-specific variant of the extractor.
  gpt-oss is post-trained on the *harmony* format, so bare-text autocomplete is out of
  distribution; this wraps each prompt in a minimal harmony conversation and seeds the
  phrase into the assistant `final` channel (empty `analysis` channel before it). It
  writes `gptoss120b_harmony_pca128.npz`, a drop-in for `correlations.py`.

Alongside the probing pipeline there is a **behavioural** counterpart — does a model
*explicitly know* where things are? — in `radec_recall.py` / `radec_score.py` (see
[Coordinate recall test](#coordinate-recall-test-behavioural) below).

## Where each part of the paper lives

| Paper | Folder |
|---|---|
| Decodability (80/20 splits), PCA charts, generic prompts, empirical nulls | `extract_pca.py`, `correlations.py`, [rebuttal/](rebuttal/) (`split_pca.py`, `angle_null.py`, `split_perm.py`, `fresh_test.py`) |
| Ablation | [ablation/](ablation/), intervals in [rebuttal/](rebuttal/) (`intervals.py`) |
| Steering | [steering/rotation/harmonic/](steering/rotation/harmonic/), paired statistics in [rebuttal/](rebuttal/) |
| Sphere vs flat chart vs cylinder | [rebuttal/](rebuttal/) (`metric_*.py`) |
| Correlation with candidate geometries, co-occurrence | [spectral/](spectral/), [rebuttal/cooc/](rebuttal/cooc/), `rebuttal/spectral_intervals.py` |
| Radial analysis | `build_star_metadata.py`, `radial_length_correlations.py` |
| J-space | [jlens_pipeline/](jlens_pipeline/) |
| Coordinate recall (gpt-oss anomaly) | `radec_recall.py`, `radec_score.py`, `recall/` |
| Projections in Figure 1 and Appendix H | `skyproj.py`, `make_ortho_projections.py` |

[rebuttal/README.md](rebuttal/README.md) maps every analysis added in the revision to its scripts, shipped
data and result tables; [rebuttal/figures/](rebuttal/figures/) redraws the revised figures.

## Harmonic steering experiment

Decodability is a claim about what a representation *contains*; steering asks whether
the model **uses** it. In [steering/rotation/harmonic/](steering/rotation/harmonic/)
the degree-1+2 harmonic image of the sky direction is fitted in a model's top-16 PCA
space and then, mid-generation, replaced with what that same model's chart predicts
for a constellation on the opposite side of the sky — one that minimises
`(u_A . u_B)^2` against the one the prompt asked about.

Six models are run over all 88 constellations, in three arms (untouched, self
replacement, destination replacement), with the replacement sustained through every
generated token. The answers move: on the stronger models most steered completions
name the destination or a constellation within 30° of it, while the dipole alone
achieves roughly half of that. How the map is fitted is varied separately — holding
out the evaluated prompt templates, and fitting across all object types with an
intercept per type — which costs a few points of accuracy and buys a genuinely
held-out measurement.

Rerunning the intervention requires a full 128-PC archive with its saved PCA
transform; the bundled cropped top-16 files are enough to refit the charts but not to
map PCA space back to raw hidden states. The experiment's own
[README](steering/rotation/harmonic/README.md) covers the method, the fit variants,
the answer categories, and the parser that reads the completions.

## Install

```bash
pip install -r requirements.txt
# FP8 checkpoints (e.g. Qwen3-235B-*-FP8) additionally need:
pip install "kernels>=0.12,<0.13"
# the coordinate recall test (radec_recall.py) uses vLLM instead of transformers:
pip install vllm
```

## Usage

### 1. Extract PCA features

```bash
# a model from the built-in registry
python extract_pca.py --model qwen32b

# choose how many PCA components to keep (default 128)
python extract_pca.py --model qwen32b --npca 64

# multi-GPU sharding for large models
CUDA_VISIBLE_DEVICES=0,1 python extract_pca.py --model mistrallarge123b --batch 16

# any other HuggingFace causal LM
python extract_pca.py --hf-repo some/other-model --name mymodel
```

Output: `pca/<name>_pca<K>.npz` (where `K` is `--npca`) containing

| key | shape | meaning |
|---|---|---|
| `pca` | `(n_samples, n_layers+1, K)` | top-K PCA of the standardized residual stream, per layer |
| `obj_ids` | `(n_samples,)` | index into `names` for each sample |
| `pr_ids` | `(n_samples,)` | prompt-template index for each sample |
| `Yunit` | `(n_objects, 3)` | unit sky-direction of each object |
| `types` | `(n_objects,)` | `star` / `constellation` / `other` |
| `names` | `(n_objects,)` | object name |

`n_samples = n_prompts × n_objects`.

### Bundled PCA features (`PCA128/`)

For convenience the repo ships the extracted features in `PCA128/<model>_pca128.npz`
so the analysis scripts run without re-extracting (which needs the GPUs + weights). To
keep the repo lightweight these bundled files are **cropped to the top-16 PCA
components** (`pca` shape `(n_samples, n_layers, 16)`; ≈70 MB total instead of ≈540 MB).

* Every analysis here uses `k ≤ 8`, so the crop is transparent; scripts clamp a larger
  `--k` / `--npca` to the 16 available and print a note.
* **Stored-basis** tables (`correlations.py`, `skyproj.py`, `pc_table.py`) read the
  leading components directly, so their first-16 columns are **identical** to full-128.
* The decodability tests of the revised paper refit standardization and PCA inside each
  training split from the raw activations (`rebuttal/split_pca.py`), so they do not use these
  files; the ablation and steering experiments and the PCA charts do.
* Need all 128 components? Regenerate with `extract_pca.py --out PCA128` (see above),
  which overwrites `PCA128/<model>_pca128.npz` with the full file.

### 2. Correlation / density tables

`correlations.py` reads a `<model>_pca128.npz` (always the 128-component file; `--npca`
selects how many leading directions to use/show) and writes one per-layer table:

```bash
python correlations.py --model qwen235b --regime sky_cov_scaled --npca 128 \
    --indir PCA128 --out correlations
python correlations.py --model qwen235b --regime sky_r2 --npca 8
```

Output: `<out>/<model>_<regime>_<N>.csv`. The four regimes:

| regime | target | PCA metric | columns |
|---|---|---|---|
| `sky_cov_scaled` | sky direction (whitened to Cov = I₃) | rescale each layer as a whole so PC-1 has variance 1 — keeps relative variances | `layer, trace, d1..dN` |
| `sky_cov_equal` | sky direction (whitened) | whiten each PCA direction to variance 1 — per-direction squared canonical correlation | `layer, trace, d1..dN` |
| `type_cov_equal` | object type (star +1 / constellation −1 / other 0, standardized) | whiten each PCA direction | `layer, trace, d1..dN` |
| `sky_r2` | raw sky unit vector | first N directions, leave-one-object-out OLS (no ridge) | `layer, r2, angerr_deg, angerr_median_deg` |

For the density regimes `d_j = Σ_a Cov(w_j, target_a)²` on the (metric-transformed)
PCA direction `j`; `trace` is their sum.

### 3. Prediction-length correlations (radial analysis)

What does the *magnitude* of the decoded sky vector track? First,
`build_star_metadata.py` collects per-object metadata into `astro_metadata_full.csv`
(`name, type, Vmag, distance_ly, zipf`): stellar distances from SIMBAD trigonometric
parallax (`distance_ly = 3.261564 · 1000 / plx_mas`, each star resolved by name through
the SIMBAD sim-script interface) and text-corpus frequency from `wordfreq` (the appended
" constellation" tag is stripped so a constellation is scored by its real name). `Vmag`
comes from the catalog's `notes`, with SIMBAD's V as a fallback.

```bash
pip install wordfreq                 # SIMBAD access itself uses only the stdlib
python build_star_metadata.py        # data/astro_objects.csv -> astro_metadata_full.csv
```

Then `radial_length_correlations.py`, for every model with a `PCA128/<model>_pca128.npz`,
picks the top-8 layer with the smallest **median** angular error (from
`correlations/<model>_sky_r2_8.csv`). It scores each object's angular error with
leave-one-object-out OLS (exactly as in `correlations.py`'s `sky_r2_table`) and drops
objects whose mean-square angular error exceeds 2× the across-object average. For every
surviving object it then measures the **length of that object's mean sky projection** —
the 3D predictions of an OLS map fit on **all** samples (no leave-one-out), averaged over
the object's prompts and then normed (length of the mean vector, not the mean of the
per-sample lengths) — and correlates it with distance, frequency, −Vmag, and the type
indicators.

```bash
# prerequisite: the sky_r2 tables the layer choice reads
for m in qwen32b qwen235b gptoss120b gptoss120b_harmony llama33_70b mixtral8x22b mistrallarge123b glm45air; do
  python correlations.py --model $m --regime sky_r2 --npca 8
done
python radial_length_correlations.py     # -> radial_corr/corr_summary.csv + <model>_perobject.csv
```

`radial_corr/corr_summary.csv` has one row per model: the chosen layer, kept/dropped
counts, and the seven correlations — Spearman for distance (stars); Pearson for frequency
(stars, constellations), −Vmag (stars) and the star / constellation / other indicators
(all surviving objects) — each with its p-value and subset size.

## Coordinate recall test (behavioural)

A separate, behavioural probe: rather than decoding a *represented* direction, just
ask each reasoning model what it explicitly knows. For every non-constellation object
(85 stars + 15 deep-sky objects; constellations have no single RA/Dec) the model is
prompted, in its own chat template with thinking/reasoning **on**:

> Give the right ascension and declination of *X* with no commentary.

Then the answer's RA/Dec is parsed and compared to the true J2000 position by
great-circle angular error. Only the reasoning-capable models are used
(`qwen32b`, `qwen235b`, `gptoss120b`, `glm45air`).

```bash
# 1. generate answers (one model at a time; uses vLLM)
python radec_recall.py --model qwen32b        # -> recall/radec_answers_qwen32b.csv
python radec_recall.py --model qwen235b       # tp=2; qwen32b/gptoss120b tp=1; glm45air tp=1

# 2. parse + score every answer table in the folder
python radec_score.py --indir recall --outdir recall
```

`radec_score.py` parses the many formats models emit (lettered `06h45m08.9s`,
colon `06:45:08.9`, space-sexagesimal `06 45 08.9`, decimal degrees, and gpt-oss's
harmony `final` channel). Answers with no committed coordinate — e.g. a model that
reasons past its token budget without answering — are scored **90°** (a random guess
on the sphere averages 90° away). Outputs, in `recall/`:

| file | columns |
|---|---|
| `radec_answers_<model>.csv` | `name, type, true_ra_deg, true_dec_deg, answer` |
| `radec_error_<model>.csv` | `… pred_ra_deg, pred_dec_deg, parsed, angular_error_deg` |
| `radec_error_summary.csv` | one row per model (below) |

Results over the 100 objects (mean° counts unparsed answers as 90°):

| model | parsed | mean err (all) | mean err (parsed) | median | % ≤ 1° |
|---|---|---|---|---|---|
| `glm45air` | 100/100 | 1.83° | 1.83° | 0.0° | 89% |
| `qwen235b` | 99/100 | 2.15° | 1.26° | 0.0° | 89% |
| `gptoss120b` | 91/100 | 11.91° | 4.19° | 0.0° | 73% |
| `qwen32b` | 100/100 | 13.83° | 13.83° | 0.6° | 54% |

The large reasoning models place bright stars to within an arcminute. `qwen32b` is much
weaker (it confidently misidentifies fainter stars); `gptoss120b`'s parsed answers are
good, but on 9 obscure southern stars it overthinks past the 8192-token budget and never
commits a final answer (scored 90°).

## Models

The registry in `extract_pca.py` and `correlations.py` (`--model` keys): `qwen32b`,
`qwen235b`, `gptoss120b`, `llama33_70b`, `mixtral8x22b`, `mistrallarge123b`, `glm45air`.
`extract_pca.py` also accepts any other causal LM via `--hf-repo`.

## Data

* `data/astro_objects.csv` — 188 objects (named stars with V < 2.5, all 88 constellations,
  and 15 deep-sky objects) with equatorial coordinates and the unit direction vector.
* `data/astro_prompts_location.csv` — location-style prompt templates; `X` marks where the
  object name is substituted.
* `data/astro_prompts_no_location.csv` — the generic (non-locational) templates of Appendix C.
* `rebuttal/prompts_fresh.csv` — 25 new location-style templates, fixed before extraction (Appendix D.1).
