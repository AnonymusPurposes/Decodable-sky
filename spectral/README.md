# spectral — which structure explains the residual-stream sky?

The residual stream carries a decodable map of the sky. This folder asks what that map's
*geometry* looks like: given the observed representational dissimilarity matrix (RDM) of 188
catalogue objects, how well is it explained by

- the **sphere** — the true angular separation of the objects;
- **spherical harmonics** — the sphere with a tunable smoothness, coordinate `(l, m) =
  f(l(l+1)) Y_lm`;
- the **constellation graph** — Laplacian eigenmap of "constellations that share a border are
  adjacent, each star or deep-sky object hangs off its own constellation";
- the **co-occurrence graph** — Laplacian eigenmap of "these two names occur within 10 words of
  each other in English Wikipedia, in an astronomical context";
- the **QWEM co-occurrence embedding** — the closed-form word embedding of the same counts.

Every model is scored the same way: build its RDM of squared distances, take the upper
triangle, and report the Pearson correlation with the neural one (equivalently the RSA loss
`2 - 2r` on centered unit-norm RDMs).

## Result

Pearson r, constellation pairs (`fits/model_fits.csv`, regime `const`):

| model | sphere | harmonics | border graph | cooc graph | cooc no-list | QWEM art | QWEM artnl | QWEM ungated |
|---|---|---|---|---|---|---|---|---|
| mistrallarge123b | 0.807 | **0.850** | 0.758 | 0.337 | 0.334 | 0.507 | 0.377 | 0.255 |
| qwen32b | 0.778 | **0.808** | 0.745 | 0.360 | 0.336 | 0.547 | 0.409 | 0.320 |
| llama33_70b | 0.771 | **0.808** | 0.700 | 0.339 | 0.358 | 0.512 | 0.392 | 0.264 |
| mixtral8x22b | 0.700 | **0.723** | 0.692 | 0.433 | 0.417 | 0.421 | 0.299 | 0.241 |
| glm45air | 0.686 | **0.738** | 0.704 | 0.421 | 0.363 | 0.487 | 0.316 | 0.298 |
| qwen235b | 0.657 | **0.723** | 0.712 | 0.475 | 0.431 | 0.446 | 0.250 | 0.257 |
| gptoss120b_harmony | 0.444 | **0.465** | 0.397 | 0.101 | 0.231 | 0.293 | 0.213 | 0.112 |
| gptoss120b | 0.367 | **0.388** | 0.329 | 0.188 | 0.272 | 0.232 | 0.114 | 0.190 |

and all 188 objects, where nothing co-occurrence-related is scored (regime `all`):

| model | sphere | harmonics | constellation graph (d) |
|---|---|---|---|
| mistrallarge123b | 0.659 | **0.694** | 0.479 (d = 1) |
| qwen235b | 0.605 | **0.667** | 0.542 (d = 1) |
| llama33_70b | 0.609 | **0.649** | 0.450 (d = 1) |
| qwen32b | 0.608 | **0.638** | 0.507 (d = 1) |
| glm45air | 0.579 | **0.633** | 0.435 (d = 1) |
| mixtral8x22b | 0.491 | **0.506** | 0.365 (d = 0.001) |
| gptoss120b_harmony | 0.253 | **0.272** | 0.215 (d = 0.001) |
| gptoss120b | 0.202 | 0.222 | **0.332** (d = 0.001) |

Reading:

- **The sphere wins, and smoothing it helps a little.** Harmonics beat the plain sphere in every
  model by 0.01–0.06, always with a strong exponential decay (`k ≈ 0.13–0.20` on the eigenvalue).
  That gain is close to the hard ceiling: by the addition theorem the harmonic family can only
  ever produce an *isotropic* function of angle, and the best arbitrary function of angle scores
  0.574 against qwen32b's all-pair RDM versus 0.541 for the plain sphere (`sanity_checks.log`).
- **No graph beats position.** The border graph gets within 0.03–0.06 of the sphere on
  constellation pairs, but never above it; the one exception anywhere is gptoss120b on all 188
  objects, whose sky map is weak enough that the membership graph explains more than position.
- **Corpus co-occurrence explains much more than it first appears**, provided the counting is
  gated (see below): QWEM r 0.42–0.55 in the six strong models, against 0.24–0.32 for the same
  construction on ungated counts.
- **Excluding lists and tables costs accuracy rather than buying it** — QWEM artnl is below QWEM
  art everywhere, the no-list graph is worse in five of eight models. Enumerations such as
  "bordering constellations" or "the stars of Orion" are themselves sky-structured. Useful as a
  control: the co-occurrence signal is not an artefact of list articles, but it is partly carried
  by them.

## How the two sides are built

**Neural RDM.** For each model, take every layer whose leave-one-out sky R² at k = 16 is at
least 80 % of that model's maximum (`r2_tables/<model>_sky_r2_16.csv`). Per layer: average the
top-128 PCA scores over the 25 location templates to get one centroid per object, take squared
distances between centroids, center and scale to unit norm *over the analysed pairs*, and
average over the band. The per-layer normalization matters — without it the deepest layer, whose
residual-stream norms are largest, dominates the mean.

**Two regimes.** `const` scores the 88 constellations (3828 pairs) and restricts every graph to
those nodes, so the constellation graph is the bare border graph. `all` scores all 188 objects
(17578 pairs) with the position models and the full constellation graph, where a
constellation→member edge has length `d` (Laplacian weight `1/d`), swept over `d ∈ (0, 1]`.

**Type-centroid merge** (in `all`): constellations, stars and "other" objects sit in three
separate blocks of the residual stream — block means alone are ~19 % of the RDM variance — and no
positional model can explain the difference between *writing about a constellation* and *writing
about a star*. Every embedding, neural and target alike, is therefore shifted per type so the
three centroids coincide (`T = I - Σ_t 1ₜ1ₜᵀ/nₜ`); for a target that exists only as squared
distances, the same shift goes through its Gram matrix. In `const` there is one type, so this is
ordinary centering and changes nothing.

**Decay.** Every spectral target uses `f(λ) = exp(-kλ)`, with `k` scanned on a 401-point log grid
over `[1e-3, 30]` and refined by bounded Brent. Graph targets additionally try both the
combinatorial Laplacian `D - A` and the normalized `I - D^(-1/2) A D^(-1/2)`, and keep the better
(the normalized one wins nearly everywhere).

## Co-occurrence counting

`cooc_count.py` counts over the same dump as Karkada et al. 2026 (HF `wikimedia/wikipedia`
20231101.en): 3,370,649 articles ≥ 200 tokens, 2.85 B tokens, tokenized as in that paper, plus
Greek-letter spelling and an NFKD fold so "Boötes" does not become "bo tes". Names come from
`graphs/surface_forms.csv` (322 forms: 188 canonical names, 85 Bayer designations, 50 aliases),
matched longest-first, and a multi-word form occupies one position so distances are measured in
words between names. Pair events are histogrammed by distance 1..16, so any window ≤ 16 is a
slice. Four nested variants:

| variant | mentions | tokens counted over | rule |
|---|---|---|---|
| `raw` | 782,009 | 2.85 B | every mention |
| `para` | 33,262 | 29.4 M | the paragraph is astronomical (a strong cue, or ≥ 2 weak cues and no astrology cue) |
| `art` | 23,815 | 8.97 M | ... and the article carries ≥ 5 strong cues per 1000 tokens |
| `artnl` | 22,580 | 8.89 M | ... and the mention is running text, not a list or table entry |

Two details that decide whether the numbers mean anything:

- **Polysemous names need a confirm word.** `crater`, `hydra`, `phoenix`, `cancer` and `sculptor`
  only count with "constellation" within ±10 tokens; `peacock` needs "star". 5,401 mentions were
  rejected this way. This is what lets all 88 constellations be used, including the seven names
  that earlier work had to drop. The article-level cue density is the other half: the paragraph
  gate alone admits Saint Seiya, the *Andromeda* TV series, Gemini missions and LEO satellites.
- **A gated count needs a gated denominator.** QWEM's expectation `e_ij = 2 S_f u_i u_j / N` is
  only meaningful if `N` is the token count the `u_i` were counted over. Astronomical contexts are
  0.315 % of the corpus, so using the full corpus size underestimates `e_ij` ~280× and pushes
  `M*` toward +2 everywhere. `cooc_count.py` therefore accumulates `n_tok_para/art/artnl` and
  `build_qwem_rdm.py` uses them.

The no-list rules mark a mention as list context if the article is a "List of ...", if its line
names ≥ 5 distinct objects at fewer than 8 tokens apiece, or if ≥ 4 distinct objects fall within
±4 tokens. All three count *objects*, not mentions: "Canopus (Alpha Carinae)" is one object under
two adjacent names, and counting mentions flags ordinary prose as a list. Dropping the mention
rather than the pair keeps `u`, `D` and `Cd` consistent. This removes 5.2 % of mentions but 29 %
of pair events, and the resulting graph is still connected over the 88 constellations (1
component, no isolated node, mean degree 13.1 against 26.9).

## Files

Inputs this folder does not produce: `../data/astro_objects.csv` (or `../data_astro_objects.csv`)
and `../PCA128/<model>_pca128.npz`; `paths.py` finds them in either layout.

| | |
|---|---|
| `sanity_checks.py` | pre-fit checks: isotropic ceiling, type-block confound, split-half reliability (`sanity_checks.log`) |
| `build_surface_forms.py` | catalogue + IAU-CSN → `graphs/surface_forms.csv` (needs `../ablation/score_ablation.py` for the genitive forms; the CSV is the shipped artifact) |
| `build_constellation_graph.py` | astropy's IAU boundary table → `graphs/graph_constellation.npz` (255 borders, corner-only contacts excluded) |
| `download_wiki.sh`, `launch_cooc.sh` | fetch the 41 shards; run `cooc_count.py` on all of them, one process each (~81 s per shard) |
| `cooc_count.py`, `merge_cooc.py` | the counting above → `graphs/cooc_counts_wikipedia.npz`, plus sampled contexts per form and gate level in `graphs/cooc_examples_wikipedia.jsonl` for auditing |
| `export_cooc_graph.py` | counts → `graphs/graph_cooccurrence.npz` (`A` from `art`, `A_nolist` from `artnl`) and a diagnostic summary |
| `build_qwem_rdm.py` | counts → an RDM table in `rdms/`, one per variant |
| `fit_models.py` | everything above + the neural side → `fits/model_fits.csv` |

```bash
python sanity_checks.py
python build_constellation_graph.py
bash  download_wiki.sh && bash launch_cooc.sh counts_wiki     # on the machine with the corpus
python merge_cooc.py counts_wiki graphs/cooc_counts_wikipedia.npz graphs/cooc_examples_wikipedia.jsonl
python export_cooc_graph.py
python build_qwem_rdm.py --counts graphs/cooc_counts_wikipedia.npz --variant art   --min-u 10 --out rdms/qwem_art_L16.csv
python build_qwem_rdm.py --counts graphs/cooc_counts_wikipedia.npz --variant artnl --min-u 10 --out rdms/qwem_artnl_L16.csv
python fit_models.py                       # ~7 min for 8 models x 2 regimes
```

`--min-u 10` rather than 40: gated counts are ~30× smaller than raw ones, and a threshold of 40
would drop 15 constellations; at 10 all 88 survive, so every table scores the full 3828 pairs.
`fit_models.py` picks up any RDM table in `rdms/` as a parameter-free target, so adding a
competing embedding means dropping one more 188×188 CSV there.

## Neural data

`paths.py` reads the neural side from the object centroids in the full standardized residual stream
(`rebuttal/data/fullstd`, made by `rebuttal/full_centroids.py`), as in the revised paper. Set
`SKY_NEURAL=pca128` to use the bundled top-128 PCA scores in `../PCA128` instead (the footing of the
first version); `sanity_checks.py` needs this setting, since it reads per-prompt rows.
`fits/model_fits.csv` holds the fits on the full-space footing.
