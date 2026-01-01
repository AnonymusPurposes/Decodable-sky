# Replacing a model's live harmonic sky component

Six models are asked which constellation lies next to constellation **A**. While they
answer, the part of the residual stream that encodes sky position is overwritten with
what the same model's own chart predicts for a constellation **B** on the far side of
the sky. The question is whether the answer moves with it.

The intervention is causal and lives entirely inside the forward pass: no prompt token
changes, and every arm runs the identical prompt batch.

## The map

For one model, take the top-16 PCA scores `z` of the last-token residual stream at a
layer, averaged over prompt templates per object, and fit

    F(u) = b + W phi(u)

where `phi` are the eight nonconstant real spherical harmonics of degrees 1 and 2,
orthonormal on the sphere (`E[phi phi.T] = I`). With `Q` an orthonormal basis of
`col(W)`, the live edit replaces only that image:

    z' = b + (I - Q Q.T)(z - b) + W phi(u_B)

The change is lifted back to raw hidden states through the archive's stored PCA
transform and standardization, and only the lifted difference is added to the original
activation. The complement of the fitted image is preserved up to stored-loading and
activation-dtype precision; this is not raw-space orthogonality.

**Layers.** Every hidden state whose leave-one-object-out sky R² is at least `--frac`
(0.8) of that model's maximum, from [r2_tables/](r2_tables/), minus the final state,
which is post-final-norm and is never hooked.

**Pairs.** Each of the 88 catalogue constellations is A once; its destination B
minimises `(u_A . u_B)^2`, so the two directions are as close to orthogonal as the
catalogue allows. Pairs are never chosen on model behaviour.

**Templates.** Two are evaluated, 0 and 18, giving 176 prompts per arm.

**Token policy.** `sustained`: the replacement is applied at the last prompt token and
at every generated token, so the model cannot drift back while answering.

**Arms.** `untouched`, `self` (A -> A, which must reproduce the untouched answer
unless the lift itself perturbs the model) and `target` (A -> B). The untouched arm is
unpatched, so it is identical across every fit variant and degree — a standing check
that a re-run changed only what it was supposed to.

## Running it

```bash
sbatch fitvariants.sbatch <model_key> <variant> [degree]
sbatch --gpus=2 fitvariants.sbatch mixtral8x22b <variant> [degree]   # 141B in BF16
```

Model keys are the registry in [models_multi.py](models_multi.py): `qwen32b`,
`qwen235b`, `llama33_70b`, `mixtral8x22b`, `mistrallarge123b`, `glm45air`. That file
also records each checkpoint's placement, and why: an FP8 checkpoint driven across two
devices in one process silently corrupts its own logits, so it must stay on one.

`variant` selects how the chart is **fitted**, holding layers, pairs, policy and
evaluation templates fixed:

| variant | templates in the fit | objects in the fit |
|---|---|---|
| `base` | all 25 | 88 constellations |
| `tmplheldout` | all but 0 and 18 | 88 constellations |
| `typed` | all 25 | all 188 objects, one intercept per type |
| `heldout_typed` | all but 0 and 18 | all 188 objects, one intercept per type |

A typed fit gives stars and other objects their own intercept while the chart keeps the
constellation one, so the harmonic block is estimated from every object without
assuming the types share a mean. `degree` is 2 (l=1 and l=2, eight dimensions) or 1
(the dipole alone, three), and results land in `results_fit_<variant>[_l1]/`.

The job runs three steps, which can also be run by hand:

```bash
python prepare_multi.py --model <key> --degree 2 --out prepared_x [--fit-templates exclude-eval] [--fit-scope typed]
python run.py --prepared prepared_x --out run_x --token-policy sustained
python analyze_multi.py --prepared prepared_x --run run_x --out results_fit_base
```

Then, once every model has been scored into the same directory:

```bash
python collect_multi.py --results results_fit_base --out results_fit_base/all_models_summary.csv
python rescore_multi.py --trials results_fit_base --catalog prepared_x/catalog.json \
                        --templates prepared_x/templates.json --out results_fit_base_rich \
                        [--compare <an earlier rescore>]
```

## Scoring

Every arm is scored against **both** endpoints, so one set of mutually exclusive
categories describes all three:

| category | meaning |
|---|---|
| `no_constellation` | no catalogue constellation could be read |
| `ramble_A` | the completion only restates the question and describes A |
| `equals_A` / `equals_B` | the answer is the source / the destination |
| `within30_A_not_A` / `within30_B_not_B` | within 30° of that endpoint, but neither A nor B |
| `other` | a constellation far from both |

Destinations are near-orthogonal to their sources, so the two within-30 classes cannot
both fire, and the categories sum to 1.

`analyze_multi.py` scores with the deliberately conservative `core.parse_answer`, which
reads only a leading catalogue name. `rescore_multi.py` re-reads the same completions
with [parse_rich.py](parse_rich.py) and needs no GPU, so the parser can be corrected
after the fact and the run re-scored in a second. It writes one verdict per row, in the
same columns, and a `parser_changes.csv` against any earlier scoring so a parser change
can be audited trial by trial rather than through its summary.

The forgiving parser earns its name: it folds diacritics (models write *Boötes*), reads
multiple-choice lists as a candidate **set** scored by mean distance, resolves
alternative names (*Southern Triangle*, *Pisces Austrinus* misspelled, *Scorpio*),
deletes the question where the model restates it, and separates a completion that
merely describes its own source from one that answers. On some models this is the
difference between an apparent 60% muteness and 1%, so the frozen reading is kept in
the summary as `frozen_no_constellation` and `recovered_fraction`.

## Inputs, and what is not in this repository

`prepare_multi.py` reads two things per model:

* **`<model>_sky_r2_16.csv`** — bundled in [r2_tables/](r2_tables/).
* **a full 128-component PCA archive** — *not* bundled. `PCA128/*.npz` in this
  repository are cropped to the top 16 components and carry no PCA transform, which is
  enough to refit charts and build pairs but not to run the intervention, which has to
  map PCA space back to raw hidden states. Regenerate with `extract_pca.py`.

Both come from the location-prompt extraction. The repository's
`data/astro_prompts_location.csv` is the prompt set these use; on the GPU host the same
file is called `astro_prompts_location_fixed.csv`, and the two are byte-identical. The
older, differently-worded location prompts are not used anywhere here.

PCA and per-neuron standardization stay globally fitted, using the stored components.
Evaluation is conditional on that transductive basis and is not a held-out test of the
extraction pipeline. Catalogue centres are proxies for extended constellation areas.

## Checks and tests

`run.py` verifies the live PCA scores against the archived ones before patching
anything, reproduces the untouched arm's likelihoods and tokens exactly, and records a
trace of every intervention. `python -m unittest test_harmonic test_hooks` covers the
chart geometry (target match, complement preservation, idempotence, self clamp), the
leading-name parser, layer-group validation, and the residual hook's indexing.

## History

Earlier phases — a single-model pilot at individual layers (job 6520), the joint
46–87 layer group (6523) and the sustained token policy (6524), each with its own
protocol notes and output bundle — were superseded by this battery and removed from
the tree to keep it readable. They remain in the git history, before the commit that
deleted them.
