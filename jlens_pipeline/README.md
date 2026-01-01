# Jacobian-lens pipeline

Three scripts: fit a Jacobian lens on an open-weights decoder, measure how its
per-layer transport operators resemble each other, and test whether the sky
subspace of the Decodable-sky probes is one of the directions the lens singles out.

Each script takes every path as a flag, with an environment-variable fallback and a
relative default, so nothing here assumes a particular machine or directory layout.

| Script | Produces |
|---|---|
| `fit_lens_run.py` | `<slug>_jacobian_lens.pt`, `<slug>_effdim.csv`, `<slug>_meta.json` |
| `lens_cka.py` | `<slug>_cka_in.csv`, `<slug>_cka_out.csv`, `<slug>_cos.csv`, `<slug>.npz` |
| `jsky_align.py` | `<slug>_jsky.csv` |

## Requirements

`torch`, `numpy`, `transformers`, `huggingface_hub`, `pyarrow`, and the upstream
[jacobian-lens](https://github.com/anthropics/jacobian-lens) package (`jlens`,
Apache-2.0). Install `jlens`, or point `JLENS_UPSTREAM` at a checkout. One CUDA
device is enough for everything except fitting the largest models.

Directory flags, each with an environment fallback:

| Flag | Environment | Default |
|---|---|---|
| `--outdir` / `--lensdir` | `LENS_DIR` | `./lenses` |
| `--outdir` of `lens_cka.py` | `CKA_DIR` | `./cka` |
| `--pcadir` | `PCA_DIR` | `./pca` |
| `--outdir` of `jsky_align.py` | `JSKY_DIR` | `./jsky` |

## 1. Fit the lens

```
python fit_lens_run.py --model Qwen/Qwen3-32B --slug qwen3-32b --outdir lenses
```

Fits `J_l = E[d h_final / d h_l]` for source layers `0..L-2` against target `L-1`,
accumulating a running mean over WikiText-103 documents. It checkpoints atomically,
so a restart resumes rather than redoing work, and stops early once the running mean
stops moving.

Prompts come from `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split train,
at a pinned revision. The shards are fetched through the HuggingFace cache on first
use, or read from `WIKITEXT_DIR` if you already have a copy. This mirrors the corpus
the upstream package uses, though the selection differs: this script skips section
headers, truncates documents and shuffles with a fixed seed.

Two failure modes are guarded. Quantized checkpoints driven across several devices
in one process produce silent garbage, so the fit is bf16 only and pins itself to a
single device when one GPU is visible. A coherence gate then measures next-token
cross-entropy on real text and refuses to fit if the model looks broken.

## 2. Layer-by-layer CKA

```
python lens_cka.py --lensdir lenses --outdir cka
```

Linear CKA between the `J_l` of a single lens, for every pair of layers. Computed
through the Gram identity rather than pairwise, which turns `L^2/2` matmuls of size
`d^3` into `L` of them. The pairwise form took 46 minutes at `d=5376` and ran out of
memory at `d=8192`.

Three tables come out. `cka_out` takes output dimensions as samples and so compares
`J J^T` structure, `cka_in` takes input dimensions and compares `J^T J`, and `cos` is
the plain Frobenius cosine. Pass `--verify N` to check N random pairs per metric
against the exact fp32 pairwise form.

## 3. Sky subspace against the lens

```
python jsky_align.py --only qwen3-32b --lensdir lenses --pcadir pca --outdir jsky
```

One model per process. Reads the lens plus the matching top-128 PCA extraction of the
astro location prompts, fits a 3-dim sky subspace per layer, and asks whether the lens
treats it specially. Records normalized transport gains, spectral rank, a Frobenius
amplification ratio and a sky-to-sky transport fraction, each against 500 random
3-subspaces drawn from the same PCA span. The module docstring carries the index
alignment and the back-map from PCA coordinates to raw residual space; the column
definitions live alongside the output tables.

The `PAIRS` table at the top maps each lens slug to its PCA filename. Extend it to add
a model.
