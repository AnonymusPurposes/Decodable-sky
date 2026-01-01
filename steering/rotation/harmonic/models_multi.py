"""Checkpoint registry for the multi-model harmonic steering battery.

Repositories match `extract_pca.py` in the Decodable-sky pipeline, so each
model is patched with the same checkpoint that produced its PCA archive. Note
that `qwen235b` is the THINKING 2507 FP8 variant there, not Instruct.

`revision` is the commit the results were produced against, the repository's main
HEAD at access time, and is the value reported in the paper's model table. It is
passed to `from_pretrained` and recorded in each run manifest, so a rerun loads the
same weights even after the upstream repository moves. Leaving it unset would silently
follow `main` and stop matching the published numbers.

`device_map`:
  single  one GPU, `{'': 0}`. Mandatory for the FP8 checkpoint: a single process
          driving FP8 weights across two devices silently corrupts logits.
  auto    sharded, for checkpoints too large for one 275 GB B300.
          Mixtral-8x22B is 141B parameters in BF16 (~282 GB).

`keep_experts_impl` disables the transformers 5.x decode-time MoE swap from
`grouped_mm` to `batched_mm`, which copies selected expert weights per token and
exhausts memory at large batch.
"""

MODELS = {
    'qwen32b': dict(
        repo='Qwen/Qwen3-32B',
        revision='9216db5781bf21249d130ec9da846c4624c16137',
        device_map='single', keep_experts_impl=False),
    # FP8: transformers resolves the finegrained-fp8 Hub kernel BY VERSION, and the
    # local cache has no refs/, so that lookup needs one online call. Weights still
    # load from cache (local_files_only=True).
    'qwen235b': dict(
        repo='Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
        revision='f07f63f2bbd7540917118ebdf3812696ef303b03',
        device_map='single', keep_experts_impl=True, online_kernels=True),
    'llama33_70b': dict(
        repo='unsloth/Llama-3.3-70B-Instruct',
        revision='99cd0d2c829e92a67c844f9144c2509632e5c87f',
        device_map='single', keep_experts_impl=False),
    'mixtral8x22b': dict(
        repo='mistralai/Mixtral-8x22B-Instruct-v0.1',
        revision='cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d',
        device_map='auto', keep_experts_impl=True),
    'mistrallarge123b': dict(
        repo='mistralai/Mistral-Large-Instruct-2411',
        revision='ba78820945ae22361b0274cf0ae6d696c967c1a4',
        device_map='single', keep_experts_impl=False),
    'glm45air': dict(
        repo='zai-org/GLM-4.5-Air',
        revision='a24ceef6ce4f3536971efe9b778bdaa1bab18daa',
        device_map='single', keep_experts_impl=True),
}

# The FIXED location-prompt regime: `astro_prompts_location_fixed.csv`, extracted
# into PCA128_location_fixed/ by run_fixed.sh. This is the regime the repository
# ships -- its data/astro_prompts_location.csv is byte-identical to the fixed
# prompt file, and its cropped PCA128/*.npz are bit-identical to these archives.
# The r2 tables in r2_tables/ are the same regime, so layer selection, activations
# and prompts now all agree. PCA128_location/ holds the older, distinct prompt set
# and is deliberately not used.
#
# Paths resolve relative to this file, with an environment override each. ARCHIVE_DIR
# must hold FULL extract_pca.py archives: the intervention maps PCA space back to raw
# hidden states and needs components, feat_mean, feat_std and pca_mean, which the
# cropped top-16 files bundled in PCA128/ do not carry.
#   PCA_DIR   <model>_pca128.npz          default ../../../pca
#   R2_DIR    <model>_sky_r2_16.csv       default ./r2_tables
#   PROMPTS   prompt templates            default ../../../data/astro_prompts_location.csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

ARCHIVE_DIR = os.environ.get('PCA_DIR', os.path.join(_ROOT, 'pca'))
R2_DIR = os.environ.get('R2_DIR', os.path.join(_HERE, 'r2_tables'))
PROMPTS = os.environ.get(
    'PROMPTS', os.path.join(_ROOT, 'data', 'astro_prompts_location.csv'))
REGIME = 'location_fixed'


def archive(model, root=ARCHIVE_DIR):
    return os.path.join(root, f'{model}_pca128.npz')


def r2_table(model, root=R2_DIR):
    return os.path.join(root, f'{model}_sky_r2_16.csv')
