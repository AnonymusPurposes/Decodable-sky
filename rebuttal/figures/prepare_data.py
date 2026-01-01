#!/usr/bin/env python
"""Collect the inputs of figures.wl into rebuttal/figures/data/ (most folders; see below).

Run on the authors' working tree (paths relative to --connected); kept for the record. The committed
data/ folder is what figures.wl reads, so the figures can be redrawn without it. Folders copied here:

  split/<m>_split_r2_<k>.csv     per layer, median and 5-95% over 100 80/20 splits, in-split PCA
                                 (split_layer_csv.py on results/split_pca, k=4,8,16; gpt-oss also
                                 k=32 from results/split_pca_k32)
  null/k<k>/<m>_angle_null.csv   permutation null of the angle on the real activations (angle_null.py)
  strips/<m>_*_128.csv           the coloured PCA strips of Figures 3/12 (correlations.py)
  ablation/                      ablation summary (all_models_summary.csv), object-cluster intervals and the
                                 answer-shuffling baseline (intervals.py)
  steering/{paper,pair}_{l12,l1}.csv   all_models_summary_rich.csv of the first version's fit and of the
                                 fit refitted without A and B (templates 0 and 18)
  steering/steering_templates.csv      the refitted map per evaluation template (steering_templates.py)
  rdm/model_fits.csv, rdm/spectral_intervals.csv, rdm/qtext_variant_intervals.csv   Figures 10, 21-23
  rdm/qtext_saturation.csv       Q_text saturation with corpus size (cooc/qtext_saturation.py)

Added directly, not by this script: vs/<m>_linmap_<chart|cylinder>_nopca.csv (Figures 8, 20; from
results/kdim/full, metric_fullpca_allscore.py) and noloc/<m>_split_r2_<k>.csv (Figure 15; split_pca.py on
the generic-prompt activations, run_noloc.sbatch).

Usage: prepare_data.py --connected <AI/connected> --out <rebuttal/figures/data>
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["mistrallarge123b", "glm45air", "qwen235b", "qwen32b", "llama33_70b", "mixtral8x22b",
          "gptoss120b", "gptoss120b_harmony"]


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connected", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C, O = a.connected, a.out
    res = os.path.join(HERE, "..", "results")
    pull = os.path.join(C, "rebuttal_pull")

    # layer charts
    subprocess.run([sys.executable, os.path.join(HERE, "..", "split_layer_csv.py"),
                    "--split-dir", os.path.join(pull, "split_pca"), "--out", os.path.join(O, "split")], check=True)
    if glob.glob(os.path.join(pull, "split_pca_k32", "*.npz")):
        tmp = os.path.join(pull, "split_layers_k32")
        subprocess.run([sys.executable, os.path.join(HERE, "..", "split_layer_csv.py"),
                        "--split-dir", os.path.join(pull, "split_pca_k32"), "--out", tmp], check=True)
        for f in glob.glob(os.path.join(tmp, "*_split_r2_32.csv")):
            cp(f, os.path.join(O, "split", os.path.basename(f)))
    for k, src in (("8", os.path.join(pull, "angle_null")), ("16", os.path.join(pull, "angle_null_k16")),
                   ("32", os.path.join(pull, "angle_null_k32"))):
        for f in glob.glob(os.path.join(src, "*_angle_null.csv")):
            cp(f, os.path.join(O, "null", "k" + k, os.path.basename(f)))
    for m in MODELS:
        for s in ("sky_cov_scaled", "sky_cov_equal", "type_cov_equal"):
            f = os.path.join(C, "fixed_prompts", "correlations", "%s_%s_128.csv" % (m, s))
            if os.path.exists(f):
                cp(f, os.path.join(O, "strips", os.path.basename(f)))

    # ablation
    cp(os.path.join(C, "ablation", "results_harm_frac80", "all_models_summary.csv"), os.path.join(O, "ablation", "all_models_summary.csv"))
    for f in ("ablation_intervals.csv", "baseline_angle.csv"):
        cp(os.path.join(res, f), os.path.join(O, "ablation", f))

    # steering
    st = os.path.join(C, "steering", "multimodel_heldout_typed_20260918")
    cp(os.path.join(st, "results_fit_heldout_typed_rich_v2", "all_models_summary_rich.csv"), os.path.join(O, "steering", "paper_l12.csv"))
    cp(os.path.join(st, "results_fit_heldout_typed_l1_rich_v2", "all_models_summary_rich.csv"), os.path.join(O, "steering", "paper_l1.csv"))
    cp(os.path.join(pull, "steering_pair", "results_rich", "all_models_summary_rich.csv"), os.path.join(O, "steering", "pair_l12.csv"))
    cp(os.path.join(pull, "steering_pair", "results_l1_rich", "all_models_summary_rich.csv"), os.path.join(O, "steering", "pair_l1.csv"))
    cp(os.path.join(res, "steering_templates.csv"), os.path.join(O, "steering", "steering_templates.csv"))

    # model RDMs
    cp(os.path.join(C, "spectral", "fits", "model_fits.csv"), os.path.join(O, "rdm", "model_fits.csv"))
    cp(os.path.join(res, "spectral_intervals.csv"), os.path.join(O, "rdm", "spectral_intervals.csv"))
    q = os.path.join(pull, "qtext_variant_intervals.csv")
    if os.path.exists(q):
        cp(q, os.path.join(O, "rdm", "qtext_variant_intervals.csv"))
    cp(os.path.join(res, "qtext_saturation.csv"), os.path.join(O, "rdm", "qtext_saturation.csv"))   # Q_text saturation
    print("data ->", O)


if __name__ == "__main__":
    main()
