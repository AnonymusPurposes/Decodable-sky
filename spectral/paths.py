#!/usr/bin/env python
"""Where the shared inputs live.

This folder runs both inside the working tree (inputs under ../fixed_prompts/) and inside the
repository (../PCA128/, ../data/, and the sky_r2 tables copied into r2_tables/ here, because
the repo's own set is missing the two gpt-oss variants). Everything else a script needs --
surface forms, graphs, counts, RDM tables -- sits in this folder.

Neural side: by default (SKY_NEURAL=full) the object centroids in the full standardized residual stream
(rebuttal/data/fullstd, made by rebuttal/full_centroids.py), which Figures 10 and 21-23 of the revised paper
use. SKY_NEURAL=pca128 switches back to the bundled top-128 PCA scores (../PCA128), the footing of the
first version and of sanity_checks.py, which needs per-prompt rows.
"""
import os, glob

NEURAL = os.environ.get("SKY_NEURAL", "full")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _first(what, *cands):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("%s: looked in %s" % (what, ", ".join(cands)))


def catalogue():
    return _first("object catalogue", os.path.join(ROOT, "data_astro_objects.csv"),
                  os.path.join(ROOT, "data", "astro_objects.csv"))


def _neural_dirs():
    if NEURAL == "full":
        return [os.path.join(ROOT, "fixed_prompts", "FULLSTD"), os.path.join(ROOT, "rebuttal", "data", "fullstd")]
    return [os.path.join(ROOT, "fixed_prompts", "PCA128"), os.path.join(ROOT, "PCA128")]


def pca128(model):
    """neural file for model: full-space centroids (default) or top-128 PCA scores, same npz layout"""
    return _first("neural data (%s) for %s" % (NEURAL, model),
                  *[os.path.join(d, model + "_pca128.npz") for d in _neural_dirs()])


def sky_r2(model):
    return _first("sky_r2 table for " + model,
                  os.path.join(ROOT, "fixed_prompts", "correlations", model + "_sky_r2_16.csv"),
                  os.path.join(HERE, "r2_tables", model + "_sky_r2_16.csv"),
                  os.path.join(ROOT, "steering", "rotation", "harmonic", "r2_tables",
                               model + "_sky_r2_16.csv"))


def models():
    """every model with bundled neural data"""
    for d in _neural_dirs():
        f = sorted(glob.glob(os.path.join(d, "*_pca128.npz")))
        if f:
            return [os.path.basename(x)[:-len("_pca128.npz")] for x in f]
    raise FileNotFoundError("no neural data folder (%s) found" % NEURAL)
