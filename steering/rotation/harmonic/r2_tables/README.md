# Per-layer sky decodability tables

`<model>_sky_r2_16.csv`, one row per hidden state: `layer,r2,angerr_deg,angerr_median_deg`.
`r2` is the leave-one-object-out OLS R² of decoding the sky unit vector from the top-16
PCA scores of the location-prompt extraction.

These are reproduced by the repository's own pipeline, from a full (not cropped)
archive:

    python extract_pca.py <model> --prompts data/astro_prompts_location.csv --out PCA128_location
    python correlations.py <model> --regime sky_r2 --npca 16

They are bundled here because `prepare_multi.py` reads them to choose the layers it
patches — every hidden state whose `r2` is at least `--frac` (0.8) of that model's
maximum, minus the final state, which is post-final-norm and is never hooked. Keeping
the tables in the tree means the layer set is fixed by data in the repository rather
than by a table that has to be regenerated on a GPU node first.

The bundled `PCA128/*.npz` are cropped to the top-16 components and therefore hold no
PCA transform (`components`, `feat_mean`, `feat_std`, `pca_mean`). That is enough to
refit the harmonic maps and to build the A-B pairs, but not to run the intervention,
which has to map PCA space back to raw hidden states. For that, regenerate the full
archives with `extract_pca.py`.
