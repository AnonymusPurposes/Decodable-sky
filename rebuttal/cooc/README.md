# Co-occurrence counts and Q_text (Appendix F.3)

Counts of catalogue-object names near each other in about 1T words of web, encyclopedic and book text,
and the co-occurrence embedding Q_text built from them. Corpora, token counts and citations:
[SOURCES.md](SOURCES.md); the book harvest: [BOOKS.md](BOOKS.md) and `books/books_catalog.csv`.

## Pipeline

1. **Counting** (`cooc_stream.py`, CPU only). One task per corpus file (`tasks_web.txt`: Wikipedia, C4,
   FineWeb sample; `tasks_ext.txt`: FineWeb main dumps) or per book (`harvest_books.py`, `run_books.sh`);
   Slurm drivers `run_pilot.sh`, `run_ext.sbatch`. Text is lower-cased and normalized (æ/œ folding,
   de-hyphenation, Gutenberg headers stripped); 367 surface forms (`surface_forms_v2.csv`, including
   historical spellings and English names, the latter only inside astronomical paragraphs). For each pair
   of forms the counter records mentions at word distance d = 1..16. Two counts are kept: pure (every
   mention) and gated (astronomical paragraph in an astronomy-dense document, confirm word for polysemous
   names); Q_text uses the gated one.
2. **Merging** into corpus stages: `merge_sets.py` (Wikipedia, the 187 B pilot web sample and its 1/16 … 1/2
   prefixes, books), `merge_ext.py` (the pilot web plus growing prefixes of the FineWeb main dumps, up to
   975 B; `ext_everything` adds Wikipedia and the books), `merge_stage_halves.py` (even / odd count files of
   each stage, for split-half reliability). The merged sets are shipped in `../data/cooc/`.
3. **Q_text** (`qtext_spectrum.py`, imported by the scripts below). Counts c_ij = Σ_d (17 − d)
   [C_ij(d) + C_ji(d)] over the objects mentioned at least 10 times; M* = 2(c − e)/(c + e) with either the
   corpus-wide independence expectation of the first version (e = 2·136·u_i u_j / N, zero diagonal) or the
   expectation from the object-object table itself (e_ij = R_i R_j / T, R_i = Σ_k c_ik, T = Σ_k R_k,
   observed diagonal), which does not dilute as the corpus grows. Embedding from the positive eigenvalues
   (Q⁺) or the signed kernel (Q^±), all modes or the top 10.
   - `qtext_saturation.py`: fit to the neural RDM and split-half reliability at every corpus stage
     (Figure 24; `../results/qtext_saturation.csv`, books alone `../results/qtext_books.csv`).
   - `../figures/qtext_variant_rdms.py`: RDM tables of every construction for the scoring of Figure 10
     (`../spectral_intervals.py`; `../results/qtext_variant_intervals.csv`).
   - `qtext_negeig.py`: negative eigenvalues of M* (`../results/qtext_negeig.csv`).
   - `qtext_split_half.py`: split-half reliability of the RDMs (`../results/qtext_split_half.csv`).

## Why the table baseline

With the corpus-wide expectation, M* saturates as text is added: for two constellation names in 93 B
tokens e ≈ 0.1 while they co-occur ~100 times, so most pairs sit near M* = +2 and the fit to the neural
RDM falls as the corpus grows. With the table baseline the fit rises and levels off from about 100 B
tokens, and the split-half reliability of the RDM reaches 0.95 at 975 B (Figure 24).
