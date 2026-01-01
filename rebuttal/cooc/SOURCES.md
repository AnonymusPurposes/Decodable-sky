# Corpus sources for the co-occurrence counts (Q_text, Figures 10/20)

The largest set, `ext_everything` (the "975 B" column of Figures 10/20), is the union below.
Tokens are the counter's word tokens (lower-cased alphanumeric runs, `cooc_stream.py`), not
subword tokens, so they are not comparable with the BPE counts on the dataset cards.

| component | set name | files | word tokens | gated mentions | citation |
|---|---|---|---|---|---|
| C4, `en` variant, train split, all shards | `c4` | 1,024 | 115.7 B | 396,906 | Raffel et al. 2020; Dodge et al. 2021 |
| FineWeb `sample-100BT` | `fineweb` | 150 | 71.4 B | 219,904 | Penedo et al. 2024 |
| FineWeb, main Common Crawl dumps (random files from 110 dumps, CC-MAIN-2013-20 … 2025-26) | `ext_only` | 1,712 | 787.8 B | 2,372,476 | Penedo et al. 2024 |
| English Wikipedia, dump of 2023-11-01 (`wikimedia/wikipedia`, `20231101.en`) | `wiki` | 41 | 2.85 B | 24,149 | Wikimedia Foundation 2023 |
| Public-domain astronomy books: Project Gutenberg (807 texts) and Internet Archive OCR (4,948 texts, dated 1600–1928) | `books_*` | 5,755 | 0.45 B | 326,591 | Project Gutenberg; Internet Archive |
| **total** | `ext_everything` | | **978.2 B** | **3,340,026** | |

`ext_all` (974.9 B) is the same set without Wikipedia and the books.

Notes:
- **Books.** `ext_everything` adds every harvested book text, including the 2,360 texts that
  are second copies of a book kept elsewhere (275.6 M of the 447.5 M book tokens). The
  deduplicated book sets (`books_all_books`, 3,395 texts, 172 M tokens) are separate rows of
  the saturation figure. The duplicates add about 7 % of the gated mentions of
  `ext_everything`.
- **FineWeb overlap.** `sample-100BT` is a random sample of FineWeb, so it can overlap the
  1,712 main-dump files. Those files are ~5 % of FineWeb, so about 5 % of the sample (~4 B
  tokens, 0.4 % of the total) is expected to be counted twice.
- **Versions.** Hugging Face files were streamed from the `main` revision between 6 and 9 Nov
  2026. The exact file lists are [`tasks_web.txt`](tasks_web.txt) (C4, FineWeb sample, and
  Wikipedia) and [`tasks_ext.txt`](tasks_ext.txt) (main-dump FineWeb).
- **Book list.** Every book, with its source URL, tier, sky-guide score D, and duplicate link,
  is in [`books/books_catalog.csv`](books/books_catalog.csv). [`BOOKS.md`](BOOKS.md) describes
  how they were harvested and scored, and lists the 121 sky guides.

## References

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.
- Dodge, J., Sap, M., Marasović, A., Agnew, W., Ilharco, G., Groeneveld, D., Mitchell, M., & Gardner, M. (2021). Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus. In *Proceedings of EMNLP 2021*, 1286–1305. (The `allenai/c4` release used here.)
- Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., & Wolf, T. (2024). The FineWeb datasets: Decanting the web for the finest text data at scale. In *Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track)*. arXiv:2406.17557.
- Wikimedia Foundation. (2023). English Wikipedia dump of 1 November 2023. https://dumps.wikimedia.org, via the Hugging Face dataset `wikimedia/wikipedia` (config `20231101.en`).
- Project Gutenberg. https://www.gutenberg.org (accessed November 2026).
- Internet Archive. https://archive.org (texts collection, OCR `_djvu.txt` files; accessed November 2026).
- Storan et al. (2026). arXiv:2610.00647 (sky-guide criterion D used to tier the books).
