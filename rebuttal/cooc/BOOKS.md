# Sky-guide book corpus

Every public-domain astronomy book we could get without an access agreement, counted with `cooc_stream.py` (book mode) and scored by `book_report.py`. Full per-book table: [books/books_catalog.csv](books/books_catalog.csv) (one row per harvested text, including rejected and duplicate ones, with source URL). Search hits that had no downloadable text are in `books/candidates*.csv`.

## Sources

| source | how candidates were found | candidates | texts obtained |
|---|---|---|---|
| Project Gutenberg | catalogue `pg_catalog.csv`: English texts with LoC class QB (astronomy), or subjects/bookshelves matching astronomy, constellation, stars, telescope, zodiac, planet, celestial, comets, meteor, sky | 819 | 807 |
| Internet Archive | `advancedsearch`: mediatype texts, English, dated 1600–1928 (US public domain), subject astronomy / stars / constellations / telescopes / star atlases / observers' manuals, or a title naming constellations, star atlas, star-gazing, telescope, the heavens, night sky, celestial, star lore, star names, uranography. OCR text `*_djvu.txt` | 5,338 | 4,948 |

Not used: HathiTrust (bulk full text needs a research agreement); post-1928 guides (in copyright, IA only lends them); Wikisource (a handful of texts, nearly all also in IA).

## Scoring and tiers

- **D** is the sky-guide score of Storan et al. 2026 (arXiv:2610.00647, App. G): corr(M\*_ij, cos θ_ij) over objects named ≥ 5 times in the book, with M\* from flat-window counts and additive smoothing α = 0.25 and a one-sided object-permutation p (500 shuffles). Our window is 16 tokens, not their 30.
- **Duplicates** (same work scanned several times, or in both Gutenberg and IA) are found three ways: same title + author key; near-identical object-mention profile (cosine ≥ 0.97, length within ±25–33%); same text (MinHash Jaccard ≥ 0.25 over 5-word shingles). Gutenberg copies are preferred, otherwise the copy with the most mentions. Different editions with heavy revision (Dunkin's *Midnight Sky* 1869 vs 1891) and different volumes (Smyth's *Cycle* vol. 1 vs 2) stay separate.

| tier | rule | books | tokens | pure mentions | gated mentions |
|---|---|---|---|---|---|
| sky_guide | D > 0, p < 0.05, ≥ 15 well-counted objects (the paper's criterion) | 121 | 11.6 M | 97981 | 51593 |
| catalogue | would qualify, but the title says catalogue / almanac / navigation / annals / observations / proceedings (RA-ordered tables) | 29 | 5.1 M | 26334 | 1681 |
| astro_dense | not a sky guide, but ≥ 1 constellation mention per 1k tokens and ≥ 15 distinct constellations | 241 | 17.2 M | 62171 | 29308 |
| other | the rest: general astronomy, physics, biography, fiction or patents matched by title | 3004 | 138.0 M | 66768 | 16638 |
| duplicate | another copy of a book kept elsewhere | 2360 | 275.6 M | 474079 | 227371 |

All four guides used by Storan et al. are present and pass: Olcott *A Field Book of the Stars* (D 0.30), Serviss *Round the Year with the Stars* (0.16), Serviss *Astronomy with an Opera-glass* (0.15), Bonwick *Astronomy for Young Australians* (0.22).

## The 121 sky guides, by D

| # | source | title | author | year | tokens | mentions | D | p |
|---|---|---|---|---|---|---|---|---|
| 1 | ia | Evening Amusements: Or, the Beauty of the Heavens Displayed... | William Frend | 1820 | 39650 | 534 | 0.5151 | 0.002 |
| 2 | ia | Familiar lessons on astronomy ; designed for the use of children and y | Fowler, Lydia F. (Lydia Folger | 1848 | 43879 | 253 | 0.4015 | 0.006 |
| 3 | ia | Elements of astronomy .. | Davis, John, Rev. [from old ca | 1868 | 91534 | 407 | 0.3898 | 0.002 |
| 4 | ia | In starland with a three-inch telescope; a conveniently arranged guide | Olcott, William Tyler, 1873-19 | 1909 | 19228 | 410 | 0.3573 | 0.002 |
| 5 | gutenberg | Half-Hours with the Stars A Plain and Easy Guide to the Knowledge of t | Proctor, Richard A. (Richard A | 2007 | 8569 | 585 | 0.3339 | 0.002 |
| 6 | ia | Spendours of the sky | Lewis, Isabel Martin | 1920 | 64532 | 508 | 0.33 | 0.002 |
| 7 | ia | Star and weather gossip concerning the heavens, the atmosphere, the se | Elgie, Joseph H | 1915 | 49662 | 257 | 0.3296 | 0.002 |
| 8 | ia | How to find the stars, with indications of the most interesting object | Clarke, James Freeman, 1810-18 | 1878 | 9123 | 341 | 0.3091 | 0.002 |
| 9 | ia | A beginner's guide to the stars, 2nd ed. | Maud King Murphy | 1924 | 14591 | 334 | 0.3044 | 0.002 |
| 10 | ia | The Midnight Sky Familiar Notes on the Stars and Planets | Dunkin, Edwin | 1891 | 166880 | 3229 | 0.3001 | 0.002 |
| 11 | gutenberg | A Field Book of the Stars | Olcott, William Tyler, 1873-19 | 2007 | 23347 | 1022 | 0.2987 | 0.002 |
| 12 | ia | The constellations and how to find them; 13 maps, showing the position | Peck, William, Sir, 1862-1925 | 1887 | 21079 | 969 | 0.2928 | 0.002 |
| 13 | ia | Celestial Amusements: or, Belles and Beaux at Hay-Mount | Tyson, Winston C. | 1860 | 23826 | 175 | 0.2926 | 0.004 |
| 14 | gutenberg | The Trouvelot astronomical drawings manual | Trouvelot, E. L. (Etienne Leop | 2022 | 72856 | 279 | 0.2876 | 0.002 |
| 15 | ia | Uranography : a brief description of the constellations visible in the | Young, Charles A. (Charles Aug | 1917 | 14566 | 603 | 0.2855 | 0.002 |
| 16 | ia | Star groups : a student's guide to the constellations | Gore, J. Ellard (John Ellard), | 1891 | 6547 | 331 | 0.2842 | 0.002 |
| 17 | ia | The Phenomena and Diosemeia of Aratus | Aratus, Solensis; Lamb, John,  | 1848 | 29217 | 390 | 0.2824 | 0.002 |
| 18 | ia | Origin of architectural design; or, The archaeology of astronomy | McCoy, Lee Herbert, 1882- | 1912 | 30151 | 287 | 0.2814 | 0.004 |
| 19 | ia | The star pocket-book; or, How to find your way at night by the stars.  | Weatherhead, R. (Robert) | 1911 | 9004 | 296 | 0.2772 | 0.002 |
| 20 | ia | Star-gazer's hand-book; a brief guide for amateur students of astronom | Elson, Henry William, 1857- | 1902 | 14052 | 344 | 0.2719 | 0.002 |
| 21 | gutenberg | The Ways of the Planets | Martin, Martha Evans, -1925 | 2016 | 55113 | 343 | 0.2573 | 0.002 |
| 22 | ia | The mathematics of the sky; | Burton, Charles J. [from old c | 1920 | 21884 | 369 | 0.2548 | 0.006 |
| 23 | gutenberg | The story of the sun, moon, and stars | Giberne, Agnes, 1845-1939; Pri | 2025 | 103309 | 298 | 0.2538 | 0.016 |
| 24 | ia | A Guide To The Constellations | Barton,wm H. | 1928 | 43754 | 1834 | 0.2492 | 0.002 |
| 25 | ia | The gospel in the stars : or, Prímeval astronomy | Seiss, Joseph Augustus, 1823-1 | 1884 | 114341 | 688 | 0.2491 | 0.002 |
| 26 | ia | Observations of comets, from B. C. 611 to A. D. 1640 | Williams, John, 1797-1874 | 1871 | 62409 | 563 | 0.2411 | 0.002 |
| 27 | ia | Hebrew mythology; | Woolley, Milton. [from old cat | 1888 | 239226 | 1871 | 0.2369 | 0.002 |
| 28 | ia | Elements of astronomy; | Peabody, Selim H[obart], 1829- | 1897 | 88901 | 546 | 0.2352 | 0.002 |
| 29 | ia | The midnight sky, notes on the stars and planets | Edwin Dunkin | 1869 | 127512 | 1549 | 0.2347 | 0.002 |
| 30 | ia | The Constellations As Seen In Southern India | R. J. Pocock | 1923 | 34686 | 1001 | 0.2324 | 0.002 |
| 31 | ia | The wonders of the heavens | Flammarion, Camille, 1842-1925 | 1874 | 83041 | 409 | 0.2284 | 0.002 |
| 32 | ia | The globes, celestial and terrestrial | De Morgan, Augustus, 1806-1871 | 1845 | 67213 | 332 | 0.2232 | 0.002 |
| 33 | ia | The sky: spring and summer nights | Brown, Dorothy Louise | 1920 | 10325 | 430 | 0.2222 | 0.002 |
| 34 | gutenberg | Astronomy for Young Australians | Bonwick, James, 1817-1906 | 2018 | 17494 | 247 | 0.2188 | 0.004 |
| 35 | gutenberg | Stars of the southern skies | Orr, M. A. (Mary Acworth), 186 | 2023 | 17321 | 267 | 0.2104 | 0.0459 |
| 36 | ia | An Elementary Astronomy for Academies and Schools: Illustrated by Nume | Hiram Mattison , Mattison's El | 1849 | 84163 | 356 | 0.2094 | 0.004 |
| 37 | ia | The heavens and their story | Maunder, Annie S. D. (Annie Sc | 1910 | 70240 | 264 | 0.2042 | 0.0299 |
| 38 | gutenberg | The Witness of the Stars | Bullinger, E. W. (Ethelbert Wi | 2015 | 52571 | 542 | 0.2033 | 0.002 |
| 39 | ia | Astronomy, in five books | Long, Roger, 1680-1770 | 1742 | 203828 | 213 | 0.2027 | 0.006 |
| 40 | ia | The story of the stars. A descriptive astronomy | Steele, Joel Dorman, 1836-1886 | 1884 | 87856 | 777 | 0.1979 | 0.002 |
| 41 | ia | The beauty of the heavens a pictorial display of the astronomical phen | Charles F. Blunt | 1849 | 61302 | 451 | 0.1953 | 0.002 |
| 42 | ia | Ferguson's astronomy, explained upon Sir Isaac Newton's principles. Wi | Ferguson, James, 1710-1776 | 1811 | 130159 | 603 | 0.1919 | 0.006 |
| 43 | ia | The Chronology of Ancient Nations |  | 1879 | 200586 | 250 | 0.1909 | 0.022 |
| 44 | ia | A new treatise on the use of the globes, or, A philosophical view of t | Keith, Thomas, 1759-1824 | 1832 | 142721 | 973 | 0.1869 | 0.002 |
| 45 | ia | The stars and constellations; how and when to find and tell them | Rosser, W. H. (William Henry) | 1879 | 37112 | 1421 | 0.1862 | 0.002 |
| 46 | ia | Astronomy without a telescope: | Colbert, E[lias] [from old cat | 1869 | 60585 | 1184 | 0.1846 | 0.002 |
| 47 | ia | Astronomy, new and old | Brennan, Martin Stanislaus, 18 | 1889 | 69327 | 625 | 0.1845 | 0.002 |
| 48 | ia | An introduction to astronomy; designed as a text book for the students | Olmsted, Denison, 1791-1859 | 1854 | 122004 | 513 | 0.1843 | 0.002 |
| 49 | ia | The book of the stars for young people, | Olcott, William Tyler, 1873-19 | 1923 | 67972 | 1591 | 0.1825 | 0.002 |
| 50 | ia | Wonders of the earth and the heavens : being a popular account of the  |  | 1800 | 72833 | 289 | 0.1816 | 0.018 |
| 51 | ia | The stars and their stories; a book for young people | Griffiths, Mary Matlock, comp | 1913 | 60381 | 907 | 0.1798 | 0.002 |
| 52 | gutenberg | Astronomy of To-day: A Popular Introduction in Non-Technical Language | Dolmage, Cecil Goodrich Julius | 2009 | 104565 | 372 | 0.1766 | 0.01 |
| 53 | ia | The astronomical observer, a handbook to the observatory and the commo | Darby, William Arthur | 1864 | 83560 | 1484 | 0.1738 | 0.002 |
| 54 | ia | Trees, stars and birds; a book of outdoor science | Moseley, Edwin Lincoln, 1865-1 | 1919 | 94638 | 306 | 0.1717 | 0.024 |
| 55 | ia | Constellations and comets, or, Stories from starland [microform] | Ireland, Lilian B., b. 1877 | 1910 | 18754 | 364 | 0.1709 | 0.002 |
| 56 | ia | The stars in song and legend | Porter, Jermain Gildersleeve,  | 1901 | 24374 | 433 | 0.1701 | 0.004 |
| 57 | ia | Elementary lessons in astronomy | Lockyer, Norman, Sir, 1836-192 | 1888 | 130484 | 523 | 0.169 | 0.002 |
| 58 | ia | A popular handbook and atlas of astronomy, designed as a complete guid | Peck, William, Sir, 1862-1925 | 1891 | 103900 | 1032 | 0.1669 | 0.002 |
| 59 | ia | Elements of astronomy: accompanied with numerous illustrations, a colo | Lockyer, Norman, Sir, 1836-192 | 1875 | 104068 | 548 | 0.1665 | 0.002 |
| 60 | ia | A cycle of celestial objects : for the use of naval, military, and pri | Smyth, W. H. (William Henry),  | 1844 | 253079 | 2759 | 0.1664 | 0.002 |
| 61 | gutenberg | Round the year with the stars The chief beauties of the starry heavens | Serviss, Garrett Putman, 1851- | 2022 | 31350 | 917 | 0.1642 | 0.002 |
| 62 | ia | A compendious treatise on the use of the globes, and of maps : compile | Lathrop, John, 1772-1820; Jame | 1812 | 53616 | 412 | 0.1628 | 0.014 |
| 63 | ia | The amateur telescopist's handbook | Gibson, Frank M. (Frank Markey | 1894 | 48319 | 373 | 0.1612 | 0.0339 |
| 64 | ia | The stars night by night, being the journal of a star gazer. With over | Elgie, Joseph Henry | 1919 | 72004 | 1736 | 0.1574 | 0.002 |
| 65 | ia | Astronomy | Chambers, George F. (George Fr | 1914 | 111979 | 540 | 0.157 | 0.002 |
| 66 | ia | Journal of the British Astronomical Association | British Astronomical Associati | 1890 | 217305 | 330 | 0.1559 | 0.022 |
| 67 | ia | Urania, or, A compleat view of the heavens, containing the antient and | Hill, John, 1714?-1775 | 1754 | 392270 | 1437 | 0.1553 | 0.002 |
| 68 | ia | Guy's Elements of astronomy : and an abridgment of Keith's New treasti | Guy, Joseph, 1784-1867; Keith, | 1864 | 94316 | 497 | 0.1548 | 0.018 |
| 69 | ia | An easy guide to the constellations with a miniature atlas of the star | Gall, James | 1910 | 7534 | 436 | 0.1515 | 0.002 |
| 70 | ia | Astronomy without a telescope; a guide to the constellations, and intr | Maunder, E. Walter (Edward Wal | 1904 | 76437 | 1257 | 0.1506 | 0.002 |
| 71 | gutenberg | Astronomy with an Opera-glass A Popular Introduction to the Study of t | Serviss, Garrett Putman, 1851- | 2011 | 48822 | 956 | 0.1485 | 0.002 |
| 72 | ia | Ptolemy's cataloque of stars; a revision of the Almagest by Christian  | Ptolemy, 2nd cent; Knobel, E.  | 1915 | 105463 | 468 | 0.1447 | 0.002 |
| 73 | ia | Astronomy, as it is known at the present day; with an account of the n | Carey, George G | 1825 | 126348 | 410 | 0.1441 | 0.006 |
| 74 | ia | A high-school astronomy: in which the descriptive, physical, and pract | Mattison, Hiram, 1811-1868 | 1853 | 89256 | 370 | 0.144 | 0.02 |
| 75 | ia | The wonder and the glory of the stars | Forbes, George, 1849-1936 | 1927 | 79593 | 983 | 0.1423 | 0.002 |
| 76 | ia | Practical astronomy with the unaided eye | Macpherson, Hector, b. 1888 | 1912 | 27534 | 733 | 0.1407 | 0.002 |
| 77 | ia | Easy star lessons | Proctor, Richard Anthony, 1837 | 1888 | 41429 | 768 | 0.1403 | 0.002 |
| 78 | ia | Astronomy for all | Bürgel, Bruno Hans, 1875-; Blo | 1911 | 130760 | 574 | 0.1394 | 0.004 |
| 79 | ia | Star lore of all ages; a collection of myths, legends, and facts conce | Olcott, William Tyler, 1873-19 | 1911 | 100447 | 2522 | 0.1386 | 0.002 |
| 80 | ia | A new treatise on astronomy, and the use of globes : in two parts ... | M'Intire, James, 1799-1879 | 1866 | 107480 | 498 | 0.138 | 0.008 |
| 81 | ia | Astronomy with the naked eye; a new geography of the heavens, with des | Serviss, Garrett Putman, 1851- | 1908 | 66287 | 1417 | 0.1374 | 0.002 |
| 82 | ia | Publications of the Astronomical Society of the Pacific | Astronomical Society of the Pa | 1889 | 365285 | 570 | 0.1355 | 0.008 |
| 83 | ia | Familiar astronomy; or, An introduction to the study of the heavens | Peterson, Hannah Mary (Bouvier | 1857 | 219801 | 1989 | 0.1353 | 0.002 |
| 84 | ia | Astronomy Without A Telescope | W Thacker And Co | 1904 | 86323 | 669 | 0.1343 | 0.002 |
| 85 | gutenberg | The Story of the Heavens | Ball, Robert S. (Robert Stawel | 2008 | 201114 | 565 | 0.1325 | 0.0299 |
| 86 | ia | Researches into the origin of the primitive constellations of the Gree | Brown, Robert, 1844- | 1899 | 86298 | 904 | 0.1319 | 0.002 |
| 87 | ia | Researches on the evolution of the stellar systems | See, Thomas Jefferson Jackson, | 1896 | 304556 | 297 | 0.1315 | 0.008 |
| 88 | gutenberg | Pleasures of the telescope An Illustrated Guide for Amateur Astronomer | Serviss, Garrett Putman, 1851- | 2009 | 52598 | 504 | 0.1295 | 0.006 |
| 89 | gutenberg | Astronomy for Young Folks | Lewis, Isabel Martin, 1881-196 | 2014 | 62472 | 1113 | 0.129 | 0.002 |
| 90 | ia | The stars | Chambers, George F. (George Fr | 1909 | 46595 | 627 | 0.1285 | 0.004 |
| 91 | gutenberg | The Heavens Above: A Popular Handbook of Astronomy | Gillet, J. A. (Joseph Anthony) | 2019 | 84176 | 385 | 0.1272 | 0.014 |
| 92 | gutenberg | Half-hours with the Telescope Being a Popular Guide to the Use of the  | Proctor, Richard A. (Richard A | 2005 | 33833 | 241 | 0.1269 | 0.0299 |
| 93 | ia | A new manual of the elements of astronomy, descriptive and mathematica | Kiddle, Henry, 1824-1891 | 1868 | 97051 | 502 | 0.1251 | 0.004 |
| 94 | ia | A beginner's star-book; an easy guide to the stars and to the astronom | Murphy, Edgar Gardner, 1869-19 | 1912 | 96155 | 2206 | 0.1218 | 0.002 |
| 95 | ia | The amateur astronomer | Riegler, Gideon | 1910 | 55847 | 448 | 0.1188 | 0.014 |
| 96 | ia | The friendly stars | Martin, Martha Evans, d. 1925 | 1907 | 50936 | 1684 | 0.1161 | 0.002 |
| 97 | ia | A grammar of astronomy, with problems on the globes: to which are adde | Towle, J | 1825 | 45970 | 638 | 0.1161 | 0.002 |
| 98 | ia | A Handbook of descriptive and practical astronomy | Chambers, George Frederick, 18 | 1889 | 178618 | 1460 | 0.1151 | 0.002 |
| 99 | ia | Elements of astronomy | Rolfe, W. J. (William James),  | 1868 | 105733 | 578 | 0.1136 | 0.002 |
| 100 | ia | Daytime and evening exercises in astronomy, for schools and colleges | Whiting, Sarah Frances, 1847- | 1912 | 28088 | 416 | 0.1132 | 0.022 |
| 101 | ia | Astronomy | Rambosson, Jean Pierre, 1827-1 | 1875 | 113508 | 371 | 0.1096 | 0.0279 |
| 102 | ia | The call of the stars; a popular introduction to a knowledge of the st | Kippax, John R. (John Robert), | 1914 | 127820 | 1964 | 0.1089 | 0.002 |
| 103 | ia | A celestial atlas. A companion to the Celestial atlas | James Middleton | 1842 | 26955 | 735 | 0.106 | 0.004 |
| 104 | ia | The geography of the heavens : and class-book of astronomy : accompani | Burritt, Elijah H. (Elijah Hin | 1860 | 177331 | 1884 | 0.1037 | 0.002 |
| 105 | ia | The complete works of Thomas Dick : containing an essay on the improve | Dick, Thomas, 1774-1857 | 1850 | 739671 | 857 | 0.1026 | 0.006 |
| 106 | gutenberg | The Star People | Johnson, Gaylord | 2011 | 19721 | 402 | 0.0989 | 0.0419 |
| 107 | ia | Practical astronomy | Breen, Hugh | 1860 | 52805 | 553 | 0.0981 | 0.016 |
| 108 | ia | The wonders of the heavens, being a popular view of astronomy, includi | Bradford, Duncan | 1845 | 259648 | 555 | 0.0971 | 0.0479 |
| 109 | ia | The earth and the stars | Abbot, C. G. (Charles Greeley) | 1925 | 98875 | 530 | 0.0958 | 0.008 |
| 110 | ia | Smith's illustrated astronomy : designed for the use of the public or  | Smith, Asa; Burndy Library, do | 1849 | 53683 | 473 | 0.0957 | 0.018 |
| 111 | ia | Astronomy by observation. An elementary text-book for high-schools and | Bowen, Eliza A | 1890 | 77241 | 736 | 0.0947 | 0.004 |
| 112 | ia | A cycle of celestial objects, observed, reduced, and discussed | Smyth, W. H. (William Henry),  | 1881 | 251524 | 1746 | 0.0933 | 0.004 |
| 113 | ia | A study of the sky | Howe, Herbert A. (Herbert Alon | 1896 | 85879 | 776 | 0.0928 | 0.002 |
| 114 | ia | Popular astronomy; a review of astronomy and allied sciences | Payne, William Wallace, 1837-1 | 1894 | 227167 | 944 | 0.0892 | 0.0299 |
| 115 | gutenberg | The music of the spheres | Grondal, Florence Armstrong | 2023 | 91191 | 1560 | 0.0877 | 0.002 |
| 116 | ia | Star names their lore and meaning | Richard Hinckley Allen | 1899 | 263661 | 4673 | 0.0867 | 0.002 |
| 117 | ia | The Fixed Stars And Constellations In Astrology | Robson,vivian E. | 1923 | 57976 | 892 | 0.0863 | 0.002 |
| 118 | ia | Uranography. & Atlas | Kendall, Bzra O. [from old cat | 1850 | 93372 | 1142 | 0.0827 | 0.002 |
| 119 | gutenberg | Astronomical Curiosities: Facts and Fallacies | Gore, J. Ellard (John Ellard), | 2012 | 87200 | 913 | 0.0662 | 0.016 |
| 120 | ia | Elements of descriptive astronomy, a text-book | Howe, Herbert A | 1897 | 107238 | 707 | 0.0622 | 0.0339 |
| 121 | ia | Poole bros' Celestial handbook. Companion to their Celestial planisphe | Colas, Jules A | 1892 | 45162 | 1057 | 0.0514 | 0.002 |

## Catalogue / almanac tier (29), by mentions

| source | title | year | mentions | D |
|---|---|---|---|---|
| ia | A catalogue of 9766 stars in the southern hemisphere, for the beginnin | 1847 | 8988 | 0.3399 |
| ia | Improved nautical almanac, or True navigator, astronomically, mathemat | 1859 | 3446 | 0.1914 |
| ia | Annals of the Astronomical Observatory of Harvard College | 1897 | 3410 | 0.3274 |
| ia | Astronomical Observations made at the East India Company’s Observatory | 1824 | 1586 | 0.1618 |
| ia | A catalogue of the magnitudes of 1081 stars lying between-30⁰ decl. an | 1898 | 957 | 0.2768 |
| ia | Results of astronomical observations made during the years 1834, 5, 6, | 1847 | 728 | -0.0731 |
| ia | A historical view of the Hindu astronomy, from the earliest dawn of sc | 1825 | 717 | 0.0824 |
| ia | Monthly notices of the Royal Astronomical Society | 1831 | 701 | -0.0034 |
| ia | An account of the Revd. John Flamsteed, the first astronomer-royal; co | 1835 | 683 | 0.0274 |
| ia | Catalogue of stars within two degrees of the North pole deduced from p | 1905 | 592 | 0.1902 |
| ia | A comprehensive system of astronomy, both theoretic and practical, wit | 1833 | 581 | 0.1089 |
| ia | Catalogue of 9842 stars, or, All stars very conspicuous to the naked e | 1911 | 415 | -0.0105 |
| ia | The catalogue of stars of the British association for the advancement  | 1845 | 389 | 0.2159 |
| ia | A complete epitome of practical navigation : containing all necessary  | 1864 | 352 | 0.2952 |
| ia | Text-book on navigation and nautical astronomy | 1918 | 327 | 0.1342 |
| ia | A new and complete epitome of practical navigation : containing all ne | 1828 | 324 | 0.3141 |
| ia | An astronomical glossary, or dictionary of terms used in astronomy : w | 1893 | 277 |  |
| ia | Time’s Telescope for 1828; or a Complete Guide to the Almanack: Contai | 1824 | 261 | 0.2369 |
| ia | Bulletin of the Chicago Academy of Sciences, Vol. 4, No. 2 | 1913 | 232 |  |
| ia | The practice of navigation and nautical astronomy | 1882 | 209 | 0.216 |
| ia | A compendious system of astronomy, in a course of familiar lectures; i | 1797 | 206 | 0.3048 |
| ia | The Cape catalogue of stars, deduced from observations made at the Roy | 1878 | 205 |  |
| ia | Southern stellar objects for small telescopes, between the equator and | 1877 | 179 | 0.3469 |
| ia | Certaine errors in navigation detected | 1610 | 168 |  |
| ia | An epitome of astronomy: containing the information requisite for the  | 1853 | 106 |  |
| ia | An elementary introduction to The Nautical almanac, and astronomical e | 1842 | 100 |  |
| ia | Astronomy; or, the true system of the planets demonstrated. Wherein ar | 1727 | 95 |  |
| ia | Manual : pocket edition, containing most of the tables in the large ed | 1898 | 58 |  |
| ia | Speculum coeleste: being an ephemeris of the celestial motions, with a | 1732 | 42 |  |
