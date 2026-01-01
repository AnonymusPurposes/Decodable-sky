#!/usr/bin/env python
"""Object co-occurrence counts in English Wikipedia (HF wikimedia/wikipedia 20231101.en,
the corpus of Karkada et al. 2026, arXiv:2602.15029), one parquet shard per call.

Tokenization follows the paper (App. A.2): lowercase, drop numerals containing commas or
decimal points, non-alphanumerics -> space, split on whitespace, skip articles with fewer
than 200 tokens. Additions: Greek letters are spelled out ("beta Gruis" from the Greek
letter) and text is NFKD-folded to ASCII ("Bootes" with diaeresis -> "bootes");
out-of-vocabulary words are kept (the paper found removing vs masking them makes no
practical difference).

Mentions: surface forms from graphs/surface_forms.csv, longest match first, within a
paragraph (a text line). A multi-word form occupies ONE position when distances are
measured, so "Large Magellanic Cloud ... Dorado" is measured in words between names.

Pair events: every ordered pair of mentions (first, second in text) at collapsed token
distance 1..16, histogrammed by distance, so any window <= 16 (e.g. 10) is a slice.

Variants (nested: raw > para > art > artnl)
  raw    every mention
  para   mention in an astronomical paragraph: a STRONG cue (constellation, magnitude,
         nebula, telescope, light years, ...) or >= 2 distinct WEAK cues (star, sky,
         orbit, planet, ...) and no astrology cue; `confirm` forms (crater, hydra, phoenix,
         cancer, sculptor, peacock) additionally need their confirm word within +-10 tokens
  art    para AND the article has >= ART_DENS strong cues per 1000 tokens. Removes
         fiction / anime / spaceflight / aircraft articles that use sky vocabulary
         (Saint Seiya gold saints, "Andromeda" TV series, Gemini missions, LEO orbits).
  artnl  art AND the mention is running text rather than a list or table entry. Three
         rules, any of which marks a mention as list context: the article is a "List of
         ..."; the line names >= PARA_LIST_MIN distinct objects at fewer than
         PARA_LIST_RATIO tokens apiece (an enumeration or a table row); >= LIST_MIN
         distinct objects fall within +-LIST_PAD tokens of it (a comma run). All three
         count OBJECTS, not mentions -- "Canopus (Alpha Carinae)" is one object named
         twice, and counting mentions flagged ordinary prose as a list. Dropping the
         mention rather than the pair keeps u / D / Cd consistent for QWEM's e_ij.
  art_list  the older pair-level view of the same thing, kept for comparison: art pair
         events inside a name-dense stretch (>= 4 mentions within [first - 4, second + 4]).

Output part_<shard>.npz: forms, Cd_<variant> (S,S,16) ordered pair counts by distance,
D_<variant> (S,S) articles mentioning both (upper triangle), u_<variant> (S,) mentions,
n_tok_<variant> the tokens those mentions were counted over (QWEM's N),
token/article/paragraph statistics; examples_<shard>.jsonl with reservoir-sampled
contexts per (form, level) where level = rejected | para_only | art_inlist | art_nolist.

Usage: cooc_count.py <parquet> <surface_forms.csv> <out_dir> [max_articles]
"""
import sys, os, re, csv, json, time, bisect, random, unicodedata
import numpy as np

LMAX = 16
MIN_DOC_TOK = 200
CONFIRM_W = 10
ART_DENS = 5.0               # strong cues per 1000 tokens
LIST_PAD, LIST_MIN = 4, 4    # >= LIST_MIN objects within +-LIST_PAD tokens = name-dense stretch
PARA_LIST_MIN = 5            # a line naming this many objects, packed this tightly, is a list
PARA_LIST_RATIO = 8.0        # tokens per named object on that line
N_EX = 6                     # reservoir-sampled contexts per (form, level)
FLUSH = 3_000_000
VARIANTS = ("raw", "para", "art", "artnl")
TITLE_LIST_RE = re.compile(r"\s*(list|lists|index|glossary|outline|table)s? of\b")

STRONG = set("""astronomy astronomical astronomer astronomers astrophysics astrophysical
astrophysicist astrophysicists constellation constellations asterism asterisms magnitude
magnitudes luminosity nebula nebulae nebulas galaxy galaxies galactic extragalactic telescope
telescopes telescopic declination ascension ecliptic celestial parsec parsecs lightyear
lightyears spectral supergiant supergiants hypergiant subgiant stellar interstellar
circumpolar messier ngc hipparcos arcsecond arcseconds arcminute arcminutes arcsec redshift
quasar quasars pulsar pulsars supernova supernovae supernovas exoplanet exoplanets stargazing
stargazers observatory observatories planisphere uranometria almagest parallax photometry
photometric flamsteed""".split())
STRONG_BI = ["light year", "light years", "night sky", "naked eye", "main sequence",
             "white dwarf", "red giant", "red dwarf", "star cluster", "open cluster",
             "globular cluster", "variable star", "double star", "binary star", "milky way"]
WEAK = set("""star stars sky skies brightest faint planet planets orbit orbits orbital sun
solar moon lunar cluster clusters binary dwarf giant horizon hemisphere zodiac""".split())
ASTROL = set("astrology astrological astrologer astrologers astrologically horoscope horoscopes".split())
ASTROL_BI = ["sun sign", "sun signs", "star sign", "star signs", "zodiac sign", "zodiac signs"]
CONFIRM_WORDS = {"constellation": {"constellation", "constellations"}, "star": {"star", "stars"}}

NUM_RE = re.compile(r"\b\d[\d,\.]*[,\.]\d[\d,\.]*\b")
NONAL = re.compile(r"[^a-z0-9]+")


def _greek_table():
    tr = {}
    for cp in range(0x3B1, 0x3CA):
        nm = unicodedata.name(chr(cp), "")
        if nm.startswith("GREEK SMALL LETTER "):
            w = nm[len("GREEK SMALL LETTER "):].lower().replace("final ", "").replace("lamda", "lambda")
            tr[cp] = " " + w + " "
    return tr


GREEK_TR = _greek_table()


def tokenize(line):
    t = line.lower()
    if not t.isascii():
        t = t.translate(GREEK_TR)
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return NONAL.sub(" ", NUM_RE.sub(" ", t)).split()


def para_gate(p):
    """-> "strong" | "weak" (passes) or "" (fails)"""
    s = set(p)
    joined = None
    strong = not STRONG.isdisjoint(s)
    if not strong:
        joined = " " + " ".join(p) + " "
        strong = any(" " + b + " " in joined for b in STRONG_BI if b.split()[0] in s)
    if strong:
        return "strong"
    if len(WEAK & s) < 2 or not ASTROL.isdisjoint(s):
        return ""
    joined = joined or " " + " ".join(p) + " "
    return "" if any(" " + b + " " in joined for b in ASTROL_BI) else "weak"


def load_forms(path):
    forms, confirm, objs = [], [], []
    seen = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["form"] in seen:
                continue
            seen[r["form"]] = len(forms); forms.append(r["form"]); confirm.append(r["confirm"])
            objs.append(r["object"])
    oid = {o: i for i, o in enumerate(sorted(set(objs)))}
    obj_of = [oid[o] for o in objs]          # list rules count objects, not forms: "canopus
                                             # alpha carinae" is one object named twice
    first = {}
    for fid, f in enumerate(forms):
        toks = f.split()
        first.setdefault(toks[0], []).append((toks, fid))
    for k in first:
        first[k].sort(key=lambda x: -len(x[0]))
    return forms, confirm, obj_of, first


def find_mentions(paras, FIRST, FIRST_KEYS):
    """-> [(raw_pos, n_tokens, form_id, paragraph_index)], longest match first"""
    ment = []
    off = 0
    for k, p in enumerate(paras):
        if not FIRST_KEYS.isdisjoint(p):
            end = -1
            for j, w in enumerate(p):
                if j < end or w not in FIRST_KEYS:
                    continue
                for toks, fid in FIRST[w]:
                    L = len(toks)
                    if p[j:j + L] == toks:
                        ment.append((off + j, L, fid, k)); end = j + L
                        break
        off += len(p)
    return ment


def main():
    pq_path, forms_csv, out_dir = sys.argv[1:4]
    max_art = int(sys.argv[4]) if len(sys.argv) > 4 else None
    import pyarrow.parquet as pq
    forms, confirm, OBJ_OF, FIRST = load_forms(forms_csv)
    FIRST_KEYS = set(FIRST)
    S = len(forms)
    tag = os.path.basename(pq_path).split(".")[0]
    os.makedirs(out_dir, exist_ok=True)

    NB = S * S * LMAX
    ckeys = VARIANTS + ("art_list",)
    Cd = {k: np.zeros(NB, np.int64) for k in ckeys}
    D = {k: np.zeros(S * S, np.int64) for k in VARIANTS}
    U = {k: np.zeros(S, np.int64) for k in VARIANTS}
    bufC = {k: [] for k in ckeys}
    bufD = {k: [] for k in VARIANTS}
    bufU = {k: [] for k in VARIANTS}
    st = dict(n_art_all=0, n_art=0, n_tok=0, n_art_mention=0, n_art_dense=0, n_para_mention=0,
              n_para_astro=0, n_para_astro_weak=0, n_confirm_fail=0,
              n_list_title=0, n_list_para=0, n_list_dense=0,
              n_tok_para=0, n_tok_art=0, n_tok_artnl=0,
              **{"n_mention_" + k: 0 for k in VARIANTS})

    def flush():
        for k in ckeys:
            if bufC[k]:
                Cd[k] += np.bincount(np.asarray(bufC[k], np.int64), minlength=NB); bufC[k].clear()
        for k in VARIANTS:
            if bufD[k]:
                D[k] += np.bincount(np.asarray(bufD[k], np.int64), minlength=S * S); bufD[k].clear()
            if bufU[k]:
                U[k] += np.bincount(np.asarray(bufU[k], np.int64), minlength=S); bufU[k].clear()

    rnd = random.Random(tag)
    ex, seen_ex = {}, {}

    def add_example(fid, level, info, allt, pos, L):
        key = (fid, level)
        n = seen_ex[key] = seen_ex.get(key, 0) + 1
        lst = ex.setdefault(key, [])
        r = len(lst) if len(lst) < N_EX else rnd.randrange(n)
        if r < N_EX:
            ctx = dict(info, text=" ".join(allt[max(0, pos - 12): pos]) + " [" + " ".join(allt[pos:pos + L])
                       + "] " + " ".join(allt[pos + L: pos + L + 12]))
            if r == len(lst):
                lst.append(ctx)
            else:
                lst[r] = ctx

    t0 = time.time()
    pf = pq.ParquetFile(pq_path)
    done = False
    for batch in pf.iter_batches(batch_size=512, columns=["title", "text"]):
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            st["n_art_all"] += 1
            if max_art and st["n_art_all"] > max_art:
                done = True; break
            if not text:
                continue
            paras = [p for p in (tokenize(l) for l in text.split("\n")) if p]
            ntok = sum(len(p) for p in paras)
            if ntok < MIN_DOC_TOK:
                continue
            st["n_art"] += 1; st["n_tok"] += ntok
            # tokens a mention could have fallen in, per variant: the denominator N of QWEM's
            # e_ij = 2 S_f u_i u_j / N has to be the tokens the gated u_i were counted over
            pg = [para_gate(p) for p in paras]
            dens = 1000.0 * sum(sum(w in STRONG for w in p) for p in paras) / ntok
            dense = dens >= ART_DENS
            gated_tok = sum(len(p) for k, p in enumerate(paras) if pg[k])
            st["n_tok_para"] += gated_tok
            has_form = not all(FIRST_KEYS.isdisjoint(p) for p in paras)
            ment = find_mentions(paras, FIRST, FIRST_KEYS) if has_form else []
            if dense:
                st["n_tok_art"] += gated_tok
                if not ment:
                    st["n_tok_artnl"] += gated_tok
            if not ment:
                continue
            st["n_art_mention"] += 1
            allt = [w for p in paras for w in p]
            st["n_art_dense"] += dense
            route = {k: pg[k] for k in sorted({m[3] for m in ment})}
            st["n_para_mention"] += len(route); st["n_para_astro"] += sum(bool(r) for r in route.values())
            st["n_para_astro_weak"] += sum(r == "weak" for r in route.values())

            cp, cum = [], 0                     # collapsed positions: a multi-word form = one slot
            for pos, L, fid, k in ment:
                cp.append(pos - cum); cum += L - 1
            oid = [OBJ_OF[m[2]] for m in ment]
            in_para = {}
            for a, (_, _, _, k) in enumerate(ment):
                in_para.setdefault(k, set()).add(oid[a])
            title_list = bool(TITLE_LIST_RE.match(title.lower())) if title else False
            para_list = {k: len(o) >= PARA_LIST_MIN and len(paras[k]) / len(o) < PARA_LIST_RATIO
                         for k, o in in_para.items()}
            if dense:
                st["n_tok_artnl"] += 0 if title_list else sum(
                    len(p) for k, p in enumerate(paras) if pg[k] and not para_list.get(k, False))

            fids, lev = [], []                  # level 0 raw / 1 para / 2 art / 3 artnl
            for a, (pos, L, fid, k) in enumerate(ment):
                ok = bool(route[k])
                if ok and confirm[fid]:
                    words = CONFIRM_WORDS[confirm[fid]]
                    if not any(w in words for w in allt[max(0, pos - CONFIRM_W): pos + L + CONFIRM_W]):
                        ok = False; st["n_confirm_fail"] += 1
                if ok and dense:
                    lo = bisect.bisect_left(cp, cp[a] - LIST_PAD)
                    near = len(set(oid[lo:bisect.bisect_right(cp, cp[a] + LIST_PAD)]))
                    st["n_list_title"] += title_list
                    st["n_list_para"] += (not title_list) and para_list[k]
                    st["n_list_dense"] += not (title_list or para_list[k]) and near >= LIST_MIN
                    lv = 2 if (title_list or para_list[k] or near >= LIST_MIN) else 3
                else:
                    lv = 1 if ok else 0
                add_example(fid, ("rejected", "para_only", "art_inlist", "art_nolist")[lv],
                            dict(route=route[k], dens=round(dens, 1), title=title), allt, pos, L)
                fids.append(fid); lev.append(lv)
                for v in range(lv + 1):
                    bufU[VARIANTS[v]].append(fid)
            m = len(cp)
            for v in range(len(VARIANTS)):
                st["n_mention_" + VARIANTS[v]] += sum(l >= v for l in lev)
            for a in range(m):
                ca, fa, la = cp[a], fids[a], lev[a]
                for b in range(a + 1, m):
                    d = cp[b] - ca
                    if d > LMAX:
                        break
                    i = (fa * S + fids[b]) * LMAX + d - 1
                    lab = min(la, lev[b])
                    bufC["raw"].append(i)
                    if lab >= 1:
                        bufC["para"].append(i)
                    if lab >= 2:
                        bufC["art"].append(i)
                        lo = bisect.bisect_left(cp, ca - LIST_PAD); hi = bisect.bisect_right(cp, cp[b] + LIST_PAD)
                        if hi - lo >= LIST_MIN:
                            bufC["art_list"].append(i)
                    if lab >= 3:
                        bufC["artnl"].append(i)
            for v in range(len(VARIANTS)):
                sv = sorted({f for f, l in zip(fids, lev) if l >= v})
                for x in range(len(sv)):
                    for y in range(x + 1, len(sv)):
                        bufD[VARIANTS[v]].append(sv[x] * S + sv[y])
            if len(bufC["raw"]) > FLUSH or len(bufD["raw"]) > FLUSH:
                flush()
        if done:
            break
    flush()

    out = os.path.join(out_dir, "part_%s.npz" % tag)
    np.savez_compressed(out + ".tmp.npz", forms=np.array(forms),
                        **{"Cd_" + k: v.reshape(S, S, LMAX) for k, v in Cd.items()},
                        **{"D_" + k: v.reshape(S, S) for k, v in D.items()},
                        **{"u_" + k: v for k, v in U.items()},
                        **{k: np.int64(v) for k, v in st.items()},
                        LMAX=np.int64(LMAX), ART_DENS=np.float64(ART_DENS), src=np.array(os.path.basename(pq_path)))
    os.replace(out + ".tmp.npz", out)
    with open(os.path.join(out_dir, "examples_%s.jsonl" % tag), "w", encoding="utf-8") as fh:
        for (fid, level), ctx in sorted(ex.items(), key=lambda kv: kv[0]):
            fh.write(json.dumps(dict(form=forms[fid], level=level, n=seen_ex[(fid, level)], contexts=ctx)) + "\n")
    print("%s: %s  %.1fs" % (tag, json.dumps(st), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
