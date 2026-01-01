#!/usr/bin/env python
"""Streaming co-occurrence counter (pilot v2 of spectral/cooc_count.py).

Same counting, gates and output format as spectral/cooc_count.py (nested variants
raw > para > art > artnl, plus art_list; Cd_* by distance 1..16, D_*, u_*, n_tok_*), so
merge_cooc.py / build_qwem_rdm.py read the parts unchanged. Two of the variants are the
headline counts:
  raw  "pure": every mention, no context test (what plain QWEM assumes)
  art  "gated": astronomical paragraph in an astronomy-dense document, confirm word for the
       polysemous names; lists KEPT (dropping them, artnl, fitted worse on Wikipedia)
Proximity-word weighting (v3 "prox") is not repeated: it moved r(cooc, sphere) by <= 0.007.

What changed vs cooc_count.py (text normalization, all corpora):
  * ligatures/letters NFKD does not fold: ae/oe digraphs (Cassiopeiae, Ursae), sharp s,
    o-slash, ... (before: "cassiopei" + dropped letter -> Bayer forms missed); soft hyphens
    and zero-width characters removed; long s folds through NFKD. Boötes -> bootes as before.
  * words hyphenated across a line break are rejoined ("Androm-\\neda" -> andromeda).
  * book mode (Gutenberg / OCR): hard-wrapped lines are joined into paragraphs (blank-line
    separated), so "Ursa\\nMajor" is one mention and the paragraph gate sees a paragraph,
    not a 70-character line; very long blocks (OCR without blank lines) are cut into
    ~300-token paragraphs; Gutenberg licence header/footer stripped.
  * more surface forms (surface_forms_v2.csv): historical spellings (Wega, Betelgeux,
    Benetnasch, Camelopardus, Serpentarius, Scorpio, ...) count everywhere; English names
    (Great Bear, Big Dipper, Northern Cross, Dog Star, ...) carry confirm="astro" and count
    ONLY inside an astronomical paragraph -- they are dropped from raw as well, so the pure
    count is not polluted by Great Bear Lake or roller coasters.

Sources:
  parquet  <path or https URL>  (URL is downloaded to $COOC_TMP and deleted after)
  jsonlgz  <https URL>          streamed through gzip (C4)
  books    <list.txt>           one book file per line; ONE part per book (per-book stats)

Usage: cooc_stream.py <forms_csv> <out_dir> <kind> <src> [tag]
"""
import sys, os, re, csv, json, time, bisect, random, unicodedata, gzip, io, shutil, subprocess
import numpy as np

LMAX = 16
MIN_DOC_TOK = 200
CONFIRM_W = 10
ART_DENS = 5.0
LIST_PAD, LIST_MIN = 4, 4
PARA_LIST_MIN = 5
PARA_LIST_RATIO = 8.0
BOOK_PARA_MAX = 300
N_EX = 4
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
DEHYPH = re.compile(r"([A-Za-z])-[ \t]*\r?\n[ \t]*([a-z])")
BLANK = re.compile(r"\n[ \t]*\n")
GUT_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
GUT_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def _fold_table():
    tr = {}
    for cp in range(0x3B1, 0x3CA):
        nm = unicodedata.name(chr(cp), "")
        if nm.startswith("GREEK SMALL LETTER "):
            w = nm[len("GREEK SMALL LETTER "):].lower().replace("final ", "").replace("lamda", "lambda")
            tr[cp] = " " + w + " "
    for a, b in {"æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "þ": "th",
                 "ð": "d", "ı": "i", "­": "", "​": "", "‌": "", "‍": "",
                 "﻿": ""}.items():
        tr[ord(a)] = b
    return tr


FOLD = _fold_table()


def tokenize(line):
    t = line.lower()
    if not t.isascii():
        t = t.translate(FOLD)
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return NONAL.sub(" ", NUM_RE.sub(" ", t)).split()


def paragraphs(text, book):
    if "-" in text:
        text = DEHYPH.sub(r"\1\2", text)
    if not book:
        return [p for p in (tokenize(l) for l in text.split("\n")) if p]
    out = []
    for blk in BLANK.split(text.replace("\r\n", "\n")):
        p = tokenize(blk.replace("\n", " "))
        for s in range(0, len(p), BOOK_PARA_MAX):
            if p[s:s + BOOK_PARA_MAX]:
                out.append(p[s:s + BOOK_PARA_MAX])
    return out


def strip_gutenberg(text):
    m = GUT_START.search(text)
    if m:
        text = text[m.end():]
    m = GUT_END.search(text)
    if m:
        text = text[:m.start()]
    return text


def para_gate(p):
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
    obj_of = [oid[o] for o in objs]
    first = {}
    for fid, f in enumerate(forms):
        toks = f.split()
        first.setdefault(toks[0], []).append((toks, fid))
    for k in first:
        first[k].sort(key=lambda x: -len(x[0]))
    return forms, confirm, obj_of, first


def find_mentions(paras, FIRST, FIRST_KEYS):
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


class Counter:
    def __init__(self, forms_csv):
        self.forms, self.confirm, self.OBJ_OF, self.FIRST = load_forms(forms_csv)
        self.FIRST_KEYS = set(self.FIRST)
        self.S = len(self.forms)
        self.reset()

    def reset(self):
        S = self.S
        self.NB = S * S * LMAX
        self.ckeys = VARIANTS + ("art_list",)
        self.Cd = {k: np.zeros(self.NB, np.int64) for k in self.ckeys}
        self.D = {k: np.zeros(S * S, np.int64) for k in VARIANTS}
        self.U = {k: np.zeros(S, np.int64) for k in VARIANTS}
        self.bufC = {k: [] for k in self.ckeys}
        self.bufD = {k: [] for k in VARIANTS}
        self.bufU = {k: [] for k in VARIANTS}
        self.st = dict(n_art_all=0, n_art=0, n_tok=0, n_art_mention=0, n_art_dense=0,
                       n_para_mention=0, n_para_astro=0, n_para_astro_weak=0, n_confirm_fail=0,
                       n_astro_only_dropped=0, n_list_title=0, n_list_para=0, n_list_dense=0,
                       n_tok_para=0, n_tok_art=0, n_tok_artnl=0,
                       **{"n_mention_" + k: 0 for k in VARIANTS})
        self.ex, self.seen_ex = {}, {}
        self.t0 = time.time()

    def flush(self):
        S = self.S
        for k in self.ckeys:
            if self.bufC[k]:
                self.Cd[k] += np.bincount(np.asarray(self.bufC[k], np.int64), minlength=self.NB); self.bufC[k].clear()
        for k in VARIANTS:
            if self.bufD[k]:
                self.D[k] += np.bincount(np.asarray(self.bufD[k], np.int64), minlength=S * S); self.bufD[k].clear()
            if self.bufU[k]:
                self.U[k] += np.bincount(np.asarray(self.bufU[k], np.int64), minlength=S); self.bufU[k].clear()

    def add_example(self, fid, level, info, allt, pos, L):
        key = (fid, level)
        n = self.seen_ex[key] = self.seen_ex.get(key, 0) + 1
        lst = self.ex.setdefault(key, [])
        r = len(lst) if len(lst) < N_EX else self.rnd.randrange(n)
        if r < N_EX:
            ctx = dict(info, text=" ".join(allt[max(0, pos - 12): pos]) + " [" + " ".join(allt[pos:pos + L])
                       + "] " + " ".join(allt[pos + L: pos + L + 12]))
            if r == len(lst):
                lst.append(ctx)
            else:
                lst[r] = ctx

    def doc(self, title, text, book=False):
        st, S = self.st, self.S
        st["n_art_all"] += 1
        if not text:
            return
        paras = paragraphs(text, book)
        ntok = sum(len(p) for p in paras)
        if ntok < MIN_DOC_TOK:
            return
        st["n_art"] += 1; st["n_tok"] += ntok
        pg = [para_gate(p) for p in paras]
        dens = 1000.0 * sum(sum(w in STRONG for w in p) for p in paras) / ntok
        dense = dens >= ART_DENS
        gated_tok = sum(len(p) for k, p in enumerate(paras) if pg[k])
        st["n_tok_para"] += gated_tok
        has_form = not all(self.FIRST_KEYS.isdisjoint(p) for p in paras)
        ment = find_mentions(paras, self.FIRST, self.FIRST_KEYS) if has_form else []
        if ment:                                   # English names exist only in astro paragraphs
            keep = [m for m in ment if not (self.confirm[m[2]] == "astro" and not pg[m[3]])]
            st["n_astro_only_dropped"] += len(ment) - len(keep)
            ment = keep
        if dense:
            st["n_tok_art"] += gated_tok
            if not ment:
                st["n_tok_artnl"] += gated_tok
        if not ment:
            return
        st["n_art_mention"] += 1
        allt = [w for p in paras for w in p]
        st["n_art_dense"] += dense
        route = {k: pg[k] for k in sorted({m[3] for m in ment})}
        st["n_para_mention"] += len(route); st["n_para_astro"] += sum(bool(r) for r in route.values())
        st["n_para_astro_weak"] += sum(r == "weak" for r in route.values())
        cp, cum = [], 0
        for pos, L, fid, k in ment:
            cp.append(pos - cum); cum += L - 1
        oid = [self.OBJ_OF[m[2]] for m in ment]
        in_para = {}
        for a, (_, _, _, k) in enumerate(ment):
            in_para.setdefault(k, set()).add(oid[a])
        title_list = bool(TITLE_LIST_RE.match(title.lower())) if title else False
        para_list = {k: len(o) >= PARA_LIST_MIN and len(paras[k]) / len(o) < PARA_LIST_RATIO
                     for k, o in in_para.items()}
        if dense:
            st["n_tok_artnl"] += 0 if title_list else sum(
                len(p) for k, p in enumerate(paras) if pg[k] and not para_list.get(k, False))
        fids, lev = [], []
        for a, (pos, L, fid, k) in enumerate(ment):
            ok = bool(route[k])
            cf = self.confirm[fid]
            if ok and cf in CONFIRM_WORDS:
                words = CONFIRM_WORDS[cf]
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
            self.add_example(fid, ("rejected", "para_only", "art_inlist", "art_nolist")[lv],
                             dict(route=route[k], dens=round(dens, 1), title=(title or "")[:80]), allt, pos, L)
            fids.append(fid); lev.append(lv)
            for v in range(lv + 1):
                self.bufU[VARIANTS[v]].append(fid)
        m = len(cp)
        for v in range(len(VARIANTS)):
            st["n_mention_" + VARIANTS[v]] += sum(l >= v for l in lev)
        bufC = self.bufC
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
                    self.bufD[VARIANTS[v]].append(sv[x] * S + sv[y])
        if len(bufC["raw"]) > FLUSH or len(self.bufD["raw"]) > FLUSH:
            self.flush()

    def save(self, out_dir, tag, src):
        self.flush()
        S = self.S
        out = os.path.join(out_dir, "part_%s.npz" % tag)
        np.savez_compressed(out + ".tmp.npz", forms=np.array(self.forms),
                            **{"Cd_" + k: v.reshape(S, S, LMAX) for k, v in self.Cd.items()},
                            **{"D_" + k: v.reshape(S, S) for k, v in self.D.items()},
                            **{"u_" + k: v for k, v in self.U.items()},
                            **{k: np.int64(v) for k, v in self.st.items()},
                            LMAX=np.int64(LMAX), ART_DENS=np.float64(ART_DENS), src=np.array(src))
        os.replace(out + ".tmp.npz", out)
        with open(os.path.join(out_dir, "examples_%s.jsonl" % tag), "w", encoding="utf-8") as fh:
            for (fid, level), ctx in sorted(self.ex.items(), key=lambda kv: kv[0]):
                fh.write(json.dumps(dict(form=self.forms[fid], level=level, n=self.seen_ex[(fid, level)],
                                         contexts=ctx)) + "\n")
        print("%s: %s  %.1fs" % (tag, json.dumps(self.st), time.time() - self.t0), flush=True)


def fetch(url, dst, tries=5):
    for t in range(tries):
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", dst, url])
        if r.returncode == 0 and os.path.getsize(dst) > 0:
            return
        time.sleep(5 + 10 * t)
    raise RuntimeError("download failed: " + url)


def iter_parquet(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in ("title", "text") if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        tt = batch.column("title").to_pylist() if "title" in cols else [None] * batch.num_rows
        for title, text in zip(tt, batch.column("text").to_pylist()):
            yield title, text


def iter_jsonlgz(url):
    for t in range(5):
        n = 0
        try:
            p = subprocess.Popen(["curl", "-sfL", "--retry", "3", url], stdout=subprocess.PIPE)
            with gzip.open(p.stdout, "rt", encoding="utf-8") as fh:
                for line in fh:
                    n += 1
                    yield None, json.loads(line)["text"]
            p.wait()
            if p.returncode == 0 and n:
                return
        except (OSError, EOFError) as e:
            if n:
                raise
        time.sleep(5 + 10 * t)
    raise RuntimeError("stream failed: " + url)


def main():
    forms_csv, out_dir, kind, src = sys.argv[1:5]
    tag = sys.argv[5] if len(sys.argv) > 5 else re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(src).split(".")[0])
    os.makedirs(out_dir, exist_ok=True)
    C = Counter(forms_csv)
    C.rnd = random.Random(tag)
    if kind == "books":
        for line in open(src, encoding="utf-8"):
            path = line.strip()
            if not path:
                continue
            btag = os.path.basename(path)[:-4]
            if os.path.exists(os.path.join(out_dir, "part_%s.npz" % btag)):
                continue
            C.reset(); C.rnd = random.Random(btag)
            raw = open(path, "rb").read().decode("utf-8", "replace")
            text = strip_gutenberg(raw) if btag.startswith("gutenberg_") else raw
            C.doc(btag, text, book=True)
            C.save(out_dir, btag, path)
        return
    if os.path.exists(os.path.join(out_dir, "part_%s.npz" % tag)):
        print(tag, "exists"); return
    if kind == "parquet":
        local = src
        if src.startswith("http"):
            tmpd = os.environ.get("COOC_TMP", "/tmp"); os.makedirs(tmpd, exist_ok=True)
            local = os.path.join(tmpd, tag + ".parquet")
            fetch(src, local)
        try:
            for title, text in iter_parquet(local):
                C.doc(title, text)
        finally:
            if local != src and os.path.exists(local):
                os.remove(local)
    elif kind == "jsonlgz":
        for title, text in iter_jsonlgz(src):
            C.doc(title, text)
    else:
        raise SystemExit("unknown kind " + kind)
    C.save(out_dir, tag, src)


if __name__ == "__main__":
    main()
