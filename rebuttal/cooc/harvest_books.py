#!/usr/bin/env python
"""Harvest public-domain astronomy / sky-guide books -> books/raw/*.txt + books/candidates.csv

Sources (everything we can get without an access agreement):
  gutenberg  Project Gutenberg catalogue (pg_catalog.csv): English texts whose LoC class is QB
             (astronomy), or whose subjects / bookshelves mention astronomy, constellations,
             stars, telescopes, the zodiac, ...  Text from /cache/epub/<n>/pg<n>.txt.
  ia         Internet Archive scanned books (OCR *_djvu.txt), mediatype texts, English,
             dated 1600-1928 (US public domain), subject astronomy / stars / constellations /
             telescopes, or a title naming constellations, star atlases, star-gazing,
             telescopes, the heavens, the night sky, celestial ...
Not used (documented in BOOKS.md): HathiTrust (bulk full text needs a research agreement),
post-1928 guides (in copyright; IA only lends them), Wikisource (a handful, mostly in IA).

Nothing is filtered for "sky-tour-ness" here -- book_report.py scores every text, so the
selection is documented rather than assumed.

Usage: harvest_books.py <out_dir> [max_ia]
"""
import sys, os, csv, io, json, re, time, gzip, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "decodable-sky-research/0.1 (cooccurrence statistics; contact via repo)"}
GUT_CAT = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
GUT_TXT = "https://www.gutenberg.org/cache/epub/%s/pg%s.txt"
IA_SEARCH = "https://archive.org/advancedsearch.php"
IA_DJVU = "https://archive.org/download/%s/%s_djvu.txt"
IA_META = "https://archive.org/metadata/%s/files"

GUT_SUBJ = re.compile(r"astronom|constellation|\bstars\b|telescop|zodiac|planet|celestial|"
                      r"comets|meteor|cosmograph|uranograph|star catalog|sky", re.I)
IA_BASE = 'mediatype:texts AND date:[1600-01-01 TO 1928-12-31] AND (language:eng OR language:English)'
IA_TOPIC = ('(subject:astronomy OR subject:stars OR subject:constellations OR subject:telescopes '
            'OR subject:"star atlases" OR subject:"observers manuals" OR title:constellations '
            'OR title:"star atlas" OR title:stargazing OR title:"star-gazing" OR title:"star gazing" '
            'OR title:telescope OR title:"the heavens" OR title:"night sky" OR title:celestial '
            'OR title:"star lore" OR title:"star-lore" OR title:"star names" OR title:uranography)')


def get(url, tries=4, timeout=120):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                return None
            time.sleep(2 + 4 * t)
        except Exception:
            time.sleep(2 + 4 * t)
    return None


def gutenberg_candidates():
    raw = get(GUT_CAT, timeout=600)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    out = []
    for r in rows:
        if r.get("Type") != "Text" or "en" not in r.get("Language", "").split("; "):
            continue
        locc = r.get("LoCC", "")
        subj = r.get("Subjects", "") + " ; " + r.get("Bookshelves", "")
        if "QB" in locc.split("; ") or any(c.startswith("QB") for c in locc.split("; ")) \
                or GUT_SUBJ.search(subj):
            out.append(dict(source="gutenberg", id=r["Text#"], title=r["Title"].replace("\n", " "),
                            author=r.get("Authors", ""), year=r.get("Issued", "")[:4],
                            subjects=(r.get("Subjects", "") + " | LoCC " + locc)[:300],
                            url=GUT_TXT % (r["Text#"], r["Text#"])))
    return out


def ia_candidates(max_ia):
    q = IA_BASE + " AND " + IA_TOPIC
    out, page = [], 1
    while True:
        params = [("q", q), ("rows", "1000"), ("page", str(page)), ("output", "json")] + \
                 [("fl[]", f) for f in ("identifier", "title", "creator", "date", "subject", "downloads")]
        d = json.loads(get(IA_SEARCH + "?" + urllib.parse.urlencode(params)))
        docs = d["response"]["docs"]
        for x in docs:
            def s(v):
                return "; ".join(v) if isinstance(v, list) else (v or "")
            out.append(dict(source="ia", id=x["identifier"], title=s(x.get("title")).replace("\n", " "),
                            author=s(x.get("creator")), year=s(x.get("date"))[:4],
                            subjects=s(x.get("subject"))[:300],
                            url=IA_DJVU % (x["identifier"], x["identifier"])))
        if len(docs) < 1000 or len(out) >= max_ia:
            break
        page += 1
    return out[:max_ia]


def fetch(c, raw_dir):
    path = os.path.join(raw_dir, "%s_%s.txt" % (c["source"], re.sub(r"[^A-Za-z0-9._-]", "_", c["id"])))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "cached"
    b = get(c["url"])
    if b is None and c["source"] == "ia":            # OCR text under another file name
        m = get(IA_META % c["id"])
        if m:
            names = [f["name"] for f in json.loads(m).get("result", []) if f["name"].endswith("_djvu.txt")]
            if names:
                b = get("https://archive.org/download/%s/%s" % (c["id"], urllib.parse.quote(names[0])))
    if b is None:
        return "", "missing"
    with open(path + ".tmp", "wb") as fh:
        fh.write(b)
    os.replace(path + ".tmp", path)
    if c["source"] == "gutenberg":
        time.sleep(1.0)                                # Gutenberg asks crawlers to go slowly
    return path, "ok"


def main():
    out_dir = sys.argv[1]
    max_ia = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    gut = gutenberg_candidates(); print("gutenberg candidates:", len(gut), flush=True)
    ia = ia_candidates(max_ia); print("internet archive candidates:", len(ia), flush=True)
    cands = gut + ia
    rev = os.environ.get("HARVEST_REVERSE") == "1"      # a second worker from the other end
    if rev:
        cands = cands[::-1]
    res = {}
    with ThreadPoolExecutor(2) as pg, ThreadPoolExecutor(int(os.environ.get("HARVEST_THREADS", "8"))) as pi:
        futs = [((c["source"], c["id"]), (pg if c["source"] == "gutenberg" else pi).submit(fetch, c, raw_dir))
                for c in cands]
        for i, (k, f) in enumerate(futs):
            res[k] = f.result()
            if i % 250 == 0:
                print("  fetched %d/%d" % (i, len(futs)), flush=True)
    with open(os.path.join(out_dir, "candidates_rev.csv" if rev else "candidates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "id", "title", "author", "year", "subjects", "url",
                                           "path", "status", "bytes"])
        w.writeheader()
        for c in cands:
            p, st = res[(c["source"], c["id"])]
            w.writerow(dict(c, path=p, status=st, bytes=os.path.getsize(p) if p else 0))
    print("done:", sum(v[1] != "missing" for v in res.values()), "texts of", len(cands), flush=True)


if __name__ == "__main__":
    main()
