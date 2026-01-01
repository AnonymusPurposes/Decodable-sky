#!/usr/bin/env python
"""
Score ablation completions: what the model named, and how far it is from the prompt object.

Parsing (rules ported from steering/rotation/harmonic/parse_rich.py in the
Decodable-sky repo, adapted to this catalogue, which also holds stars and DSOs):
  * line breaks are flattened FIRST, so a completion that answers on the second line
    ("?\nThe closest constellation to X is Ophiuchus") is read, not truncated;
  * text is folded to unmarked letters, so "Boötes" matches the catalogue's "Bootes";
  * surface forms = catalogue names ("<Name> constellation" -> "<Name>"), Latin genitives,
    aliases (M42, Alpha Centauri), spelling variants ("Scorpio", "Pisces Austrinus"), and
    English names: multi-word phrases anywhere ("the Great Dog" can mean nothing else),
    single ordinary words only after a naming cue ("constellation of the Telescope"),
    since "seen through a telescope" is not an answer. "Crater" stays case-sensitive
    (lunar craters); everything else is case-insensitive;
  * the prompt's own phrasing is removed before reading: a literal restatement of the
    template, a generic question echo ("the closest constellation to X is"), an answer
    preamble ("the best answer is"), leading punctuation/bullets/markdown, and the
    question's object in place ("to X", "near the X constellation");
  * an enumerated list ("A. Orion B. Lyra ...", "1) ...", bullets) is read as a SET of
    candidates and scored by the MEAN angle of its members to the prompt object -- these
    are the completions the earlier parser discarded as "quiz";
  * named stars down to V = 4 that the catalogue (V <= 2.49) lacks are also recognised,
    from bright_stars_v4.csv (build_bright_stars.py, SIMBAD), matched case-sensitively
    because several of those names are ordinary words ("Atlas", "Chow", "Polis");
  * a completion that OPENS with a sky landmark -- the celestial equator, the ecliptic, the
    north/south celestial pole -- is scored as `landmark`: for a great circle the distance is
    |90 - angle(object, pole of that circle)|, i.e. |declination| for the equator and
    |ecliptic latitude| for the ecliptic; for a pole it is the angle to that point
    (90 -/+ declination). Only at the start: "15 degrees south of the celestial equator"
    later in a sentence is not an answer;
  * otherwise the FIRST catalogue mention decides: another object -> "object" (scored),
    X itself -> "self" (counted separately), nothing -> "no_object".
Distance = great-circle angle between catalogue positions (constellations: area centroids).
median_ci95_lo/hi = 2.5/97.5 percentiles of the bootstrap distribution of the MEDIAN
(prompt objects resampled as clusters) -- a CI on the statistic, not the spread of angles.
ang_p05/ang_p95 = 5th/95th percentiles of the individual angles (the spread).
*_any columns pool single answers with enumerations (at their mean angle) and landmarks.

Usage: python score_ablation.py completions_<model>.csv --objects astro_objects.csv
Writes <stem>_scored.csv (per row) and <stem>_summary.csv (per condition).
"""
import argparse
import csv
import math
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np

GENITIVE = {
    "Andromedae": "Andromeda", "Antliae": "Antlia", "Apodis": "Apus", "Aquarii": "Aquarius",
    "Aquilae": "Aquila", "Arae": "Ara", "Arietis": "Aries", "Aurigae": "Auriga",
    "Bootis": "Bootes", "Caeli": "Caelum", "Camelopardalis": "Camelopardalis",
    "Cancri": "Cancer", "Canum Venaticorum": "Canes Venatici", "Canis Majoris": "Canis Major",
    "Canis Minoris": "Canis Minor", "Capricorni": "Capricornus", "Carinae": "Carina",
    "Cassiopeiae": "Cassiopeia", "Centauri": "Centaurus", "Cephei": "Cepheus", "Ceti": "Cetus",
    "Chamaeleontis": "Chamaeleon", "Circini": "Circinus", "Columbae": "Columba",
    "Comae Berenices": "Coma Berenices", "Coronae Australis": "Corona Australis",
    "Coronae Borealis": "Corona Borealis", "Corvi": "Corvus", "Crateris": "Crater",
    "Crucis": "Crux", "Cygni": "Cygnus", "Delphini": "Delphinus", "Doradus": "Dorado",
    "Draconis": "Draco", "Equulei": "Equuleus", "Eridani": "Eridanus", "Fornacis": "Fornax",
    "Geminorum": "Gemini", "Gruis": "Grus", "Herculis": "Hercules", "Horologii": "Horologium",
    "Hydrae": "Hydra", "Hydri": "Hydrus", "Indi": "Indus", "Lacertae": "Lacerta",
    "Leonis": "Leo", "Leonis Minoris": "Leo Minor", "Leporis": "Lepus", "Librae": "Libra",
    "Lupi": "Lupus", "Lyncis": "Lynx", "Lyrae": "Lyra", "Mensae": "Mensa",
    "Microscopii": "Microscopium", "Monocerotis": "Monoceros", "Muscae": "Musca",
    "Normae": "Norma", "Octantis": "Octans", "Ophiuchi": "Ophiuchus", "Orionis": "Orion",
    "Pavonis": "Pavo", "Pegasi": "Pegasus", "Persei": "Perseus", "Phoenicis": "Phoenix",
    "Pictoris": "Pictor", "Piscium": "Pisces", "Piscis Austrini": "Piscis Austrinus",
    "Puppis": "Puppis", "Pyxidis": "Pyxis", "Reticuli": "Reticulum", "Sagittae": "Sagitta",
    "Sagittarii": "Sagittarius", "Scorpii": "Scorpius", "Sculptoris": "Sculptor",
    "Scuti": "Scutum", "Serpentis": "Serpens", "Sextantis": "Sextans", "Tauri": "Taurus",
    "Telescopii": "Telescopium", "Trianguli": "Triangulum",
    "Trianguli Australis": "Triangulum Australe", "Tucanae": "Tucana",
    "Ursae Majoris": "Ursa Major", "Ursae Minoris": "Ursa Minor", "Velorum": "Vela",
    "Virginis": "Virgo", "Volantis": "Volans", "Vulpeculae": "Vulpecula",
}
# Bayer designations ("Alpha Orionis") are matched through the genitive -> constellation,
# except where a specific catalogue object is meant:
ALIASES = {
    "Alpha Centauri": "Rigil Kentaurus", "Proxima Centauri": "Rigil Kentaurus",
    "Beta Centauri": "Hadar", "Big Dipper": "Ursa Major", "Little Dipper": "Ursa Minor",
    "Southern Cross": "Crux", "Great Bear": "Ursa Major", "North Star": "Polaris",
    "Pole Star": "Polaris", "Seven Sisters": "Pleiades", "M45": "Pleiades",
    "M42": "Orion Nebula", "M1": "Crab Nebula", "M31": "Andromeda Galaxy",
    "M33": "Triangulum Galaxy", "M51": "Whirlpool Galaxy", "M13": "Hercules Cluster",
    "M57": "Ring Nebula", "M104": "Sombrero Galaxy", "M44": "Beehive Cluster",
    "Praesepe": "Beehive Cluster", "M8": "Lagoon Nebula", "LMC": "Large Magellanic Cloud",
    "Rigil Kent": "Rigil Kentaurus",
}
ENGLISH = {
    "Andromeda": ["princess", "chained maiden", "chained woman", "chained princess"],
    "Antlia": ["air pump", "pump"], "Apus": ["bird of paradise"],
    "Aquarius": ["water bearer", "water carrier"], "Aquila": ["eagle"], "Ara": ["altar"],
    "Aries": ["ram"], "Auriga": ["charioteer"], "Bootes": ["herdsman", "ploughman"],
    "Caelum": ["chisel", "graving tool"], "Camelopardalis": ["giraffe"], "Cancer": ["crab"],
    "Canes Venatici": ["hunting dogs"], "Canis Major": ["great dog", "greater dog", "big dog"],
    "Canis Minor": ["little dog", "lesser dog", "small dog"],
    "Capricornus": ["sea goat", "sea-goat", "goat"], "Carina": ["keel"],
    "Cassiopeia": ["queen", "seated queen"], "Centaurus": ["centaur"], "Cepheus": ["king"],
    "Cetus": ["whale", "sea monster"], "Chamaeleon": ["chameleon"],
    "Circinus": ["compasses", "pair of compasses"], "Columba": ["dove"],
    "Coma Berenices": ["berenice's hair"], "Corona Australis": ["southern crown"],
    "Corona Borealis": ["northern crown"], "Corvus": ["crow", "raven"], "Crater": ["cup"],
    "Crux": ["cross", "southern cross"], "Cygnus": ["swan", "northern cross"],
    "Delphinus": ["dolphin"], "Dorado": ["goldfish", "swordfish", "dolphinfish"],
    "Draco": ["dragon"], "Equuleus": ["little horse", "foal", "pony"], "Eridanus": ["river"],
    "Fornax": ["furnace"], "Gemini": ["twins"], "Grus": ["crane"],
    "Hercules": ["strongman", "hero hercules"], "Horologium": ["clock", "pendulum clock"],
    "Hydra": ["water snake", "sea serpent", "female water snake"],
    "Hydrus": ["male water snake", "little water snake", "lesser water snake"],
    "Indus": ["indian"], "Lacerta": ["lizard"], "Leo": ["lion"],
    "Leo Minor": ["little lion", "lesser lion", "smaller lion"], "Lepus": ["hare"],
    "Libra": ["scales", "balance"], "Lupus": ["wolf"], "Lyra": ["lyre", "harp"],
    "Mensa": ["table mountain"], "Microscopium": ["microscope"], "Monoceros": ["unicorn"],
    "Musca": ["fly"], "Norma": ["level", "carpenter's square", "set square"],
    "Octans": ["octant"], "Ophiuchus": ["serpent bearer", "serpent-bearer", "serpent holder"],
    "Orion": ["hunter"], "Pavo": ["peacock"], "Pegasus": ["winged horse", "flying horse"],
    "Perseus": ["hero"], "Phoenix": ["phoenix"], "Pictor": ["painter's easel", "easel", "painter"],
    "Pisces": ["fishes", "fish"], "Piscis Austrinus": ["southern fish"],
    "Puppis": ["stern", "poop deck"], "Pyxis": ["compass", "mariner's compass"],
    "Reticulum": ["reticle", "net", "small net"], "Sagitta": ["arrow"],
    "Sagittarius": ["archer"], "Scorpius": ["scorpion"], "Sculptor": ["sculptor"],
    "Scutum": ["shield"], "Serpens": ["serpent", "snake"], "Sextans": ["sextant"],
    "Taurus": ["bull"], "Telescopium": ["telescope"], "Triangulum": ["triangle"],
    "Triangulum Australe": ["southern triangle"], "Tucana": ["toucan"],
    "Ursa Major": ["great bear", "big bear", "greater bear"],
    "Ursa Minor": ["little bear", "lesser bear", "small bear"], "Vela": ["sails"],
    "Virgo": ["virgin", "maiden"], "Volans": ["flying fish"], "Vulpecula": ["fox", "little fox"],
}
# Spelling variants models actually write (parse_rich.VARIANTS). Longest-form-first
# matching keeps "Pisces Austrinus" from losing its first word to "Pisces".
VARIANTS = {
    "Piscis Austrinus": ["Pisces Austrinus"], "Scorpius": ["Scorpio"],
    "Triangulum Australe": ["Triangulum Australis"],
}
# Leading noise; a bullet that opens a list item is NOT noise (no space after it).
NOISE = re.compile(r'^(?:[\s?!.,:;"\'‘’“”]|[*•–—-](?!\s))+')
ECHO = re.compile(
    r'^\s*(?:the\s+)?(?:closest|nearest|neighbou?ring|adjacent|bordering)\s+'
    r'(?:constellations?|stars?|objects?)\s+(?:to|of)\s+.{0,48}?\s+(?:is|are|include[sd]?)\s*[:,]?\s*',
    re.I)
PREAMBLE = re.compile(
    r'^\s*(?:the\s+)?(?:best\s+|correct\s+|right\s+)?answer\s*(?:is|:)\s*', re.I)
# "A. " / "A) " / "1. " / "* " opening a list ITEM: a word must follow the marker, so a
# decimal in running text ("1. 1 degrees away") is not an enumeration.
MARKER = re.compile(r'(?:^|\s)(?:[A-Ha-h][.):]|\d{1,2}[.)]|[*•–-])\s+(?=[A-Za-z*"\'(])')
# Naming cues that license an English name which is also an ordinary word.
CUE = (r'(?:constellations?\s+(?:of|named|called)|known\s+as|called|nicknamed|'
       r'[A-Ha-h][.):]|\d{1,2}[.)]|\(|\*\*|[*•–-]\s)\s*(?:the\s+)?')


# Sky landmarks a model may answer with instead of an object. Counted ONLY when the cleaned
# completion STARTS with one (mentioning the equator later in a sentence is not an answer).
# A great circle is stored by its pole: distance = |90 - angle(object, pole)|, so the celestial
# equator gives the object's |declination| and the ecliptic its |ecliptic latitude|. A pole is a
# point: distance = angle(object, pole) = 90 -/+ declination.
OBLIQUITY = 23.4392911          # J2000 mean obliquity; north ecliptic pole at RA 270 deg
LANDMARKS = {
    "celestial equator": ("circle", (90.0, 0.0)),          # (dec, ra) of the defining pole
    "ecliptic": ("circle", (90.0 - OBLIQUITY, 270.0)),
    "north celestial pole": ("point", (90.0, 0.0)),
    "south celestial pole": ("point", (-90.0, 0.0)),
}
LANDMARK_RE = [
    ("celestial equator", re.compile(r"^(?:the\s+)?celestial\s+equator\b", re.I)),
    ("ecliptic", re.compile(r"^(?:the\s+)?(?:plane\s+of\s+the\s+)?ecliptic\b", re.I)),
    ("north celestial pole", re.compile(
        r"^(?:the\s+)?(?:north(?:ern)?\s+celestial\s+pole|celestial\s+north\s+pole"
        r"|north(?:ern)?\s+pole\s+of\s+the\s+(?:sky|celestial\s+sphere))\b", re.I)),
    ("south celestial pole", re.compile(
        r"^(?:the\s+)?(?:south(?:ern)?\s+celestial\s+pole|celestial\s+south\s+pole"
        r"|south(?:ern)?\s+pole\s+of\s+the\s+(?:sky|celestial\s+sphere))\b", re.I)),
]


def landmark_at_start(body):
    """The sky landmark the completion opens with, if any."""
    for name, pat in LANDMARK_RE:
        if pat.match(body):
            return name
    return None


def landmark_distance(name, u):
    """Angular distance from object `u` to the landmark (see LANDMARKS)."""
    kind, (dec, ra) = LANDMARKS[name]
    p = unit(dec, ra)
    a = ang(u, p)
    return abs(90.0 - a) if kind == "circle" else a


def fold(text):
    """Drop combining marks, so "Boötes" reads as the catalogue's "Bootes"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def flatten(text):
    """Line breaks first: a completion may answer on a later line."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def unit(dec, ra):
    d, r = math.radians(dec), math.radians(ra)
    return np.array([math.sin(d), math.cos(d) * math.sin(r), math.cos(d) * math.cos(r)])


def ang(u, v):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(u, v)), float(np.dot(u, v))))


def load_stars(path, pos, known):
    """Extra named stars (V < 4) that are not in the 188-object catalogue.

    Returns {surface form -> canonical name} and fills `pos` with their unit vectors.
    Built by build_bright_stars.py from SIMBAD. Matched CASE-SENSITIVELY by the caller:
    several of these names are also ordinary words ("Atlas", "Chow", "Polis").
    """
    forms = {}
    if not path or not os.path.exists(path):
        return forms
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["surface"] in known or r["canonical"] in known:
                continue
            pos[r["canonical"]] = unit(float(r["dec_deg"]), float(r["ra_deg"]))
            forms[fold(r["surface"])] = r["canonical"]
    return forms


def load_catalogue(path, stars_path=None):
    """(positions, anywhere lookup, cued lookup, star lookup, compiled matcher)."""
    pos, canon = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pos[r["name"]] = unit(float(r["a_dec_deg"]), float(r["l_ra_deg"]))
            canon[r["name"]] = r["name"]
            if r["type"] == "constellation":
                canon[r["name"].replace(" constellation", "")] = r["name"]
    anywhere = dict(canon)
    for src in (GENITIVE, ALIASES):
        for form, target in src.items():
            if target in canon:
                anywhere[form] = canon[target]
    for target, forms in VARIANTS.items():
        for form in forms:
            if target in canon:
                anywhere[form] = canon[target]
    cued = {}
    for target, words in ENGLISH.items():
        if target not in canon:
            continue
        for w in words:                      # multi-word phrases name nothing else
            (anywhere if " " in w else cued)[w] = canon[target]
    stars = load_stars(stars_path, pos, set(anywhere) | set(cued))
    fold_keys = lambda d: {fold(k).casefold(): v for k, v in d.items()}
    a_keys = sorted((k for k in anywhere if k != "Crater"), key=len, reverse=True)
    a_alt = "|".join(["(?-i:Crater)"] + [re.escape(k) for k in a_keys])
    c_alt = "|".join(re.escape(k) for k in sorted(cued, key=len, reverse=True))
    s_alt = "|".join(re.escape(k) for k in sorted(stars, key=len, reverse=True))
    star_part = (r"|(?<![\w-])(?P<star>(?-i:" + s_alt + r"))(?![\w-])") if stars else ""
    pat = re.compile(r"(?<![\w-])(?P<any>" + a_alt + r")(?![\w-])" + star_part +
                     r"|(?:" + CUE + r")(?P<cued>" + c_alt + r")(?![\w-])", re.I)
    return pos, fold_keys(anywhere), fold_keys(cued), stars, pat


def mentions(text, anywhere, cued, stars, pat):
    """Catalogue objects (and extra named stars) in `text`, in order, without repeats."""
    seen = []
    for m in pat.finditer(text):
        if m.group("any"):
            name = anywhere[fold(m.group("any")).casefold()]
        elif stars and m.group("star"):
            name = stars[fold(m.group("star"))]
        else:
            name = cued[fold(m.group("cued")).casefold()]
        if name not in seen:
            seen.append(name)
    return seen


def source_forms(name):
    """The prompt object's surface forms, longest first ("Orion constellation", "Orion")."""
    forms = {name, name.replace(" constellation", "")}
    return "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))


def strip_lead(text, limit=8):
    """Remove leading punctuation, question restatements and answer preambles."""
    out = text
    for _ in range(limit):
        prev = out
        out = PREAMBLE.sub("", ECHO.sub("", NOISE.sub("", out), count=1), count=1)
        if out == prev:
            break
    return out


def drop_question_restatement(text, name):
    """Delete a leading restatement of the question; say whether there was one."""
    alt = source_forms(name)
    named = r"(?:constellation\s+(?:of\s+)?(?:the\s+)?(?:%s)|(?:%s)\s+constellation)" % (alt, alt)
    head = re.match(r"\s*(?:\?\s*(?:the\s+)?(?:constellation\s+(?:of\s+)?(?:the\s+)?)?(?:%s)"
                    r"(?:\s+constellation)?|(?:the\s+)?%s)\b" % (alt, named), text, re.I)
    if head is None:
        return text, False
    rest = text[head.end():]
    while True:
        more = re.match(r"\s*[,(\[–—-]\s*(?:(?:also\s+)?(?:known\s+as|called|nicknamed|named)"
                        r"\s+|or\s+)?(?:the\s+)?(?:%s)(?:\s+constellation)?\s*[)\],]?" % alt,
                        rest, re.I)
        if more is None:
            return rest, True
        rest = rest[more.end():]


def drop_question_object(text, name):
    """Delete "to/near/of (the) <X>" - the question's object, never an answer."""
    return re.sub(r"\b(?:to|of|near|beside|around|from)\s+(?:the\s+)?(?:%s)"
                  r"(?:\s+constellation)?\b" % source_forms(name), " ", text, flags=re.I)


def read(body, anywhere, cued, stars, pat):
    """Return (names, kind) with kind in {'options', 'single', 'none'}."""
    marks = list(MARKER.finditer(" " + body))
    if marks:
        cut = max(marks[0].start() - 1, 0)
        if mentions(body[:cut], anywhere, cued, stars, pat):
            body = body[:cut]                      # answer given before the list
        else:
            ids = mentions(body[cut:], anywhere, cued, stars, pat)
            if ids:                                # a truncated list of one = single answer
                return ids, "options" if len(ids) >= 2 else "single"
    ids = mentions(body, anywhere, cued, stars, pat)
    return (ids, "single") if ids else ([], "none")


def parse(text, self_name, phrase, anywhere, cued, stars, pat):
    """(category, answer, next_other_after_self, kind, names).

    category: object (another catalogue object named first) | options (an enumeration,
    scored by the MEAN angle of its members) | landmark (the completion opens with the
    celestial equator / ecliptic / a celestial pole) | self (X itself named first) |
    no_object | empty.
    """
    body = flatten(fold(text))
    if not re.search(r"[A-Za-z]", body):
        return "empty", None, None, "none", []
    if phrase:
        body = re.sub(re.escape(flatten(fold(phrase))), " ", body, flags=re.I)
    body, restated = drop_question_restatement(body, self_name)
    body = drop_question_object(strip_lead(body), self_name)
    lm = landmark_at_start(body)
    if lm:
        return "landmark", lm, None, "landmark", [lm]
    names, kind = read(body, anywhere, cued, stars, pat)
    others = [n for n in names if n != self_name]
    if kind == "options":
        if len(others) >= 2:
            return "options", "|".join(others), None, kind, others
        if len(others) == 1:
            return "object", others[0], None, "single", others
        if names:
            return "self", self_name, None, kind, names
        return "no_object", None, None, "ramble" if restated else "none", []
    if not names:
        return "no_object", None, None, "ramble" if restated else "none", []
    if names[0] == self_name:
        return "self", self_name, (others[0] if others else None), kind, names
    return "object", names[0], None, kind, [names[0]]


def median_ci(per_obj, n_boot=2000, seed=0):
    """95% CI of the pooled median, resampling prompt objects (clusters) with replacement."""
    keys = [k for k, v in per_obj.items() if v]
    if not keys:
        return "", ""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(per_obj[k]) for k in keys]
    meds = [np.median(np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))]))
            for _ in range(n_boot)]
    return round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("completions")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--stars", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "bright_stars_v4.csv"),
                    help="extra named stars (V < 4) from build_bright_stars.py; empty to disable")
    args = ap.parse_args()
    pos, anywhere, cued, stars, pat = load_catalogue(args.objects, args.stars)
    stem = args.completions[:-4] if args.completions.endswith(".csv") else args.completions

    with open(args.completions, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scored = []
    for r in rows:
        cat, nm, nxt, kind, names = parse(r["completion"], r["object"], r.get("prompt", ""),
                                          anywhere, cued, stars, pat)
        if cat == "object":
            d = ang(pos[nm], pos[r["object"]])
        elif cat == "options":
            d = float(np.mean([ang(pos[n], pos[r["object"]]) for n in names]))
        elif cat == "landmark":
            d = landmark_distance(nm, pos[r["object"]])
        else:
            d = None
        scored.append(dict(r, category=cat, kind=kind, n_named=len(names), answer=nm or "",
                           next_other_after_self=nxt or "",
                           ang_deg="" if d is None else round(d, 2)))
    with open(stem + "_scored.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)

    base = {(r["pr_id"], r["obj_id"]): r["answer"] for r in scored if r["condition"] == "none"}
    conds = list(dict.fromkeys(r["condition"] for r in scored))
    out = []
    for cond in conds:
        for typ in ("all", "star", "constellation", "other"):
            sel = [r for r in scored if r["condition"] == cond and (typ == "all" or r["type"] == typ)]
            n = max(len(sel), 1)
            frac = lambda c: round(sum(r["category"] == c for r in sel) / n, 4)
            pick = lambda c: np.array([float(r["ang_deg"]) for r in sel if r["category"] == c])
            dd, dopt, dlm = pick("object"), pick("options"), pick("landmark")
            parts = [a for a in (dd, dopt, dlm) if len(a)]
            dany = np.concatenate(parts) if parts else np.array([])
            scored_cats = ("object", "options", "landmark")
            di = np.array([float(r["ang_deg"]) if r["category"] in scored_cats else 0.0
                           for r in sel if r["category"] in scored_cats + ("self",)])
            per_obj = defaultdict(list)
            for r in sel:
                if r["category"] == "object":
                    per_obj[r["obj_id"]].append(float(r["ang_deg"]))
            lo, hi = median_ci(per_obj)
            same = [r["answer"] == base[(r["pr_id"], r["obj_id"])]
                    for r in sel if base.get((r["pr_id"], r["obj_id"]))]
            q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else ""
            out.append(dict(
                condition=cond, type=typ, n=len(sel),
                frac_parsed=frac("object"), frac_options=frac("options"),
                frac_landmark=frac("landmark"),
                frac_parsed_any=round(frac("object") + frac("options") + frac("landmark"), 4),
                frac_self=frac("self"), frac_no_object=frac("no_object"), frac_empty=frac("empty"),
                median_ang_deg=q(dd, 50), median_ci95_lo=lo, median_ci95_hi=hi,
                ang_p05=q(dd, 5), ang_p95=q(dd, 95),
                mean_ang_deg=round(float(dd.mean()), 2) if len(dd) else "",
                frac_within_30deg=round(float((dd <= 30).mean()), 4) if len(dd) else "",
                median_options_ang_deg=q(dopt, 50), median_landmark_ang_deg=q(dlm, 50),
                median_ang_deg_any=q(dany, 50),
                frac_within_30deg_any=round(float((dany <= 30).mean()), 4) if len(dany) else "",
                median_ang_incl_self_as_0=q(di, 50),
                frac_same_answer_as_none=round(float(np.mean(same)), 4) if base else ""))
    with open(stem + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for o in out:
        if o["type"] == "all":
            print(o)


if __name__ == "__main__":
    main()
