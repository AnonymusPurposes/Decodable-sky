#!/usr/bin/env python
"""Surface-form table for the corpus co-occurrence graph -> graphs/surface_forms.csv

One row per (form, object). Forms are in the corpus tokenizer's normal form
(lowercase, ASCII, single spaces). Counting happens per form; which forms are merged
into an object is decided later at export, so every row carries a `kind`:

  canonical   the catalogue name ("Kaus Australis", "Ursa Major", "Andromeda Galaxy")
  bayer       Greek letter + Latin genitive from IAU-CSN ("beta gruis" -> Tiaki).
              Many of the 85 stars got their IAU names in 2016-18 and Wikipedia mostly
              uses the Bayer designation. "alpha centauri" maps to BOTH Rigil Kentaurus
              (alf Cen A) and Toliman (alf Cen B).
  alias       curated unambiguous alternatives (Messier / NGC ids, Praesepe, Pole Star ...)

`confirm` names a word that must occur within +-10 tokens for a mention to count in
the astronomical variant: forms whose other sense survives an astronomy-context gate
(lunar craters, Pluto's moon Hydra, the Phoenix Mars lander, the Tropic of Cancer, sculptors, the
peacock that Pavo depicts).
"""
import os, re, ast
import pandas as pd
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "graphs")

GREEK = dict(alf="alpha", bet="beta", gam="gamma", ksi="xi", omi="omicron", tet="theta",
             iot="iota", kap="kappa", lam="lambda", mu="mu", nu="nu", pi="pi", rho="rho",
             sig="sigma", tau="tau", ups="upsilon", phi="phi", chi="chi", psi="psi",
             ome="omega", eps="epsilon", zet="zeta", eta="eta", **{"del": "delta"})
ABBR = {  # IAU abbreviation -> catalogue constellation name
    "And": "Andromeda", "Ant": "Antlia", "Aps": "Apus", "Aqr": "Aquarius", "Aql": "Aquila",
    "Ara": "Ara", "Ari": "Aries", "Aur": "Auriga", "Boo": "Bootes", "Cae": "Caelum",
    "Cam": "Camelopardalis", "Cnc": "Cancer", "CVn": "Canes Venatici", "CMa": "Canis Major",
    "CMi": "Canis Minor", "Cap": "Capricornus", "Car": "Carina", "Cas": "Cassiopeia",
    "Cen": "Centaurus", "Cep": "Cepheus", "Cet": "Cetus", "Cha": "Chamaeleon", "Cir": "Circinus",
    "Col": "Columba", "Com": "Coma Berenices", "CrA": "Corona Australis", "CrB": "Corona Borealis",
    "Crv": "Corvus", "Crt": "Crater", "Cru": "Crux", "Cyg": "Cygnus", "Del": "Delphinus",
    "Dor": "Dorado", "Dra": "Draco", "Equ": "Equuleus", "Eri": "Eridanus", "For": "Fornax",
    "Gem": "Gemini", "Gru": "Grus", "Her": "Hercules", "Hor": "Horologium", "Hya": "Hydra",
    "Hyi": "Hydrus", "Ind": "Indus", "Lac": "Lacerta", "Leo": "Leo", "LMi": "Leo Minor",
    "Lep": "Lepus", "Lib": "Libra", "Lup": "Lupus", "Lyn": "Lynx", "Lyr": "Lyra", "Men": "Mensa",
    "Mic": "Microscopium", "Mon": "Monoceros", "Mus": "Musca", "Nor": "Norma", "Oct": "Octans",
    "Oph": "Ophiuchus", "Ori": "Orion", "Pav": "Pavo", "Peg": "Pegasus", "Per": "Perseus",
    "Phe": "Phoenix", "Pic": "Pictor", "Psc": "Pisces", "PsA": "Piscis Austrinus", "Pup": "Puppis",
    "Pyx": "Pyxis", "Ret": "Reticulum", "Sge": "Sagitta", "Sgr": "Sagittarius", "Sco": "Scorpius",
    "Scl": "Sculptor", "Sct": "Scutum", "Ser": "Serpens", "Sex": "Sextans", "Tau": "Taurus",
    "Tel": "Telescopium", "Tri": "Triangulum", "TrA": "Triangulum Australe", "Tuc": "Tucana",
    "UMa": "Ursa Major", "UMi": "Ursa Minor", "Vel": "Vela", "Vir": "Virgo", "Vol": "Volans",
    "Vul": "Vulpecula"}
ALIASES = {
    "Rigil Kentaurus": ["rigil kent"],
    "Polaris": ["north star", "pole star", "polestar"],
    "Crux": ["southern cross"],
    "Pleiades": ["seven sisters", "m45", "messier 45"],
    "Orion Nebula": ["m42", "messier 42", "ngc 1976", "great orion nebula"],
    "Crab Nebula": ["messier 1", "ngc 1952"],          # bare m1..m9 = spectral classes
    "Andromeda Galaxy": ["m31", "messier 31", "ngc 224", "andromeda nebula", "great andromeda nebula"],
    "Triangulum Galaxy": ["m33", "messier 33", "ngc 598"],
    "Whirlpool Galaxy": ["m51", "messier 51", "ngc 5194"],
    "Hercules Cluster": ["m13", "messier 13", "ngc 6205", "hercules globular cluster",
                         "great globular cluster in hercules", "great hercules cluster"],
    "Ring Nebula": ["m57", "messier 57", "ngc 6720"],
    "Sombrero Galaxy": ["m104", "messier 104", "ngc 4594"],
    "Beehive Cluster": ["m44", "messier 44", "ngc 2632", "praesepe"],
    "Lagoon Nebula": ["messier 8", "ngc 6523"],
    "Omega Centauri": ["ngc 5139"],
    "Large Magellanic Cloud": ["lmc"],
    "Carina Nebula": ["ngc 3372", "eta carinae nebula", "great carina nebula"],
    "Tarantula Nebula": ["30 doradus", "ngc 2070"],
}
CONFIRM = {"crater": "constellation", "hydra": "constellation", "phoenix": "constellation", "sculptor": "constellation",
           "cancer": "constellation", "peacock": "star"}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def main():
    cat = pd.read_csv(paths.catalogue())
    gen = os.path.join(ROOT, "ablation", "score_ablation.py")     # genitive forms live there
    if not os.path.exists(gen):
        raise SystemExit("needs %s from the parent project; graphs/surface_forms.csv is the "
                         "shipped artifact of this script" % gen)
    src = open(gen, encoding="utf-8").read()
    genitive = {c: g for g, c in ast.literal_eval(re.search(r"GENITIVE = (\{.*?\n\})", src, re.S).group(1)).items()}
    lines = open(os.path.join(OUT, "IAU-CSN.txt"), encoding="utf-8").read().splitlines()
    hdr = next(l for l in lines if l.startswith("#Name/ASCII"))
    cc, ci = hdr.index("Con"), hdr.index("ID ")
    csn = {l[:18].strip(): (l[ci:ci + 5].strip(), l[cc:cc + 3].strip())
           for l in lines if l.strip() and not l.startswith(("#", "$"))}

    rows = []
    for nm, tp in zip(cat.name, cat.type):
        base = nm.replace(" constellation", "")
        f = norm(base)
        rows.append(dict(form=f, object=nm, type=tp, kind="canonical", confirm=CONFIRM.get(f, "")))
        if tp == "star":
            bay, con = csn[nm]
            rows.append(dict(form=norm(GREEK[bay] + " " + genitive[ABBR[con]]), object=nm, type=tp,
                             kind="bayer", confirm=""))
        for a in ALIASES.get(base, []):
            rows.append(dict(form=norm(a), object=nm, type=tp, kind="alias", confirm=""))
    df = pd.DataFrame(rows)
    assert not df.duplicated(["form", "object"]).any()
    multi = df.groupby("form").object.nunique()
    print("%d rows, %d distinct forms; forms mapping to >1 object: %s"
          % (len(df), df.form.nunique(), multi[multi > 1].index.tolist()))
    clash = set(df[df.kind != "canonical"].form) & set(df[df.kind == "canonical"].form)
    print("non-canonical forms equal to a canonical name:", sorted(clash))
    print(df.kind.value_counts().to_dict(), "| confirm-gated:", df[df.confirm != ""].form.tolist())
    df.to_csv(os.path.join(OUT, "surface_forms.csv"), index=False)


if __name__ == "__main__":
    main()
