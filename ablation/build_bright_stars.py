#!/usr/bin/env python
"""
Build the extra-star table the scorer uses to recognise named stars that are NOT in the
188-object catalogue (which stops at V = 2.49).

Source: SIMBAD TAP (the same service build_star_metadata.py queries), standard library only.
Selects every object with V < --vmag that carries at least one "NAME <proper name>"
identifier, keeps genuine stars (otype containing '*', excluding clusters/asterisks/galaxies),
and writes one row per surface form:

    surface, canonical, ra_deg, dec_deg, vmag, main_id, otype

Surface forms must be >= 4 letters (dropping "Ain", "Ran", "Tyl", which would collide with
ordinary words), and objects already in the catalogue are skipped so the catalogue stays
authoritative. The scorer matches these CASE-SENSITIVELY: unlike the 88 constellations,
several of these names are also ordinary words ("Atlas", "Chow", "Polis"), and models
capitalise a star name they mean to use.

Usage: python build_bright_stars.py [--vmag 4.0] [--objects ../data/astro_objects.csv]
"""
import argparse
import csv
import os
import re
import urllib.parse
import urllib.request

TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
QUERY = ("SELECT b.main_id, b.otype, b.ra, b.dec, f.V, i.id "
         "FROM basic b JOIN allfluxes f ON b.oid = f.oidref JOIN ident i ON i.oidref = b.oid "
         "WHERE f.V < {vmag} AND b.ra IS NOT NULL AND i.id LIKE 'NAME %'")
# otypes that carry a '*' but are not single stars
NOT_A_STAR = {"Cl*", "OpC", "GlC", "As*", "G", "AGN", "GiC", "GiG", "IG", "SNR", "HII", "PN"}
SURFACE = re.compile(r"[A-Za-zÀ-ɏ' -]{4,}")
# Star names that are also ordinary words models capitalise in astronomy prose. "Atlas"
# (27 Tau) matched 78 times in the stored completions, every one of them a book title
# ("Sky Atlas 2000", "Clementine Atlas"); "Chow" is a surname/common noun. Every other
# added name was checked against the completions and matched only genuine star mentions.
AMBIGUOUS = {"Atlas", "Chow"}


def fetch(vmag, timeout=180):
    body = urllib.parse.urlencode({"request": "doQuery", "lang": "adql", "format": "csv",
                                   "maxrec": "20000", "query": QUERY.format(vmag=vmag)})
    with urllib.request.urlopen(TAP, data=body.encode(), timeout=timeout) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmag", type=float, default=4.0, help="keep stars brighter than this V")
    ap.add_argument("--objects", default="../data/astro_objects.csv")
    ap.add_argument("--out", default="bright_stars_v4.csv")
    args = ap.parse_args()

    with open(args.objects, newline="", encoding="utf-8-sig") as f:
        catalogue = {r["name"] for r in csv.DictReader(f)}

    rows = fetch(args.vmag)
    by_obj = {}
    for r in rows:
        if "*" not in r["otype"] or r["otype"] in NOT_A_STAR:
            continue
        name = r["id"].replace("NAME ", "").strip()
        if not SURFACE.fullmatch(name) or name in AMBIGUOUS:
            continue
        by_obj.setdefault(r["main_id"], dict(otype=r["otype"], ra=r["ra"], dec=r["dec"],
                                             v=r["V"], names=[]))["names"].append(name)

    out = []
    for main_id, o in sorted(by_obj.items()):
        names = sorted(set(o["names"]), key=len)
        if any(n in catalogue for n in names):        # catalogue wins
            continue
        canonical = names[0]
        for n in names:
            out.append(dict(surface=n, canonical=canonical, ra_deg=round(float(o["ra"]), 6),
                            dec_deg=round(float(o["dec"]), 6), vmag=o["v"],
                            main_id=main_id, otype=o["otype"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["surface", "canonical", "ra_deg", "dec_deg",
                                          "vmag", "main_id", "otype"])
        w.writeheader()
        w.writerows(out)
    print(f"saved {args.out}: {len(out)} surface forms for "
          f"{len({r['canonical'] for r in out})} stars with V < {args.vmag} "
          f"(catalogue already holds {len(catalogue)} objects)")


if __name__ == "__main__":
    main()
