#!/usr/bin/env python
"""Constellation graph over the 188 catalogue objects (order = data_astro_objects.csv).

Edges
  * constellation -- constellation  iff the two IAU regions share a border of positive
    length (corner-only contacts are NOT edges);
  * star / other -- the constellation containing it (astropy get_constellation, ICRS).

Borders are computed exactly from the table astropy itself uses (Roman 1987, B1875):
astropy labels a point by the first row with ral < RA < rau and Dec > decl, so every
boundary lies on a table RA or Dec breakpoint. Labelling the centre of each cell of the
breakpoint grid is therefore exact, and a shared cell edge between two labels is a real
border; its length on the sphere is dDec (meridian edge) or 15 dRA cos(Dec) (parallel).
Independent checks: (a) cell labels vs astropy's public get_constellation on the B1875
cell centres, (b) random close point pairs through get_constellation in ICRS,
(c) star membership vs the IAU-CSN "Con" column.

Outputs (spectral/graphs/):
  constellation_borders.csv    88-constellation pairs with border length (deg)
  object_membership.csv        object -> constellation (+ IAU-CSN check for stars)
  graph_constellation.npz      A (188x188 0/1), border_deg (188x188, const-const only),
                               names, types, const_of (index of parent constellation or -1)
  graph_constellation_adjacency.csv
"""
import os, re, warnings, unicodedata
import numpy as np, pandas as pd
import paths
import astropy.units as u
import astropy.coordinates as ac
from astropy.coordinates import SkyCoord, get_constellation, PrecessedGeocentric
import scipy.sparse.csgraph as cg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "graphs")
DATA = os.path.join(os.path.dirname(ac.__file__), "data")
warnings.filterwarnings("ignore")


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", s.lower())


# ---- Roman 1987 table, exactly as astropy reads it
rows = [l.split() for l in open(os.path.join(DATA, "constellation_data_roman87.dat"))
        if l.strip() and not l.startswith("#")]
ral = np.array([float(r[0]) for r in rows]); rau = np.array([float(r[1]) for r in rows])
decl = np.array([float(r[2]) for r in rows]); abbr = np.array([r[3] for r in rows])
long_of = {}
for l in open(os.path.join(DATA, "constellation_names.dat"), encoding="utf-8"):
    if l.strip() and not l.startswith("#"):
        a, nm = l.strip().split(" ", 1); long_of[a] = nm


def label(ra_h, dec_d):
    idx = -np.ones(ra_h.shape, int)
    for i in range(len(rows)):
        m = (idx == -1) & (ral[i] < ra_h) & (ra_h < rau[i]) & (dec_d > decl[i])
        idx[m] = i
    assert (idx >= 0).all()
    return abbr[idx]


RAb = np.unique(np.concatenate([ral, rau, [0.0, 24.0]]))
DEb = np.unique(np.concatenate([decl, [-90.0, 90.0]]))
DEb = DEb[(DEb >= -90) & (DEb <= 90)]
rac = 0.5 * (RAb[1:] + RAb[:-1]); dec_c = 0.5 * (DEb[1:] + DEb[:-1])
RR, DD = np.meshgrid(rac, dec_c)                                  # (nDec, nRA)
lab = label(RR, DD)
print("breakpoint grid: %d RA x %d Dec cells, %d constellations present"
      % (len(rac), len(dec_c), len(np.unique(lab))))

border = {}


def add(a, b, length):
    if a == b or length <= 0:
        return
    k = tuple(sorted((a, b))); border[k] = border.get(k, 0.0) + length


dDec = np.diff(DEb)
for j in range(len(dec_c)):                        # meridian edges, incl. 24h -> 0h wrap
    for i in range(len(rac)):
        add(lab[j, i], lab[j, (i + 1) % len(rac)], dDec[j])
dRA = np.diff(RAb)
for j in range(len(dec_c) - 1):                    # parallel edges at Dec = DEb[j+1]
    c = np.cos(np.radians(DEb[j + 1]))
    for i in range(len(rac)):
        add(lab[j, i], lab[j + 1, i], 15.0 * dRA[i] * c)

# (a) exact labels agree with astropy public API on cell centres (B1875 frame)
cc = SkyCoord(ra=RR.ravel() * 15 * u.deg, dec=DD.ravel() * u.deg,
              frame=PrecessedGeocentric(equinox="B1875"))
api = get_constellation(cc, short_name=True)
print("(a) cell labels vs astropy get_constellation: %d / %d disagree"
      % ((api != lab.ravel()).sum(), lab.size))

# (b) random close pairs through the public API in ICRS
rng = np.random.default_rng(0); n = 3_000_000
v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
t = rng.normal(size=(n, 3)); t -= (t * v).sum(1, keepdims=True) * v
t /= np.linalg.norm(t, axis=1, keepdims=True)
eps = np.radians(0.1); w = np.cos(eps) * v + np.sin(eps) * t


def sc(x):
    return SkyCoord(x=x[:, 0], y=x[:, 1], z=x[:, 2], representation_type="cartesian", frame="icrs")


la = get_constellation(sc(v), short_name=True); lb = get_constellation(sc(w), short_name=True)
hit = {}
for a, b in zip(la[la != lb], lb[la != lb]):
    k = tuple(sorted((a, b))); hit[k] = hit.get(k, 0) + 1
# corner-only contacts: labels meeting diagonally at a grid vertex without a shared edge
corner = set()
nr = len(rac)
for j in range(len(dec_c) - 1):
    for i in range(nr):
        a, b = lab[j, i], lab[j + 1, (i + 1) % nr]; c2, d2 = lab[j + 1, i], lab[j, (i + 1) % nr]
        for x, y in ((a, b), (c2, d2)):
            if x != y and tuple(sorted((x, y))) not in border:
                corner.add(tuple(sorted((x, y))))
print("    corner-only contacts (not edges): %s" % sorted(corner))
exact = set(border); samp = set(hit)
print("(b) random 0.1-deg pairs: %d crossing pairs found; exact borders %d;\n"
      "    sampled-but-not-exact %s;\n    exact-but-not-sampled %s"
      % (len(samp), len(exact), sorted(samp - exact),
         sorted((k, round(border[k], 3)) for k in exact - samp)))

# ---- objects, membership
cat = pd.read_csv(paths.catalogue())
names = cat.name.tolist(); types = cat.type.to_numpy(); N = len(cat)
const_idx = {norm(nm.replace(" constellation", "")): i for i, nm in enumerate(names)
             if types[i] == "constellation"}
ASTROPY_SPELLING = {"chamaleon": "chamaeleon", "ophiucus": "ophiuchus",      # typos in astropy
                    "piscesaustrinus": "piscisaustrinus"}              # constellation_names.dat
abbr_to_obj = {a: const_idx[ASTROPY_SPELLING.get(norm(nm), norm(nm))] for a, nm in long_of.items()}
assert len(abbr_to_obj) == 88 and len(set(abbr_to_obj.values())) == 88

coords = SkyCoord(ra=cat.l_ra_deg.to_numpy() * u.deg, dec=cat.a_dec_deg.to_numpy() * u.deg, frame="icrs")
mem = get_constellation(coords, short_name=True)
csn = {}                      # IAU-CSN is fixed-width; slice columns by header position
lines = open(os.path.join(OUT, "IAU-CSN.txt"), encoding="utf-8").read().splitlines()
hdr = next(l for l in lines if l.startswith("#Name/ASCII"))
c_con, c_id = hdr.index("Con"), hdr.index("ID ")
for l in lines:
    if l.startswith(("#", "$")) or not l.strip():
        continue
    csn[l[:18].strip()] = dict(con=l[c_con:c_con + 3].strip(), bayer=l[c_id:c_id + 5].strip())
const_of = -np.ones(N, int); recs = []
for i in range(N):
    if types[i] == "constellation":
        continue
    const_of[i] = abbr_to_obj[mem[i]]
    chk = csn[names[i]]["con"] if types[i] == "star" and names[i] in csn else ""
    recs.append(dict(name=names[i], type=types[i], constellation=names[abbr_to_obj[mem[i]]].replace(" constellation", ""), abbr=mem[i],
                     iau_csn_con=chk, agree=(chk == mem[i]) if chk else ""))
memdf = pd.DataFrame(recs)
stars = memdf[memdf.type == "star"]
print("(c) stars: %d, found in IAU-CSN %d, membership disagreements: %s"
      % (len(stars), (stars.iau_csn_con != "").sum(),
         stars[(stars.iau_csn_con != "") & (stars.agree != True)][["name", "abbr", "iau_csn_con"]].values.tolist()))
print("    others:", "; ".join("%s->%s" % (r.name, r.abbr) for r in memdf[memdf.type == "other"].itertuples()))

A = np.zeros((N, N), np.int8); B = np.zeros((N, N))
for (a, b), L in border.items():
    i, j = abbr_to_obj[a], abbr_to_obj[b]; A[i, j] = A[j, i] = 1; B[i, j] = B[j, i] = L
for i in range(N):
    if const_of[i] >= 0:
        A[i, const_of[i]] = A[const_of[i], i] = 1

# ---- stats
deg = A.sum(1)
ncomp, _ = cg.connected_components(A, directed=False)
cdeg = {long_of[a]: int((B[abbr_to_obj[a]] > 0).sum()) for a in long_of}
print("\nborders: %d pairs, border-degree min %d (%s) median %.0f max %d (%s)"
      % (len(border), min(cdeg.values()), min(cdeg, key=cdeg.get), np.median(list(cdeg.values())),
         max(cdeg.values()), max(cdeg, key=cdeg.get)))
for c in ("Crux", "Ursa Minor", "Serpens", "Orion", "Hydra"):
    ia = abbr_to_obj[[a for a in long_of if long_of[a] == c][0]]
    nb = [names[j].replace(" constellation", "") for j in np.where(B[ia] > 0)[0]]
    print("  %-10s (%d): %s" % (c, len(nb), ", ".join(nb)))
short = sorted(border.items(), key=lambda kv: kv[1])[:6]
print("  shortest borders (deg):", ", ".join("%s-%s %.2f" % (a, b, L) for (a, b), L in short))
leaves = np.bincount(const_of[const_of >= 0], minlength=N)
print("graph: %d nodes, %d edges, %d connected component(s); leaves per constellation max %d (%s), "
      "constellations with 0 leaves %d"
      % (N, int(A.sum()) // 2, ncomp, leaves.max(), names[leaves.argmax()],
         sum(leaves[i] == 0 for i in range(N) if types[i] == "constellation")))

# ---- save
pd.DataFrame([dict(const_a=names[abbr_to_obj[a]].replace(" constellation", ""), const_b=names[abbr_to_obj[b]].replace(" constellation", ""), abbr_a=a, abbr_b=b, border_deg=round(L, 5))
              for (a, b), L in sorted(border.items())]).to_csv(os.path.join(OUT, "constellation_borders.csv"),
                                                                index=False)
memdf.to_csv(os.path.join(OUT, "object_membership.csv"), index=False)
np.savez(os.path.join(OUT, "graph_constellation.npz"), A=A, border_deg=B, names=np.array(names),
         types=types.astype(str), const_of=const_of)
pd.DataFrame(A, index=names, columns=names).to_csv(os.path.join(OUT, "graph_constellation_adjacency.csv"))
print("saved to", OUT)
