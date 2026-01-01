#!/usr/bin/env python
"""Generate steering/rotation/harmonic/run_pair_holdout.py from run.py with a minimal patch.

Kept as a generator so the difference from run.py stays explicit: only the per-prompt choice of
map changes. run.py itself is not modified.
"""
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "steering" / "rotation" / "harmonic"
s = (D / "run.py").read_bytes().decode().replace("\r\n", "\n")


def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


sub('"""Single or joint-layer Mistral interventions, with optional sustained replacement."""',
    '"""run.py with one fitted map per evaluated pair (see prepare_pair_holdout.py).\n\n'
    "Identical to run.py except that the self and target arms use, for each prompt, the map fitted\n"
    'without that prompt\'s source and destination constellations.\n"""')
sub("import argparse\nimport importlib.metadata\n", "import argparse\nimport copy\nimport importlib.metadata\n")
sub("from core import MODEL, REVISION, SEED, Chart, digest, display, harmonics,layer_groups,group_label\n",
    "from core import MODEL, REVISION, SEED, Chart, basis, digest, display, harmonics,layer_groups,group_label\n")
sub("                ideal=chart.replace(live[i],target)\n                ideal_trace=chart.trace(live[i],ideal,target)\n",
    "                ch=chart[i] if isinstance(chart,list) else chart\n"
    "                ideal=ch.replace(live[i],target)\n                ideal_trace=ch.trace(live[i],ideal,target)\n")
sub("                trace=chart.trace(live[i],actual,target)\n", "                trace=ch.trace(live[i],actual,target)\n")
sub("    snapshot=out/'inference_snapshot'; snapshot.mkdir()\n",
    "    pair_charts={}\n"
    "    for (l,k),base in charts.items():\n"
    "        z=np.load(prep/f'pair_charts_L{l}_K{k}.npz')\n"
    "        items=[]\n"
    "        for b,W in zip(z['b'],z['W']):\n"
    "            ch=copy.copy(base); ch.b=b; ch.W=W; ch.Q,ch.singular_values=basis(W); items.append(ch)\n"
    "        pair_charts[l,k]=items\n"
    "    snapshot=out/'inference_snapshot'; snapshot.mkdir()\n")
sub("            hooks.charts={l:charts[l,k] for l in group}\n",
    "            hooks.charts={l:(charts[l,k] if arm=='untouched' else pair_charts[l,k]) for l in group}\n")
(D / "run_pair_holdout.py").write_bytes(s.encode())
print("wrote", D / "run_pair_holdout.py")
