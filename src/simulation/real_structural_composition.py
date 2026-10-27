"""Structural composition (paraffins / naphthenes / aromatics) of the real samples.

Reference (measured): the carbon-number x DBE grids in ``data/exp/hydrocarbon type``
aggregated as DBE 0 / 1-3 / >= 4; 33# uses the value uploaded earlier
(39.7 / 38.3 / 22.0).

Predicted, two routes over the same slices and the same area weights.  Both are
olefin-free by construction (the samples are hydrotreated middle distillates):

  rule_*    the chemically correct whole-molecule reading:
            paraffins  = p(desc_is_chain_alkane) + p(desc_is_isoalkane)
                         (paraffins = acyclic saturated, i.e. n-alkanes AND branched
                          alkanes -- an earlier version of the project rule counted only
                          the n-alkane flag and therefore under-counted paraffins)
            aromatics  = p(desc_is_aromatic)
            naphthenes = 1 - paraffins - aromatics
  direct_*  the ring-system motif route: the aromatic / aliphatic ring bits decide the
            two ring classes and the rest is paraffinic (residual -> paraffins)

Output: data/exp/data-wt/results/structural_composition.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE / "src"), str(BASE / "src" / "model_training"),
                str(BASE / "src" / "simulation")]
from predict_mixture import load_training_data                      # noqa: E402
from real_samples import SAMPLES, OUT                               # noqa: E402
from evaluate_rule_based_type_mapping import rule_classes           # noqa: E402

MD = BASE / "models" / "simulated_mixture_validation"
HT = BASE / "data" / "exp" / "hydrocarbon type"
data, spec, zcols, _ = load_training_data()
tr = joblib.load(MD / "augmented_z_model_split.joblib")


def dbe3(v):
    r = np.clip(np.rint(v), 0, 10)
    return np.where(r == 0, 0, np.where(r <= 3, 1, 2))


measured = {}
for s in SAMPLES:
    f = HT / ("%s#.xlsx" % s)
    if f.exists():
        d = pd.read_excel(f, header=0)
        d.columns = ["C", "DBE", "I"]
        d = d.apply(pd.to_numeric, errors="coerce").dropna()
        k = dbe3(d.DBE.to_numpy(float))
        v = np.array([d.I[k == j].sum() for j in range(3)], float)
        measured[s] = 100 * v / v.sum()
    else:
        measured[s] = np.array([39.7, 38.3, 22.0])

rows = []
for s in SAMPLES:
    d = pd.read_csv(OUT / ("slices_%s.csv" % s))
    X = np.hstack([d[[c for c in d.columns if c.startswith("mz_")]].to_numpy(float),
                   d.ri.to_numpy(float)[:, None]])
    model_pred = tr.predict(X)
    mass_area = d.area.to_numpy(float)
    w = mass_area / mass_area.sum()
    Z = pd.DataFrame(np.clip(model_pred, 0, 1), columns=zcols)

    # route 1: whole-molecule reading, paraffins = acyclic saturated (chain + branched)
    # the class is "acyclic saturated" (n-alkanes *and* branched alkanes), but the two
    # flags are two pieces of evidence about that ONE class and the model does not keep
    # them exclusive (on real spectra both sit at ~equal, mid values), so the class mass
    # fraction is their max -- summing them double-counts
    par = np.maximum(np.clip(Z["desc_is_chain_alkane"].to_numpy(), 0, 1),
                     np.clip(Z["desc_is_isoalkane"].to_numpy(), 0, 1))
    aro = np.clip(Z["desc_is_aromatic"].to_numpy(), 0, 1)
    # every class from its OWN bit evidence, then normalised to sum to one (2026-10-14):
    # "naphthenes = 1 - paraffins - aromatics" made the naphthene class absorb all the
    # error and inflated it (+8 pp on 33#); the aliphatic-ring bits give the ring mass
    # directly, and the three evidences are then rescaled to 100 %.
    alip = Z[[c for c in zcols if c.startswith("ring_aliphatic")]].to_numpy(float).max(1)
    ev = np.vstack([par, np.clip(alip, 0, 1), aro]).T
    ev = ev / np.clip(ev.sum(1, keepdims=True), 1e-9, None)
    r3 = (ev * w[:, None]).sum(0) * 100

    # route 2: ring-system motif route (aromatic ring bit / aliphatic ring bit)
    ARO = [zcols.index(c) for c in zcols if "ring_aromatic" in c]
    ALI = [zcols.index(c) for c in zcols if "ring_aliphatic" in c]
    a_b = Z.iloc[:, ARO].to_numpy(float).max(1)
    l_b = Z.iloc[:, ALI].to_numpy(float).max(1)
    dd = (np.vstack([np.clip(1 - np.maximum(a_b, l_b), 0, None), l_b, a_b]).T
          * w[:, None]).sum(0) * 100
    dd = 100 * dd / dd.sum()

    rows.append({"sample": s,
                 "meas_paraffins": measured[s][0], "meas_naphthenes": measured[s][1],
                 "meas_aromatics": measured[s][2],
                 "rule_paraffins": r3[0], "rule_naphthenes": r3[1], "rule_aromatics": r3[2],
                 "direct_paraffins": dd[0], "direct_naphthenes": dd[1],
                 "direct_aromatics": dd[2],
                 "MAE_rule": float(np.abs(r3 - measured[s]).mean()),
                 "MAE_direct": float(np.abs(dd - measured[s]).mean())})
t = pd.DataFrame(rows)
pd.set_option("display.width", 230)
print(t.round(1).to_string(index=False))
print("\nmean MAE: rule-based %.1f pp   direct %.1f pp"
      % (t.MAE_rule.mean(), t.MAE_direct.mean()))
t.to_csv(OUT / "structural_composition.csv", index=False)
