#!/usr/bin/env python
"""Co-eluting test bed: how the two frozen translators scale with the overlap.

Every test spectrum is **unseen**: the components come from the 1,045 held-out
molecules (876 of them inside RI 860-3100) and the overlap is redrawn.  For each
spectrum one anchor RI is drawn from the same bell as the training windows and is
**rejection-sampled until the +-40 RI window holds n candidates**; without that
an n=20 draw silently falls back to "the 20 nearest molecules" and the window
widens to ~125 RI.

Why the production setting is n = 3 (see coelution_complexity_anchors.csv):
  * the training windows are three-component windows, and 2-4 co-eluting
    compounds is what a GC-MS window actually contains;
  * from the held-out pool the +-40 RI window holds >= n candidates for 100 % of
    the anchors up to n = 6 (99.7 % at n = 8, 82 % at n = 12, 62 % at n = 20),
    so n <= 8 needs no anchor rejection at all and stays unbiased.

Outputs (models/simulated_mixture_validation/)
  coelution_complexity.csv        per (n, seed, pipeline): fingerprint + property
                                  metrics, raw and normalised by the truth spread
  coelution_bit_error.csv         per (n, seed, pipeline, bit): |error| / range
  coelution_complexity_anchors.csv  anchor availability per n
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data                          # noqa: E402
from z_fingerprint_model import sum_normalize                           # noqa: E402
from split_protocol import split_mask, held_out_mask                    # noqa: E402
from select_virtual_mixture import draw_local_windows                   # noqa: E402
from mixture_augmented_models import (component_mw, rule_value)         # noqa: E402

MD = BASE / "models" / "simulated_mixture_validation"
# every integer, so the figures have no gap at n = 7 and n = 9
NCOMP = list(range(1, 12))
ANCHOR_N = list(range(1, 21))                              # anchor diagnostic
SEEDS = [777, 801, 902]
N_SPEC = 2000
N_ANCHOR_SAMPLE = 20000
RI_LO, RI_HI, BELL_C, BELL_S = 860.0, 3100.0, 1980.0, 1000.0

# the mixing rule of every property, in ONE place (same rules as
# ``mixing_rules.py`` and as the virtual-mixture / real-sample layers):
#   Tb      mass-fraction weighted mean
#   Tc/Pc/Vc Kay's rule (mole-fraction weighted)
#   density specific-volume mixing
PROPS = {
    "Tb": ("chem_Tb_K", "mean"),
    "Tc": ("chem_Tc_K", "kay"),
    "Pc": ("chem_Pc_Pa", "kay"),
    "Vc": ("chem_Vc_m3_mol", "kay"),
    "density": ("density_full_g_ml", "reciprocal"),
    # The mixture molecular weight of this layer is the MASS-WEIGHTED mean of the
    # component molecular weights, because that is the only average the mass-weighted
    # mean fingerprint Z-bar can carry (its desc_C and desc_H are mass-weighted means).
    # The virtual-mixture and real-sample layers use the number average instead, where
    # the composition of every slice is known.  See docs/manuscript_data_and_outline.md.
    # ``MW_full_g_mol`` is the completed molecular-weight label (PubChem value where
    # available, the formula value otherwise); the raw PubChem column has 118 gaps and
    # would put NaN into the window targets.
    "MW": ("MW_full_g_mol", "mean"),
}


def fit_plain(X, y, n_trees=120):
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=n_trees,
                                            random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(X, y)
    return m


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    flags = [c for c in zcols if c.startswith("desc_")
             and set(np.unique(data[c])) <= {0.0, 1.0}]
    motifs = [c for c in zcols if not c.startswith("desc_")]
    cont = [c for c in zcols if c not in flags and c not in motifs]
    use = [zcols.index(c) for c in cont]
    _tr, _te, allowed = split_mask(len(data))
    hold = held_out_mask(allowed)
    ri = data["desc_airi_ri"].to_numpy(float)
    pool = np.where((ri >= RI_LO) & (ri <= RI_HI) & hold)[0]
    Z = data[zcols].to_numpy(float)
    rng_scale = {c: (float(data[c].max()) - float(data[c].min())) or 1.0
                 for c in zcols}

    imp = SimpleImputer()
    imp.fit(np.hstack([data[spec].to_numpy(float), ri[:, None]]))
    pipelines = {
        "individual compounds only": joblib.load(MD / "pure_z_model_split.joblib"),
        "+ co-eluting augmentation":
            joblib.load(MD / "augmented_z_model_split.joblib")}

    # property models: individual-only on the training compounds, and
    # mixture-augmented trained on the production three-component windows
    # (the same sampler the translator uses, see mixture_augmented_models)
    from mixture_augmented_models import fit_mixture_augmented
    prop = {}
    for name, (col, rule) in PROPS.items():
        vals = data[col].to_numpy(float)
        if name == "Pc":                       # report Pc in MPa ...
            vals = vals * 1e-6
        elif name == "Vc":                     # ... and Vc in cm3/mol
            vals = vals * 1e6
        prop[name] = {
            "individual compounds only": fit_plain(Z[allowed], vals[allowed]),
            "+ co-eluting augmentation":
                fit_mixture_augmented(zcols, vals, rule=rule,
                                      allowed_pool=allowed)}
    print("property models fitted")

    mw_comp = component_mw(data)

    def truth(name, idx, w):
        """The mixing-rule value of the window (same rule as the model target)."""
        vals = data[PROPS[name][0]].to_numpy(float)
        if name == "Pc":
            vals = vals * 1e-6
        elif name == "Vc":
            vals = vals * 1e6
        return rule_value(PROPS[name][1], idx, w, vals, mw_comp)

    rows, bit_rows = [], []
    for n in NCOMP:
        for seed in SEEDS:
            X, Y, idx, w, _anchor = draw_local_windows(
                data, spec, zcols, N_SPEC, np.random.default_rng(seed),
                allowed=np.isin(np.arange(len(data)), pool), n_comp=n,
                min_candidates=None if n <= 6 else n, return_anchor=True)
            X = imp.transform(X)
            t_true = {p: truth(p, idx, w) for p in PROPS}
            for tag in pipelines:
                P = pipelines[tag].predict(X)
                r2s = [r2_score(Y[:, j], P[:, j]) for j in use]
                bit_err = {}
                for c in motifs + flags:
                    j = zcols.index(c)
                    bit_err[c] = float(np.abs(P[:, j] - Y[:, j]).mean()
                                       / rng_scale[c])
                    bit_rows.append({
                        "n_components": n, "seed": seed, "pipeline": tag,
                        "bit": c, "kind": "motif" if c in motifs else "flag",
                        "mae": float(np.abs(P[:, j] - Y[:, j]).mean()),
                        "mae_over_range": bit_err[c]})
                row = {"n_components": n, "seed": seed, "pipeline": tag,
                       "cont_R2_mean": float(np.mean(r2s)),
                       "bin_MAE": float(np.mean(
                           [np.abs(P[:, zcols.index(c)] - Y[:, zcols.index(c)]).mean()
                            for c in motifs + flags])),
                       "bin_MAE_over_range": float(np.mean(list(bit_err.values()))),
                       "bin_MAE_over_sd": float(np.mean([
                           np.abs(P[:, zcols.index(c)] - Y[:, zcols.index(c)]).mean()
                           / max(Y[:, zcols.index(c)].std(), 1e-9)
                           for c in motifs + flags])),
                       "motif_MAE": float(np.mean(
                           [np.abs(P[:, zcols.index(c)] - Y[:, zcols.index(c)]).mean()
                            for c in motifs])),
                       "flag_MAE": float(np.mean(
                           [np.abs(P[:, zcols.index(c)] - Y[:, zcols.index(c)]).mean()
                            for c in flags]))}
                for p in PROPS:
                    # the pipeline's own property model ...
                    mae = float(np.abs(prop[p][tag].predict(P) - t_true[p]).mean())
                    row[f"{p}_MAE"] = mae
                    row[f"{p}_MAE_over_sd"] = mae / max(t_true[p].std(), 1e-9)
                    row[f"{p}_sd"] = float(t_true[p].std())
                    # ... and the cross configuration (2x2 factorial: property
                    # model x translator), so the end-to-end gain can be split
                    # into "better fingerprint" and "augmented property model"
                    other = ("+ co-eluting augmentation"
                             if tag == "individual compounds only"
                             else "individual compounds only")
                    row[f"{p}_MAE_cross"] = float(np.abs(
                        prop[p][other].predict(P) - t_true[p]).mean())
                # the second, independent route to the same quantity: read the C and H
                # counts off the predicted mean fingerprint and add the atomic masses.
                # It is drawn as a dashed reference next to the property-model MW.
                _mwc = (12.011 * P[:, zcols.index("desc_C")]
                        + 1.008 * P[:, zcols.index("desc_H")])
                row["MW_count_MAE"] = float(np.abs(_mwc - t_true["MW"]).mean())
                rows.append(row)
            print(f"  n = {n}  seed {seed} done")
    pd.DataFrame(rows).to_csv(MD / "coelution_complexity.csv", index=False)
    pd.DataFrame(bit_rows).to_csv(MD / "coelution_bit_error.csv", index=False)

    # anchor availability (how many bell-drawn anchors can host n components)
    ar = np.clip(np.random.default_rng(2026).normal(BELL_C, BELL_S, N_ANCHOR_SAMPLE),
                 RI_LO, RI_HI)
    r_pool = ri[pool]
    cnt = np.array([np.sum(np.abs(r_pool - a) <= 40.0) for a in ar])
    pd.DataFrame([{"n_components": n,
                   "share_of_anchors_ok": float(np.mean(cnt >= n)),
                   "median_candidates": float(np.median(cnt)),
                   "min_candidates": int(cnt.min())} for n in ANCHOR_N]
                 ).to_csv(MD / "coelution_complexity_anchors.csv", index=False)
    print("saved coelution_complexity.csv / coelution_bit_error.csv / "
          "coelution_complexity_anchors.csv")


if __name__ == "__main__":
    main()
