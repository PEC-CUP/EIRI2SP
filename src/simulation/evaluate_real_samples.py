"""Real diesel samples (5#, 9#, 33#, 36#): EI+RI -> Z -> composition and properties.

No GC-FID calibration. Slice areas are the GC-MS TIC integrated to a linear
baseline and converted to mass-proportional weights with unit response for all
compound classes. The RI comes from the n-alkane anchors of the GC-MS run.

  per slice   TIC-normalised EI spectrum (m/z 50-500) + RI  ->  Z (46 positions)
  aggregate   response-factor-corrected mass weights: C wt%, H wt%, H/C,
              density, the 27-bit composition, and the distillation curve
  compare     data_list/property.xlsx and data_list/distillation curve.xlsx

Outputs (data/exp/data-wt/results/):
  real_samples_summary.csv     one row per sample: measured vs predicted
  real_samples_curves.csv      the distillation curves (measured + predicted)
  slices_<id>.csv              the slice tables (rt, ri, area, spectrum)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data            # noqa: E402
from split_protocol import split_mask                     # noqa: E402
from mixture_augmented_models import fit_mixture_augmented  # noqa: E402
from real_samples import (SAMPLES, OUT, load_properties,  # noqa: E402
                          load_distillation, slice_sample)

MD = BASE / "models" / "simulated_mixture_validation"
ZCOLS = None
SPEC = None
VIS = None


def fit_plain(X, y):
    m = ExtraTreesRegressor(n_estimators=120, random_state=2026,
                            min_samples_leaf=2, n_jobs=1)
    m.fit(X, y)
    return m


def curve(vals, weights, pct):
    o = np.argsort(vals)
    return np.interp(np.array(pct, float) / 100.0,
                     np.cumsum(weights[o]), vals[o])


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    _tr, _te, allowed = split_mask(len(data))
    Z = data[zcols].to_numpy(float)
    tb_lab = data["chem_Tb_K"].to_numpy(float)
    rho_lab = data["density_full_g_ml"].to_numpy(float)

    translator = joblib.load(MD / "augmented_z_model_split.joblib")
    models = {
        "ind": {"Tb": fit_plain(Z[allowed], tb_lab[allowed]),
                "density": fit_plain(Z[allowed], rho_lab[allowed])},
        "aug": {"Tb": fit_mixture_augmented(zcols, tb_lab, rule="mean",
                                            allowed_pool=allowed),
                "density": fit_mixture_augmented(zcols, rho_lab,
                                                 rule="reciprocal",
                                                 allowed_pool=allowed)},
    }
    print("property models ready")

    props = load_properties()
    rows, curves = [], []
    for s in SAMPLES:
        sl = slice_sample(s)
        X = np.hstack([sl[[c for c in sl.columns if c.startswith("mz_")]]
                       .to_numpy(float), sl.ri.to_numpy(float)[:, None]])
        P = translator.predict(X)
        mass_area = sl.area.to_numpy(float)
        w = mass_area / mass_area.sum()
        C = P[:, zcols.index("desc_C")]
        H = P[:, zcols.index("desc_H")]
        dbe = P[:, zcols.index("desc_DBE")]
        mw_slice = 12.011 * C + 1.008 * H
        mole_weight = w / np.clip(mw_slice, 1e-9, None)
        mole_weight = mole_weight / mole_weight.sum()
        c_wt = float((w * 12.011 * C / mw_slice).sum() * 100)
        h_wt = float((w * 1.008 * H / mw_slice).sum() * 100)
        hc = (h_wt / 1.008) / (c_wt / 12.011)

        rho = {}
        tb = {}
        for tag in ("ind", "aug"):
            r_i = models[tag]["density"].predict(P)
            rho[tag] = float(1.0 / np.sum(w / np.clip(r_i, 1e-6, None)))
            tb[tag] = models[tag]["Tb"].predict(P)

        meas = props.loc[s]
        pct = load_distillation(s).yield_pct.to_numpy(float)
        bp_meas = load_distillation(s).bp_C.to_numpy(float)
        row = {"sample": s, "slices": len(sl), "RI_lo": sl.ri.min(),
               "RI_hi": sl.ri.max(), "C_measured": meas.C, "C_pred": c_wt,
               "H_measured": meas.H, "H_pred": h_wt,
               "HC_measured": meas["H/C"], "HC_pred": hc,
               "C_number_pred": float((mole_weight * C).sum()),
               "DBE_pred": float((mole_weight * dbe).sum()),
               "density_measured": meas.density,
               "density_ind": rho["ind"], "density_aug": rho["aug"]}
        for tag in ("ind", "aug"):
            # the measured curve is in degC, the model outputs K
            tc = curve(tb[tag] - 273.15, w, pct)
            d = tc - bp_meas
            row["dist_MAE_%s" % tag] = float(np.abs(d).mean())
            row["dist_bias_%s" % tag] = float(d.mean())
            row["dist_R2_%s" % tag] = float(r2_score(bp_meas, tc))
            row["T50_meas"] = float(np.interp(50, pct, bp_meas))
            row["T50_%s" % tag] = float(np.interp(50, pct, tc))
            curves += [{"sample": s, "pipeline": tag, "yield_pct": p,
                        "bp_pred_C": v} for p, v in zip(pct, tc)]
        curves += [{"sample": s, "pipeline": "measured", "yield_pct": p,
                    "bp_pred_C": v} for p, v in zip(pct, bp_meas)]
        rows.append(row)
        print("  %-3s done (C %.2f vs %.2f wt%%, density %.4f/%.4f vs %.4f)"
              % (s, c_wt, meas.C, rho["ind"], rho["aug"], meas.density))

    OUT.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "real_samples_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "real_samples_curves.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n", summ.round(3).to_string(index=False))
    print("\nsaved", OUT / "real_samples_summary.csv")


if __name__ == "__main__":
    main()
