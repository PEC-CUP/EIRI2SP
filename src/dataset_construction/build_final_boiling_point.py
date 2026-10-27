#!/usr/bin/env python
"""Build the final boiling-point label with a documented, evaluated hierarchy.

Hierarchy:
  1. Experimental: ef049903u SI (matched by canonical SMILES).
  2. chemicals chem_Tb_K, if present and consistent with the reference model.
  3. Reference model:
       - C <  15: Joback
       - C >= 15: calibrated Marrero-Gani, Tb = 1.1392 * mg_Tb - 56.9

Calibration of Marrero-Gani was fitted against the 56 SI-matched systematic
compounds (n-alkanes + n-alkylcyclopentanes/cyclohexanes), giving MAE=4.46 K
and RMSE=6.25 K on those compounds.  chemicals chem_Tb_K on the same set gives
MAE=2.48 K, so it is preferred when it is not an outlier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MG_SLOPE = 1.1392
MG_INTERCEPT = -56.9
OUTLIER_TOL_K = 80.0


def main() -> None:
    base = BASE
    rec = pd.read_csv(base / "data" / "hydrocarbon_records.csv")
    tab = pd.read_csv(base / "data" / "modeling" / "modeling_table.csv")
    joback = pd.read_csv(base / "data" / "modeling" / "joback_properties.csv")
    mg = pd.read_csv(base / "data" / "modeling" / "marrero_gani_properties.csv")
    si = pd.read_csv(base / "data" / "reference" / "si_experimental_Tb_matched.csv")

    d = rec[["identifier", "C", "NAME"]].merge(
        tab[["identifier", "chem_Tb_K"]], on="identifier", how="left"
    ).merge(
        joback[["identifier", "joback_Tb_K"]], on="identifier", how="left"
    ).merge(
        mg[["identifier", "mg_Tb_K"]], on="identifier", how="left"
    ).merge(
        si[["identifier", "si_exp_Tb_K"]], on="identifier", how="left"
    )

    mg_cal = MG_SLOPE * d["mg_Tb_K"].to_numpy(float) + MG_INTERCEPT
    ref = np.where(d["C"].to_numpy(float) < 15.0, d["joback_Tb_K"].to_numpy(float), mg_cal)

    corrected = d["si_exp_Tb_K"].to_numpy(float).copy()
    no_si = np.isnan(corrected)
    chem = d["chem_Tb_K"].to_numpy(float)
    chem_ok = no_si & np.isfinite(chem) & (np.abs(chem - ref) <= OUTLIER_TOL_K)
    corrected[chem_ok] = chem[chem_ok]
    use_ref = no_si & (~np.isfinite(chem) | (np.abs(chem - ref) > OUTLIER_TOL_K))
    corrected[use_ref] = ref[use_ref]

    out = d[["identifier", "C", "NAME"]].copy()
    out["Tb_K_source"] = "reference"
    out.loc[d["si_exp_Tb_K"].notna(), "Tb_K_source"] = "experimental_si"
    out.loc[chem_ok, "Tb_K_source"] = "chemicals_consistent"
    out["Tb_K"] = corrected
    out.to_csv(base / "data" / "modeling" / "corrected_Tb_K.csv", index=False)

    print("source counts:")
    print(out["Tb_K_source"].value_counts().to_string())
    print("Tb_K describe:")
    print(out["Tb_K"].describe().round(1).to_string())

    # Validation on SI-matched compounds (fallback model error, not final error).
    val = out.merge(si[["identifier", "si_exp_Tb_K"]], on="identifier")
    chem_err = (val["Tb_K"] - val["si_exp_Tb_K"]).abs()
    print(f"SI-matched n={len(val)}, corrected-vs-SI MAE={chem_err.mean():.2f} K (should be ~0, SI overrides)")


if __name__ == "__main__":
    main()
