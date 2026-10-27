#!/usr/bin/env python
"""Complete the density labels of the 5,225-molecule library.

Two completions were tested on the same held-out molecules (seed-2026 80/20
split, scored on the molecules that carry a library value):

  Rackett from the Joback critical properties (the original completion)
      2,097 library values + 3,130 filled  ->  R2 0.828   REJECTED
      (the Joback Vc/Tc extrapolation breaks down for heavy molecules and gives
      1.4-2.5 g/mL for C>28)

  group-contribution fit of the library values (this script)  ->  R2 0.933
      rho = sum_i c_i * Z_i, a ridge fit on the 46-position fingerprint.  Five-fold
      cross-validated on the 2,088 screened library values: R2 0.925,
      MAE 0.021 g/mL, MAPE 2.4 %.  Used as the label for the 3,139 molecules
      that carry no library value; every molecule that has one keeps it.

Library values outside 0.55-1.25 g/mL (the condensed-phase window of a
hydrocarbon at 298 K, n=9) are treated as missing and refilled.

Output: data/modeling/completed_density_labels.csv
        identifier, density_lib_g_ml, density_gc_g_ml, density_label_g_ml,
        density_label_source
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
from predict_mixture import load_training_data          # noqa: E402

OUT = BASE / "data" / "modeling" / "completed_density_labels.csv"
LO, HI = 0.55, 1.25
SEED = 2026


def ridge() -> Pipeline:
    return Pipeline([("i", SimpleImputer()),
                     ("m", RidgeCV(alphas=np.logspace(-6, 4, 21)))])


def main() -> None:
    data, _spec, zcols, _props = load_training_data()
    Z = data[zcols].to_numpy(float)
    lib = data["fuel_density_g_ml_298K"].to_numpy(float)
    have = np.isfinite(lib) & (lib >= LO) & (lib <= HI)
    print(f"library density values: {int(np.isfinite(lib).sum())} "
          f"-> screened to {int(have.sum())} (dropped {int(np.isfinite(lib).sum() - have.sum())})")

    # cross-validated accuracy of the completion itself
    oof = np.full(len(data), np.nan)
    idx = np.where(have)[0]
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(idx):
        oof[idx[b]] = ridge().fit(Z[idx[a]], lib[idx[a]]).predict(Z[idx[b]])
    m = np.isfinite(oof)
    print(f"completion, 5-fold CV on the library values (n={int(m.sum())}): "
          f"R2 {r2_score(lib[m], oof[m]):.4f}, MAE {np.abs(oof[m] - lib[m]).mean():.4f} g/mL, "
          f"MAPE {(np.abs(oof[m] - lib[m]) / lib[m]).mean() * 100:.2f} %")

    fill = ridge().fit(Z[idx], lib[idx]).predict(Z)
    out = pd.DataFrame({
        "identifier": data["identifier"].to_numpy(),
        "density_lib_g_ml": np.where(have, lib, np.nan),
        "density_gc_g_ml": fill,
    })
    out["density_label_g_ml"] = out["density_lib_g_ml"].fillna(out["density_gc_g_ml"])
    out["density_label_source"] = np.where(out["density_lib_g_ml"].notna(),
                                           "library_thermo", "gc_fit")
    out.to_csv(OUT, index=False)
    print(out["density_label_source"].value_counts().to_string())
    print(f"coverage {out.density_label_g_ml.notna().sum()}/{len(out)}; "
          f"filled range {out.loc[out.density_label_source == 'gc_fit', 'density_label_g_ml'].min():.3f}"
          f"-{out.loc[out.density_label_source == 'gc_fit', 'density_label_g_ml'].max():.3f} g/mL")
    print("saved", OUT)


if __name__ == "__main__":
    main()
