#!/usr/bin/env python
"""Unified per-bit EI+RI -> Z structural fingerprint model.

Z is the 47-dimensional interpretable structure space (20 pure descriptors +
27 substructure bits).

The model below is a single ExtraTrees regressor with a **multi-output** target
(sklearn fits one forest whose splits optimise the criterion averaged over the
47 outputs).  This is the route-C translator and is NOT the same as training one
regressor per bit: multi-output trees share their splits across the outputs, so
per-output R^2 depends on which other outputs are in the same model.  The
genuinely per-bit alternative (route E_ET, one regressor per bit) is implemented
in ``train_qspr_routes.py``; see also
``src/model_training/compare_multioutput_vs_bitwise.py``.

Spectra are TIC (sum) normalized before training and inference so that a
co-eluting mixture spectrum maps approximately to the area-weighted average
structure E[Z].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


MZ_LO = 50
MZ_HI = 500


def mass_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c.startswith("mz_") and MZ_LO <= int(c.split("_")[1]) <= MZ_HI]


def sum_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def fit_z_model(data: pd.DataFrame, zcols: list[str], seed: int = 2026) -> tuple[Pipeline, list[str]]:
    spec = mass_columns(data)
    X = np.hstack([
        sum_normalize(data[spec].to_numpy(float)),
        data["desc_airi_ri"].to_numpy(float)[:, None],
    ])
    model = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", ExtraTreesRegressor(n_estimators=80, random_state=seed, min_samples_leaf=2, n_jobs=1)),
    ])
    model.fit(X, data[zcols].to_numpy(float))
    binary = [c for c in zcols if set(np.unique(data[c].dropna())) <= {0.0, 1.0}]
    return model, binary


def predict_z_model(model: Pipeline, vm: pd.DataFrame, zcols: list[str], binary: list[str]) -> pd.DataFrame:
    vm_spec = vm[[f"mz_{i}" for i in range(MZ_LO, MZ_HI + 1)]].to_numpy(float)
    X = np.hstack([
        sum_normalize(vm_spec),
        vm["ri"].to_numpy(float)[:, None],
    ])
    Z = pd.DataFrame(model.predict(X), columns=zcols)
    for c in binary:
        Z[c] = Z[c].clip(0.0, 1.0)
    return Z
