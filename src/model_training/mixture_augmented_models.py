#!/usr/bin/env python
"""Build the mixture-augmented training set for a bulk property model.

For a property that does not mix linearly, a model trained on individual
compounds does not predict the *mixture* value when it is fed the mass-weighted
mean fingerprint.  The fix is to train it on synthetic co-eluting mixtures
whose input is the mean fingerprint and whose target is the value given by the
mixing rule (see ``validate_typed_augmentation`` for the augmentation design).

    X = sum_j w_j Z_j          (mass-weighted mean fingerprint)
    y = rule(values of the components)

``rule``
  "mean"        mass-weighted mean        -> boiling point, mass-specific enthalpy
  "reciprocal"  1/rho = sum w_j / rho_j   -> density

The windows are drawn with ``select_virtual_mixture.draw_local_components`` -- the
**same** sampler (fixed bell, 20 composition cells, family-weighted component draw,
Dirichlet masses) that trains the fingerprint translator, so the two layers of the
pipeline share one definition of a co-eluting window.  (Before 2026-10-13 this module
used the older equal-probability sampler ``mixture_sampling.draw_components``; the two
agree to <= 2 % on every reported metric, see
``src/simulation/compare_property_model_samplers.py``.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from predict_mixture import load_training_data          # noqa: E402
from mixture_sampling import N_COMP, N_MIX, RI_WINDOW   # noqa: E402
from select_virtual_mixture import (                    # noqa: E402
    RI_HI as ANCHOR_HI, RI_LO as ANCHOR_LO, draw_local_components)

RULES = ("mean", "reciprocal", "kay")

__all__ = ["mixture_pairs", "fit_mixture_augmented", "mixture_value"]


def mixture_value(rule: str, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The mixing-rule value of a mixture, from its components.

    ``rule``
      "mean"        mass-fraction weighted mean   -> boiling point, mass-specific
                                                     enthalpy (mass-based quantities)
      "reciprocal"  specific-volume mixing        -> density
      "kay"         Kay's rule: mole-fraction weighted mean -> Tc, Pc, Vc
                    (the same rule as ``mixing_rules.kay_rule`` and as the
                    real-sample pipeline)

    ``w`` and ``v`` are [n, n_comp] arrays: mass fractions and component values.
    For ``kay`` the mole fractions are formed with the component molecular
    weights, which are computed from the fingerprint's C and H counts.
    """
    w = np.asarray(w, float)
    v = np.asarray(v, float)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    if rule == "reciprocal":
        return 1.0 / np.sum(w / np.clip(v, 1e-9, None), axis=1)
    if rule == "kay":
        raise ValueError("kay needs the component molecular weights; use "
                         "mixture_value_kay()")
    if rule == "mean":
        return np.sum(w * v, axis=1)
    raise ValueError("unknown mixing rule %r (use one of %s)" % (rule, RULES))


def mixture_value_kay(w: np.ndarray, v: np.ndarray, mw: np.ndarray) -> np.ndarray:
    """Kay's rule (mole-fraction weighted mean) of one mixture."""
    w = np.asarray(w, float)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    z = w / np.clip(np.asarray(mw, float), 1e-9, None)
    z = z / z.sum(axis=1, keepdims=True)
    return np.sum(z * np.asarray(v, float), axis=1)


def rule_value(rule: str, idx: np.ndarray, w: np.ndarray, vals: np.ndarray,
               mw: np.ndarray) -> np.ndarray:
    """``mixture_value`` / ``mixture_value_kay`` on the components ``idx``."""
    idx = np.asarray(idx)
    v = np.asarray(vals, float)[idx]
    if rule == "kay":
        return mixture_value_kay(w, v, np.asarray(mw, float)[idx])
    return mixture_value(rule, w, v)


def component_mw(data) -> np.ndarray:
    """Component molecular weights from the C/H counts of the library table."""
    return (12.011 * data["desc_C"].to_numpy(float)
            + 1.008 * data["desc_H"].to_numpy(float))


def mixture_pairs(zcols, values, *, rule="mean", n=None, seed=2026,
                  anchor_lo=ANCHOR_LO, anchor_hi=ANCHOR_HI,
                  n_comp=N_COMP, ri_window=RI_WINDOW, allowed_pool=None):
    """Return (X, y) for n synthetic co-eluting mixtures.

    The components come from ``draw_local_components`` -- the same sampler as the
    translator augmentation (20 composition cells, fixed bell over the working
    window, three components inside +-40 RI, Dirichlet masses).  The only
    differences from the translator's windows are (i) the pool is further limited
    to the compounds that carry the property label, and (ii) the input is the
    mass-weighted mean fingerprint instead of the spectrum and the target is the
    mixed property (the mixing rule).
    """
    data, spec, zc, _ = load_training_data()
    data = data.copy()
    assert list(zc) == list(zcols)
    ri = data["desc_airi_ri"].to_numpy(float)
    Z = data[zcols].to_numpy(float)
    n = N_MIX if n is None else n
    # the pool is always limited to compounds that carry the label; an extra
    # mask can restrict it further (used by the molecule-disjoint evaluation,
    # where the components must come from the training molecules only)
    allowed = np.isfinite(values)
    if allowed_pool is not None:
        allowed = allowed & np.asarray(allowed_pool, bool)
    rng = np.random.default_rng(seed)
    idx, w, _anchors = draw_local_components(
        data, rng, n, allowed=allowed, n_comp=n_comp, ri_lo=anchor_lo,
        ri_hi=anchor_hi, window=ri_window)
    X = np.einsum("ij,ijk->ik", w, Z[idx])
    # the pool only contains labelled compounds, so every window is complete
    y = rule_value(rule, idx, w, values, component_mw(data))
    return X, y


def fit_mixture_augmented(zcols, values, *, rule="mean", n=None, seed=2026,
                          allowed_pool=None):
    """Fit an ExtraTrees model on (individual compounds + synthetic mixtures)."""
    from sklearn.ensemble import ExtraTreesRegressor

    data, _spec, _zc, _p = load_training_data()
    ok = np.isfinite(values)
    if allowed_pool is not None:
        ok = ok & np.asarray(allowed_pool, bool)
    Xa, ya = mixture_pairs(zcols, values, rule=rule, n=n, seed=seed,
                           allowed_pool=allowed_pool)
    X = np.vstack([data.loc[ok, zcols].to_numpy(float), Xa])
    y = np.concatenate([values[ok], ya])
    m = ExtraTreesRegressor(n_estimators=120, random_state=2026,
                            min_samples_leaf=2, n_jobs=1)
    m.fit(X, y)
    return m
