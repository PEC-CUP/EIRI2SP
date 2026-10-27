#!/usr/bin/env python
"""The single definition of a synthetic co-eluting mixture.

Both the fingerprint translator (spectrum + RI -> Z) and the mixture-augmented
property models (mean Z -> mixed property) must draw their training mixtures
from the *same* distribution; only the input representation and the target
differ.  This module is that single definition.

Configuration (2026-10-09, main route)
  * pool          : **the whole library** (5,225 hydrocarbons) - the method is
                    not tailored to one product type; a system-specific pool can
                    be substituted later for an application study (for the
                    property models the pool is further limited to the
                    compounds that carry the label);
  * sampling      : within a +-40 RI window, every candidate is drawn with
                    **equal** probability;
  * anchor        : U(860, 3100)   -- covers the virtual complex mixture
                    (RI 860-2976) and the real sample (859-3135);
  * components    : 3, mass fractions ~ Dirichlet(1,1,1).

The earlier product-matched sampler (a plausible ring pool + class-probability
sampling) is kept as :func:`draw_components_typed` **only to reproduce the
pre-2026-10-10 results**; no production script calls it any more.  It scored
better on test mixtures drawn the same way but worse on a random-inventory
mixture and no better than no augmentation there, i.e. its gain was
distribution matching rather than a better model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from validate_ri_and_substructure_contribution import (  # noqa: E402
    AROMATIC_BITS, COMMON_RINGS, TYPE_TARGET)

ANCHOR_LO, ANCHOR_HI = 860.0, 3100.0
RI_WINDOW = 40.0
N_COMP = 3
N_MIX = 30000


def typed_pool(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the plausible compounds and their clipped DBE."""
    dbe = np.clip(np.rint(data["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    aro = data[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    alip = data[[c for c in data.columns
                 if c.startswith("ring_aliphatic_")]].to_numpy(float).sum(1) > 0
    common = data[COMMON_RINGS].to_numpy(float).sum(1) > 0
    ok = np.zeros(len(data), bool)
    for k in TYPE_TARGET:
        if k == 0:
            ok |= (dbe == 0) & (~aro) & (~alip)
        elif k <= 3:
            ok |= (dbe == k) & (~aro) & alip & common
        else:
            ok |= (dbe == k) & aro & common
    return np.where(ok)[0], dbe


def draw_components(data: pd.DataFrame, rng: np.random.Generator, n: int,
                    allowed: np.ndarray | None = None,
                    anchor_lo: float = ANCHOR_LO, anchor_hi: float = ANCHOR_HI,
                    ri_window: float = RI_WINDOW,
                    n_comp: int = N_COMP) -> tuple[np.ndarray, np.ndarray]:
    """Main route: components drawn with equal probability from the library.

    Return (component indices [n, n_comp], mass fractions [n, n_comp]).
    """
    pool = np.arange(len(data))
    if allowed is not None:
        pool = pool[allowed[pool]]
    ri = data["desc_airi_ri"].to_numpy(float)
    idx = np.zeros((n, n_comp), int)
    w = np.zeros((n, n_comp))
    for i in range(n):
        anchor = rng.uniform(anchor_lo, anchor_hi)
        near = pool[np.abs(ri[pool] - anchor) <= ri_window]
        if len(near) < n_comp:
            near = pool[np.argsort(np.abs(ri[pool] - anchor))[:20]]
        idx[i] = rng.choice(near, size=n_comp, replace=False)
        w[i] = rng.dirichlet(np.ones(n_comp))
    return idx, w


def build_mixture_xy(data: pd.DataFrame, spec: list[str], zcols: list[str],
                     n: int, rng: np.random.Generator, *,
                     normalize_spectra: bool = True,
                     return_components: bool = False,
                     **draw_kw):
    """Draw ``n`` co-eluting spectra and return the (X, Y) training pair.

    ``X`` = [TIC-normalised mixture spectrum, mass-weighted RI];  ``Y`` = the
    mass-weighted mean of the component fingerprints Z.  Both the fingerprint
    translator and (through :func:`draw_components`) the mixture-augmented
    property models use this one definition, so training and evaluation can
    never drift apart.  With ``return_components=True`` the drawn component
    indices and mass fractions are returned as well: ``(X, Y, idx, w)``.
    """
    
    idx, wm = draw_components(data, rng, n, **draw_kw)
    lib = data[spec].to_numpy(float)
    if normalize_spectra:
        lib = lib / lib.sum(1, keepdims=True)
    cf = np.ones(len(data))
    ri = data["desc_airi_ri"].to_numpy(float)
    Z = data[zcols].to_numpy(float)
    contrib = np.einsum("ij,ijk->ik", wm / cf[idx], lib[idx])
    X = np.empty((n, len(spec) + 1))
    X[:, : len(spec)] = contrib / contrib.sum(1, keepdims=True)
    X[:, len(spec)] = (wm * ri[idx]).sum(1)
    Y = np.einsum("ij,ijk->ik", wm, Z[idx])
    if return_components:
        return X, Y, idx, wm
    return X, Y


def draw_components_typed(data: pd.DataFrame, rng: np.random.Generator, n: int,
                          allowed: np.ndarray | None = None,
                          anchor_lo: float = ANCHOR_LO,
                          anchor_hi: float = ANCHOR_HI,
                          ri_window: float = RI_WINDOW,
                          n_comp: int = N_COMP) -> tuple[np.ndarray, np.ndarray]:
    """Legacy product-matched sampler (reproducing old results only)."""
    pool, dbe = typed_pool(data)
    if allowed is not None:
        pool = pool[allowed[pool]]
    ri = data["desc_airi_ri"].to_numpy(float)
    idx = np.zeros((n, n_comp), int)
    w = np.zeros((n, n_comp))
    for i in range(n):
        anchor = rng.uniform(anchor_lo, anchor_hi)
        near = pool[np.abs(ri[pool] - anchor) <= ri_window]
        if len(near) < n_comp:
            near = pool[np.argsort(np.abs(ri[pool] - anchor))[:20]]
        p = np.array([TYPE_TARGET[dbe[j]] for j in near], dtype=float)
        p /= p.sum()
        idx[i] = rng.choice(near, size=n_comp, replace=False, p=p)
        w[i] = rng.dirichlet(np.ones(n_comp))
    return idx, w
