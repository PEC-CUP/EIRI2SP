"""The frozen protocol shared by every augmentation study.

One definition, used by the translator training, the co-eluting evaluation, the
virtual complex mixtures and the studies behind figures 45/46/48:

  * **molecules**: 80/20 split of the 5,225-compound library, seed 2026
    (4,180 training / 1,045 held-out).  Nothing that is evaluated ever draws a
    component from the held-out set during training.
  * **co-eluting spectra**: `draw_local_windows` -- a local window of the same
    object the virtual complex mixtures describe (smooth bell over the RI
    window, family composition from the same 20-cell design, three components
    inside +-40 RI, Dirichlet masses).

  train beds   individual compounds + co-eluting windows **from the training
               compounds**
  test beds    held-out individual compounds, and co-eluting windows **from the
               held-out compounds** (molecule-disjoint)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from select_virtual_mixture import draw_local_windows                 # noqa: E402

SPLIT_SEED = 2026
MIX_SEED = 2026
TEST_SEED = 777           # co-eluting bed whose components are the training compounds
TEST_SEED_HELDOUT = 778   # molecule-disjoint bed
N_TEST_SPECTRA = 2000


def split_mask(n: int, seed: int = SPLIT_SEED):
    """Return (train_idx, held_out_idx, mask_of_training_compounds)."""
    tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed)
    allowed = np.zeros(n, bool)
    allowed[tr] = True
    return tr, te, allowed


def individual_beds(data, spec, zcols):
    """Individual-compound beds, already imputed.

    Returns ``(tr, te, allowed, imp, Xtr, Ytr, Xte, Yte)``.
    """
    X = np.hstack([data[spec].to_numpy(float),
                   data["desc_airi_ri"].to_numpy(float)[:, None]])
    Y = data[zcols].to_numpy(float)
    tr, te, allowed = split_mask(len(data))
    imp = SimpleImputer().fit(X[tr])
    return tr, te, allowed, imp, imp.transform(X[tr]), Y[tr], \
        imp.transform(X[te]), Y[te]


def coelution_bed(data, spec, zcols, imp, n=N_TEST_SPECTRA,
                  seed: int = MIX_SEED, allowed=None):
    """Co-eluting spectra as local windows; returns (X, Y, component_idx, w).

    ``allowed`` restricts the component pool (training compounds for a training
    bed, the complement for the molecule-disjoint test bed; ``None`` = whole
    library).
    """
    X, Y, idx, w = draw_local_windows(data, spec, zcols, n,
                                     np.random.default_rng(seed),
                                     allowed=allowed)
    return imp.transform(X), Y, idx, w


def held_out_mask(allowed: np.ndarray) -> np.ndarray:
    return ~np.asarray(allowed, bool)
