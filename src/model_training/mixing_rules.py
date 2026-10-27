#!/usr/bin/env python
"""Mixture-level mixing rules for bulk properties.

Reference: Yu Ren et al., Ind. Eng. Chem. Res. 2019, 58, 18742 (molecular
reconstruction of naphtha).  Mass-fraction based rules (the paper's Table 1,
rewritten from mole-fraction form).

  density        : specific-volume mixing   1/rho = sum(w_i / rho_i)
  molecular weight: number-average (paper)  M = 1/sum(w_i / M_i)
                    (we additionally keep the mass-average sum(w_i * M_i))
  critical props  : Kay's rule (mole-fraction weighted)
                    Tc = sum(z_i Tc_i), Pc = sum(z_i Pc_i), Vc = sum(z_i Vc_i)
  cetane number  : linear by volume fraction (volume = w_i / rho_i)
"""

from __future__ import annotations

import numpy as np


def density_mix(w: np.ndarray, rho: np.ndarray) -> float:
    """Mass-fraction specific-volume mixing. w = mass fraction, rho = g/mL."""
    rho = np.clip(rho, 1e-6, None)
    return float(1.0 / np.sum(w / rho))


def mass_to_mole(w: np.ndarray, mw: np.ndarray) -> np.ndarray:
    z = w / np.clip(mw, 1e-9, None)
    return z / z.sum()


def mass_to_volume(w: np.ndarray, rho: np.ndarray) -> np.ndarray:
    v = w / np.clip(rho, 1e-9, None)
    return v / v.sum()


def kay_rule(w: np.ndarray, mw: np.ndarray, Tc: np.ndarray, Pc: np.ndarray, Vc: np.ndarray) -> dict[str, float]:
    z = mass_to_mole(w, mw)
    return {
        "Tc_K": float(np.sum(z * Tc)),
        "Pc_Pa": float(np.sum(z * Pc)),
        "Vc_m3_mol": float(np.sum(z * Vc)),
    }


def cetane_blend_volume(w: np.ndarray, rho: np.ndarray, CN: np.ndarray) -> float:
    """Cetane number blended linearly by volume fraction."""
    v = mass_to_volume(w, rho)
    return float(np.sum(v * CN))
