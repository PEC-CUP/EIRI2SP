"""Construction of a virtual complex mixture (the single definition).

A virtual complex mixture is defined by exactly two things:

  1. **co-elution** - it is a chromatogram, sliced into narrow elution windows,
     so every "spectrum" is a mixture, not a pure compound;
  2. **abundance structure** - the families are present in the proportions a
     real system shows, not in the proportions of the spectral library (the
     library is aromatic-heavy, a real middle distillate is paraffin/naphthene-
     heavy, and the olefins sit at a few percent).

Four mutually exclusive families
    paraffin    DBE 0
    naphthene   carries an aliphatic ring, no aromatic ring
    aromatic    carries an aromatic ring
    olefinic    DBE >= 1 and no ring (olefins, dienes, enynes - the unsaturated
                acyclic part of the library, kept at a few percent)

Design (the **frozen** production definition; ``PER_SELECTION_BELLS`` switches on
the 2026-10-11 variant described at that flag)
    * **20 selections = 5 aromatic levels x 4 chain/naphthene splits**, the
      aromatic level on the inner loop (so selection ``j`` has
      ``AROMATIC_LEVELS[j % 5]`` and ``PARAFFIN_FRACTIONS[j // 5]``);
    * the mass sits on **one fixed smooth bell** over the retention index
      (centre = the middle of the working window, sigma = 1000 RI), so every
      selection covers the same 860-3100 window and the suite varies in
      composition, not in boiling-range shape (T90-T10 = 248 +- 13 K);
    * the **family composition stays a complete 5 x 4 grid** (5 aromatic levels
      x 4 paraffin fractions of the non-aromatic share), spread over the 20
      bells as a bijection (``aromatic = j mod 5``, ``paraffin = 3j mod 4``), so
      "heavy" is not confounded with "aromatic-rich" and every composition cell
      appears exactly once;
    * the olefinic share of each selection is drawn from 0.02-0.10 (a lightly
      hydrotreated middle distillate);
    * 124 molecules per selection, the family counts follow the mass fractions;
    * every molecule's weight is ``family_mass_fraction / family_count``, so the
      family lands on its target mass share exactly.

``select_virtual_mixture`` returns a frame in the same contract as
``select_virtual_mixture_uniform`` (``_dbe`` / ``_class_frac`` / ``_w``) so that
``build_slices`` can be reused unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_ri_and_substructure_contribution import (              # noqa: E402
    AROMATIC_BITS, COMMON_RINGS)

RI_LO, RI_HI = 860.0, 3100.0
N_TOTAL = 124

# 5 aromatic levels x 4 paraffin fractions (of the non-aromatic share)
AROMATIC_LEVELS = (0.08, 0.15, 0.22, 0.28, 0.35)
PARAFFIN_FRACTIONS = (0.30, 0.45, 0.60, 0.75)
UNSAT_RANGE = (0.02, 0.10)          # the olefinic (acyclic unsaturated) family

# Two separate roles for the bell, and they are NOT changed together:
#
#   test bed      (``select_virtual_mixture``)  every selection has its own bell
#                 on a light -> heavy ladder (PER_SELECTION_BELLS = True)
#   augmentation  (``draw_local_windows``)      one FIXED smooth bell over the
#                 whole working window, exactly as the frozen translator was
#                 trained: the augmentation is a *local* window (+-40 RI), its
#                 anchor marginal is not tied to the test-bed design, so
#                 changing the test bed needs no re-training.
#
# Setting PER_SELECTION_BELLS back to False reproduces the archived 100-mixture
# suite (one fixed bell, T50 243-327 C, T90-T10 = 248 +- 13 K).
PER_SELECTION_BELLS = True
N_COMPOSITIONS = len(AROMATIC_LEVELS) * len(PARAFFIN_FRACTIONS)    # 20 cells
N_SELECTIONS = 2 * N_COMPOSITIONS                                  # 40 selections
N_TRAIN_CELLS = N_COMPOSITIONS          # the augmentation sampler's cell grid
BELL_CENTRE_RANGE = (1150.0, 2600.0)
BELL_SIGMA_RANGE = (280.0, 620.0)
BELL_SIGMA = 1000.0                 # the frozen bell width

FAMILY_NAMES = ("paraffin", "naphthene", "aromatic", "olefinic")
FAMILY_LABELS = ("paraffins", "naphthenes", "aromatics",
                 "olefins + alkynes")


def family_of(data: pd.DataFrame) -> np.ndarray:
    """0 paraffin / 1 naphthene / 2 aromatic / 3 olefinic (acyclic unsaturated)."""
    dbe = np.clip(np.rint(data["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    aro = data[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    alip = data[[c for c in data.columns
                 if c.startswith("ring_aliphatic_")]].to_numpy(float).sum(1) > 0
    return np.where(aro, 2, np.where(alip, 1, np.where(dbe > 0, 3, 0)))


def bell_of_cell(j: int) -> tuple[str, float, float]:
    """(name, centre, sigma) of test-bed selection ``j``.

    ``PER_SELECTION_BELLS = True``: a light -> heavy ladder over the 40
    selections.  ``False``: the frozen single bell used by the archived suite.
    """
    if not PER_SELECTION_BELLS:
        return "fixed", 0.5 * (RI_LO + RI_HI), BELL_SIGMA
    f = (int(j) % N_SELECTIONS) / max(1, N_SELECTIONS - 1)
    centre = BELL_CENTRE_RANGE[0] + f * (BELL_CENTRE_RANGE[1] - BELL_CENTRE_RANGE[0])
    sigma = BELL_SIGMA_RANGE[0] + f * (BELL_SIGMA_RANGE[1] - BELL_SIGMA_RANGE[0])
    name = "light" if f < 1 / 3 else ("mid" if f < 2 / 3 else "heavy")
    return name, float(centre), float(sigma)


def composition_of_cell(j: int) -> tuple[float, float]:
    """(aromatic level, paraffin fraction) of composition cell ``j`` (0-19)."""
    j = int(j) % N_COMPOSITIONS
    return (AROMATIC_LEVELS[j % len(AROMATIC_LEVELS)],
            PARAFFIN_FRACTIONS[j // len(AROMATIC_LEVELS)])


def selection_cell(j: int) -> int:
    """Composition cell (0-19) of test-bed selection ``j`` (0-39).

    Selections 0-19 are the *light* half of the bell ladder, 20-39 the *heavy*
    half, and both halves walk the same 20 composition cells.  Every composition
    therefore appears exactly once in each half, so composition and boiling
    range are balanced (a 2 x 20 layout) instead of being confounded.
    """
    return int(j) % N_COMPOSITIONS


def design_fractions(seed: int) -> tuple[np.ndarray, dict]:
    """Target mass fractions (paraffin, naphthene, aromatic, olefinic)."""
    i = int(seed) % N_SELECTIONS
    aro, f_par = composition_of_cell(selection_cell(i))
    rng = np.random.default_rng(seed)
    unsat = float(rng.uniform(*UNSAT_RANGE))
    rest = 1.0 - unsat - aro
    return np.array([f_par * rest, (1.0 - f_par) * rest, aro, unsat]), {
        "design_index": i, "target_aromatic": aro,
        "target_paraffin_fraction": f_par, "target_olefinic": unsat,
        "bell_class": bell_of_cell(i)[0], "composition_cell": selection_cell(i)}


def fractions_from_index(i: int, ole: float) -> np.ndarray:
    """Family mass fractions of composition cell ``i`` (0-19), olefinic share ``ole``."""
    aro, f_par = composition_of_cell(i)
    rest = 1.0 - ole - aro
    return np.array([f_par * rest, (1.0 - f_par) * rest, aro, ole])


def draw_local_components(data: pd.DataFrame, rng: np.random.Generator, n: int,
                          *, allowed: np.ndarray | None = None, n_comp: int = 3,
                          ri_lo: float = RI_LO, ri_hi: float = RI_HI,
                          window: float = 40.0,
                          bell_sigma: float = BELL_SIGMA,
                          min_candidates: int | None = None):
    """The component draw of one co-eluting window (no spectra).

    This is *the* definition shared by

      * ``draw_local_windows`` -- the translator's augmentation (also its test bed);
      * ``mixture_augmented_models.mixture_pairs`` -- the mixture-augmented
        **property** models,

    so the two layers can no longer drift apart.  Returns
    ``(idx [n, n_comp], weights [n, n_comp], anchors [n])``.

      * anchor RI ~ N(centre of the working window, ``bell_sigma``), clipped;
      * candidates = pool molecules inside +-``window`` RI (``allowed`` restricts
        the pool, e.g. to the training compounds);
      * the components are drawn **without replacement** with probability
        proportional to one of the 20 composition cells (5 aromatic levels x 4
        paraffin fractions of the saturate part, olefinic share 0.02-0.10);
      * masses ~ Dirichlet(1,...,1).

    ``min_candidates`` rejection-samples the anchor so that the +-``window``
    neighbourhood really holds that many pool molecules.
    """
    fam = family_of(data)
    ri = data["desc_airi_ri"].to_numpy(float)
    base = (ri >= ri_lo) & (ri <= ri_hi)
    if allowed is not None:
        base = base & np.asarray(allowed, bool)
    pool = np.where(base)[0]
    # one FIXED bell over the working window + the 20 composition cells: the same
    # definition (and the same RNG order) that produced ``augmented_z_model_split.joblib``
    n_cells = N_TRAIN_CELLS
    centre = 0.5 * (ri_lo + ri_hi)
    idx = np.zeros((n, n_comp), int)
    weights = np.zeros((n, n_comp))
    anchors = np.zeros(n)
    for i in range(n):
        a = float(np.clip(rng.normal(centre, bell_sigma), ri_lo, ri_hi))
        cand = pool[np.abs(ri[pool] - a) <= window]
        if min_candidates:                     # reject anchors whose window is
            tries = 0                          # too sparse (keeps +-window true)
            while len(cand) < min_candidates and tries < 200:
                a = float(np.clip(rng.normal(centre, bell_sigma), ri_lo, ri_hi))
                cand = pool[np.abs(ri[pool] - a) <= window]
                tries += 1
        anchors[i] = a
        if len(cand) < n_comp:
            cand = pool[np.argsort(np.abs(ri[pool] - a))[: max(n_comp, 20)]]
        frac = fractions_from_index(int(rng.integers(n_cells)),
                                    float(rng.uniform(*UNSAT_RANGE)))
        p = frac[fam[cand]]
        p = p / p.sum()
        idx[i] = rng.choice(cand, size=n_comp, replace=False, p=p)
        weights[i] = rng.dirichlet(np.ones(n_comp))
    return idx, weights, anchors


def draw_local_windows(data: pd.DataFrame, spec: list[str], zcols: list[str],
                       n: int, rng: np.random.Generator,
                       allowed: np.ndarray | None = None, n_comp: int = 3,
                       ri_lo: float = RI_LO, ri_hi: float = RI_HI,
                       window: float = 40.0,
                       bell_sigma: float = BELL_SIGMA,
                       min_candidates: int | None = None,
                       return_anchor: bool = False):
    """Draw ``n`` co-eluting spectra as **local windows of a virtual mixture**.

    A co-eluting spectrum is not an arbitrary mixture: it is one elution window
    of the same object the virtual complex mixtures describe.  So it is built
    with exactly the same design:

      * each spectrum picks one of the 20 design cells and takes **that cell's**
        sampling bell for its anchor RI and that cell's family composition
        (5 aromatic levels x 4 paraffin fractions, olefinic share 0.02-0.10);
      * the ``n_comp`` components are drawn inside +-``window`` RI with
        probability proportional to that composition (renormalised over what
        the window offers, as co-elution is local);
      * masses ~ Dirichlet(1..1) and the spectrum is synthesised exactly as in
        production (mass, TIC normalised).

    ``min_candidates`` makes the anchor **rejection-sampled**: an anchor is
    accepted only if the +-``window`` RI neighbourhood holds at least that many
    pool molecules.  Without it a large ``n_comp`` silently falls back to "the
    ``n_comp`` nearest molecules", which widens the window to >100 RI.

    Returns ``(X, Y, idx, w)`` with the same layout as
    :func:`mixture_sampling.build_mixture_xy`.
    """
        from z_fingerprint_model import sum_normalize         # local: avoids a cycle

    cf = np.ones(len(data))
    lib = sum_normalize(data[spec].to_numpy(float))
    Z = data[zcols].to_numpy(float)
    ri = data["desc_airi_ri"].to_numpy(float)
    # one definition, shared with the mixture-augmented property models
    idx, weights, anchors = draw_local_components(
        data, rng, n, allowed=allowed, n_comp=n_comp, ri_lo=ri_lo, ri_hi=ri_hi,
        window=window, bell_sigma=bell_sigma, min_candidates=min_candidates)
    contrib = np.einsum("ij,ijk->ik", weights / cf[idx], lib[idx])
    X = np.empty((n, len(spec) + 1))
    X[:, : len(spec)] = contrib / contrib.sum(1, keepdims=True)
    X[:, len(spec)] = (weights * ri[idx]).sum(1)
    Y = np.einsum("ij,ijk->ik", weights, Z[idx])
    if return_anchor:
        return X, Y, idx, weights, anchors
    return X, Y, idx, weights


def select_virtual_mixture(data: pd.DataFrame, seed: int = 2026,
                           n_total: int = N_TOTAL,
                           allowed: np.ndarray | None = None,
                           ri_lo: float = RI_LO,
                           ri_hi: float = RI_HI,
                           jitter: float = 0.0,
                           bell_center: float | None = None,
                           bell_sigma: float = BELL_SIGMA) -> pd.DataFrame:
    """Pick one virtual complex mixture (a compound selection with weights)."""
    rng = np.random.default_rng(seed)
    fam = family_of(data)
    ri = data["desc_airi_ri"].to_numpy(float)
    base = (ri >= ri_lo) & (ri <= ri_hi)
    if allowed is not None:
        base = base & np.asarray(allowed, bool)
    base_idx = np.where(base)[0]
    if len(base_idx) < n_total:
        raise ValueError(f"only {len(base_idx)} candidates inside the RI window")

    frac, meta = design_fractions(seed)
    if jitter:
        frac = np.clip(frac + rng.normal(0.0, jitter, len(frac)), 0.005, None)
        frac = frac / frac.sum()

    # smooth bell over the retention index (database property, not a sample);
    # this selection's own bell, light -> heavy across the 20 selections
    if bell_center is None:
        bell_center = bell_of_cell(meta["design_index"])[1]
        bell_sigma = bell_of_cell(meta["design_index"])[2]
    bell = np.exp(-0.5 * ((ri[base_idx] - bell_center) / bell_sigma) ** 2)
    bell = np.clip(bell, 1e-6, None)
    meta.update({"bell_center": bell_center, "bell_sigma": bell_sigma})

    counts = [max(1, int(round(frac[k] * n_total))) for k in range(len(frac))]
    counts[int(np.argmax(counts))] += n_total - sum(counts)
    picked: list[int] = []
    for k in range(len(frac)):
        pool = base_idx[fam[base_idx] == k]
        n_k = min(counts[k], len(pool))
        if n_k <= 0:
            continue
        p = bell[np.searchsorted(base_idx, pool)]
        p = p / p.sum()
        picked.extend(int(i) for i in rng.choice(pool, size=n_k, replace=False,
                                                 p=p))
    out = data.iloc[sorted(set(picked))].copy()
    fam_o = family_of(out)
    n_fam = np.array([(fam_o == k).sum() for k in range(len(frac))], float)
    # every molecule carries its family's target mass divided by the family size
    w = np.array([frac[k] for k in fam_o], float) / np.maximum(n_fam[fam_o], 1)
    out["_family"] = fam_o
    out["_w"] = w / w.sum()
    out["_target_fractions"] = [tuple(frac)] * len(out)
    out["_dbe"] = np.clip(np.rint(out["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    cnt = out.groupby("_dbe")["identifier"].transform("count").to_numpy(float)
    out["_class_frac"] = cnt * out["_w"].to_numpy(float)
    meta["fractions"] = tuple(frac)
    meta["seed"] = int(seed)
    out.attrs["design"] = meta
    return out


# the five abundance settings: the *same* compound set, re-weighted so the
# aromatic share moves and the paraffin share moves the other way
ABUNDANCE_FACTORS = (0.75, 0.90, 1.00, 1.12, 1.25)


def apply_abundance(sel: pd.DataFrame, setting: int) -> pd.DataFrame:
    """Re-weight one selection (``setting`` = 1..5) without changing its molecules."""
    out = sel.copy()
    frac = np.array(sel.attrs["design"]["fractions"], float)
    k = ABUNDANCE_FACTORS[setting - 1]
    f = frac.copy()
    f[2] *= k                      # aromatic up ...
    f[0] /= k                      # ... paraffin down (the naphthene share absorbs)
    f = f / f.sum()
    fam_o = out["_family"].to_numpy(int)
    n_fam = np.array([(fam_o == j).sum() for j in range(4)], float)
    w = np.array([f[j] for j in fam_o], float) / np.maximum(n_fam[fam_o], 1)
    out["_w"] = w / w.sum()
    cnt = out.groupby("_dbe")["identifier"].transform("count").to_numpy(float)
    out["_class_frac"] = cnt * out["_w"].to_numpy(float)
    out.attrs["design"] = dict(sel.attrs["design"], setting=setting,
                               setting_fractions=tuple(f))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(BASE / "src/model_training"))
    from predict_mixture import load_training_data

    d, _s, _z, _p = load_training_data()
    for s in (3000, 3001):
        sel = select_virtual_mixture(d, seed=s)
        got = sel.groupby("_family")["_w"].sum()
        print(f"seed {s}: {len(sel)} compounds; "
              + "  ".join(f"{FAMILY_NAMES[k]} {got.get(k, 0.0):.3f}"
                          for k in range(4)))
