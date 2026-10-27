#!/usr/bin/env python
"""Rule-based mapping from the fingerprint bits to the SH/T 0606 type classes.

The 27 structural bits describe the parent ring systems; the SH/T 0606 table is
organised by those same structural types.  This script applies an explicit
*rule* (no extra trained fingerprint) that distributes the slice's mass over
the eleven type classes using

  * desc_is_chain_alkane                     -> paraffins
  * desc_is_aromatic                         -> total aromatics
  * the aromatic ring-system bits            -> alkylbenzene /
                                                indane-tetralin / naphthalene /
                                                tricyclic aromatics
  * the aliphatic ring-system bits + ring count -> 1-/2-/3-ring naphthenes

and compares it with the class composition computed from the true structures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_mixture import load_training_data                        # noqa: E402
from z_fingerprint_model import sum_normalize                          # noqa: E402
from validate_ri_and_substructure_contribution import (                # noqa: E402
    AROMATIC_BITS, build_slices)
from select_virtual_mixture_uniform import select_virtual_mixture_uniform  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "models" / "simulated_mixture_validation"
NAMES = ["paraffins", "1-ring naphthene", "2-ring naphthene", "3-ring naphthene",
         "alkylbenzene", "indane/tetralin", "indene", "naphthalene",
         "acenaphthene", "acenaphthylene", "tricyclic aromatic"]

# SH/T 0606 worked example (used as the reference for the real sample)
SH0606 = [39.7, 23.5, 11.6, 3.2, 8.4, 7.8, 2.7, 1.4, 0.6, 1.0, 0.1]

# aromatic ring-system bits -> SH/T 0606 aromatic class
ARO_CLASS = {
    "ring_aromatic_6_84480431": 4,      # benzene          -> alkylbenzene
    "ring_aromatic_9_afce84c1": 5,      # indane
    "ring_aromatic_5_9eb77984": 5,      # arene + 5-ring
    "ring_aromatic_10_c1d2687b": 5,     # tetralin
    "ring_aromatic_6_8b792288": 5,      # arene + 6-ring
    "ring_aromatic_10_01f31866": 7,     # naphthalene
    "ring_aromatic_14_541c53d1": 10,    # phenanthrene
    "ring_aromatic_14_a1e6ec18": 10,    # anthracene
    "ring_aromatic_18_1622d282": 10,    # [18]annulene
}


def true_classes(df: pd.DataFrame) -> np.ndarray:
    """Mass-weighted class composition computed from the true structures."""
    mw = 12.011 * df["desc_C"].to_numpy(float) + 1.008 * df["desc_H"].to_numpy(float)
    w = df["_w"].to_numpy(float)
    w = w / w.sum()
    dbe = np.clip(np.rint(df["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    aro = df[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    chain = df["desc_is_chain_alkane"].to_numpy(float) > 0.5
    out = np.zeros(11)
    for i in range(len(df)):
        if chain[i]:
            k = 0
        elif not aro[i]:
            k = int(min(max(round(df["desc_ring_count"].to_numpy(float)[i]), 1), 3))
        else:
            k = int(dbe[i])
            if k < 4:
                k = 4
        out[k] += w[i]
    return out


def _soft_ring_split(out: np.ndarray, base: float, rc: float) -> None:
    """Place ``base`` mass over the 1-/2-/3-ring naphthene classes (indices 1-3).

    ``rc`` is the average number of *aliphatic* rings carried by the naphthenic
    part of the slice; it is expanded linearly between the two neighbouring
    integer ring counts so a mixed slice is not forced into a single class.
    """
    lo = int(np.clip(np.floor(rc), 1, 3))
    hi = int(np.clip(np.ceil(rc), 1, 3))
    if lo == hi:
        out[lo] += base
    else:
        w_hi = rc - lo
        out[hi] += base * w_hi
        out[lo] += base * (1.0 - w_hi)


def rule_classes(z: pd.Series) -> np.ndarray:
    """Rule-based class composition from one slice's fingerprint bits."""
    # paraffins = ACYCLIC SATURATED, i.e. n-alkanes *and* branched alkanes.  Until
    # 2026-10-14 this read only the n-alkane flag, which under-counted the paraffin
    # class (and therefore over-counted naphthenes) -- fixed here; the numbers it
    # produced earlier (e.g. figures/58) must be regenerated.
    # "acyclic saturated" = n-alkanes AND branched alkanes.  The two flags are two
    # pieces of evidence about the SAME class and are not exclusive in the model output,
    # so the class mass fraction is their max, not their sum (summing double-counts).
    chain = max(float(np.clip(z["desc_is_chain_alkane"], 0, 1)),
                float(np.clip(z["desc_is_isoalkane"], 0, 1)))
    arom = float(np.clip(z["desc_is_aromatic"], 0, 1))
    naph = max(0.0, 1.0 - chain - arom)
    out = np.zeros(11)
    out[0] = chain
    # aromatic split first (these bits are the parent-structure descriptors)
    f5 = sum(float(np.clip(z[c], 0, 1)) for c in
             ["ring_aromatic_9_afce84c1", "ring_aromatic_5_9eb77984",
              "ring_aromatic_10_c1d2687b", "ring_aromatic_6_8b792288"])
    f7 = float(np.clip(z["ring_aromatic_10_01f31866"], 0, 1))
    f10 = sum(float(np.clip(z[c], 0, 1)) for c in
              ["ring_aromatic_14_541c53d1", "ring_aromatic_14_a1e6ec18",
               "ring_aromatic_18_1622d282"])
    f5, f7, f10 = (min(v, arom) for v in (f5, f7, f10))
    out[5], out[7], out[10] = f5, f7, f10
    out[4] = max(0.0, arom - f5 - f7 - f10)
    # naphthene split: the average *aliphatic* ring count carried by the
    # naphthenic part of the slice -> 1-/2-/3-ring classes
    rings = float(z["desc_ring_count"])
    aro_rings = float(z.get("desc_aromatic_ring_count_estimate", 0.0))
    alip_rings = max(0.0, rings - aro_rings)
    if naph > 0:
        _soft_ring_split(out, naph, alip_rings / naph)
    s = out.sum()
    return out / s if s > 0 else out


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    # the production translator of the frozen 80/20 protocol: 4,180 training
    # compounds plus 30,000 co-eluting spectra built from them
    zm = joblib.load(OUT / "augmented_z_model_split.joblib")

    T, R = [], []
    for s_i in range(4):
        sel = select_virtual_mixture_uniform(data, seed=3000 + s_i)
        t = true_classes(sel) * 100
        rng = np.random.default_rng(7000 + 100 * s_i)
        for a in range(5):
            vm, _ = build_slices(sel, data, spec, rng)
            Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
            Zv = pd.DataFrame(np.clip(zm.predict(Xv), 0, 1), columns=zcols)
            aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
            rule = np.array([rule_classes(Zv.iloc[i]) for i in range(len(Zv))])
            T.append(t)
            R.append((rule * aw[:, None]).sum(0) * 100)
    T, R = np.array(T), np.array(R)
    print(f"  {'class':<20} {'true %':>8} {'rule %':>8}")
    for k in range(11):
        print(f"  {NAMES[k]:<20} {T[:, k].mean():8.2f} {R[:, k].mean():8.2f}")
    mae11_rule = float(np.abs(R - T).mean())
    # three-class level, which the motif fingerprint supports directly
    def tri(A):
        return np.vstack([A[:, 0], A[:, 1:4].sum(1), A[:, 4:].sum(1)]).T
    tri_t, tri_r = tri(T), tri(R)
    mae3_rule = float(np.abs(tri_r - tri_t).mean())
    print(f"\n  11-class MAE: rule-based (motif bits) {mae11_rule:.2f} pp")
    print(f"  3-class  MAE (paraffin / naphthene / aromatic): rule-based "
          f"{mae3_rule:.2f} pp")

    out = pd.DataFrame({
        "class": NAMES,
        "true_pct": T.mean(0),
        "rule_pct": R.mean(0),
    })
    out.to_csv(OUT / "rule_based_type_mapping.csv", index=False)
    pd.DataFrame({
        "level": ["11-class", "3-class"],
        "rule_MAE_pp": [mae11_rule, mae3_rule],
    }).to_csv(OUT / "rule_based_type_mapping_mae.csv", index=False)
    print("  saved: rule_based_type_mapping.csv / _mae.csv")

    # ---------------- real mixture (245 slices) ----------------
    # quantitative areas come from the GC-FID trace of the same sample
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules_fid.csv")
    Xr = np.hstack([sum_normalize(real[spec].to_numpy(float)),
                    real["ri"].to_numpy(float)[:, None]])
    Zraw_r = zm.predict(Xr)
    Zr = pd.DataFrame(np.clip(Zraw_r, 0, 1), columns=zcols)   # 0/1 bits only
    Zcont = pd.DataFrame(Zraw_r, columns=zcols)               # continuous kept raw
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    r_cls = np.array([rule_classes(Zr.iloc[i]) for i in range(len(Zr))])
    real_pct = (r_cls * rw[:, None]).sum(0) * 100
    print("\n  real mixture (245 slices), rule-based SH/T 0606 composition:")
    print(f"  {'class':<20} {'rule %':>8} {'SH/T0606':>9}")
    for k in range(11):
        print(f"  {NAMES[k]:<20} {real_pct[k]:8.2f} {SH0606[k]:9.2f}")
    ref = np.array(SH0606)
    print(f"  11-class MAE vs SH/T 0606: {np.abs(real_pct - ref).mean():.2f} pp")
    tri_r = np.array([real_pct[0], real_pct[1:4].sum(), real_pct[4:].sum()])
    tri_e = np.array([ref[0], ref[1:4].sum(), ref[4:].sum()])
    print(f"  3-class  MAE vs SH/T 0606: {np.abs(tri_r - tri_e).mean():.2f} pp "
          f"(rule {tri_r.round(1).tolist()} vs ref {tri_e.round(1).tolist()})")
    pd.DataFrame({"class": NAMES, "rule_pct": real_pct, "SH0606_pct": ref}).to_csv(
        OUT / "rule_based_type_mapping_real.csv", index=False)
    print("  saved: rule_based_type_mapping_real.csv")

    # area-weighted structural description of the real mixture (C route: the
    # description is read off the translated fingerprint Z, no property model)
    desc_cols = ["desc_C", "desc_H", "desc_DBE", "desc_ring_count"]
    flag_cols = ["desc_is_chain_alkane", "desc_is_aromatic"]
    wmean = {c: float((Zcont[c].to_numpy(float) * rw).sum()) for c in desc_cols}
    wmean.update({c: float((Zr[c].to_numpy(float) * rw).sum()) for c in flag_cols})
    wmean["HC_ratio"] = wmean["desc_H"] / wmean["desc_C"]
    pd.DataFrame([{"quantity": c, "area_weighted_mean": v}
                  for c, v in wmean.items()]).to_csv(
        BASE / "data" / "exp" / "real_mixture_z_description.csv", index=False)
    print("  real mixture, area-weighted fingerprint description:",
          {k: round(v, 3) for k, v in wmean.items()})
    print("  saved: data/exp/real_mixture_z_description.csv")


if __name__ == "__main__":
    main()
