#!/usr/bin/env python
"""Where do RI and the 27 substructure bits contribute downstream?

Three questions, three parts (run one or all):

  A  single compounds : structure recovery (uniqueness) + property ablation
  B  composition      : DBE-Ridge with spectrum only vs spectrum + RI, and
                        aromatic attribution by the descriptor flag vs by the
                        aromatic ring bits (one of the 27)
  C  mixtures         : virtual diesel / real diesel when the Z model is fed
                        EI only vs EI + RI (identical pipeline, same seed)

Every part reuses the production helper functions so the numbers stay
comparable with the paper's tables.

    python -X utf8 src/simulation/validate_ri_and_substructure_contribution.py --part A
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_mixture import load_training_data  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from z_fingerprint_model import sum_normalize  # noqa: E402
from validate_mixture_augmented_training import make_mixtures  # noqa: E402
from validate_virtual_diesel_coelution import select_virtual_diesel  # noqa: E402
from run_type_composition_validations import dbe_one_hot, make_dbe_mixtures  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "models" / "simulated_mixture_validation"

SH0606 = np.array([39.7, 23.5, 11.6, 3.2, 8.4, 7.8, 2.7, 1.4, 0.6, 1.0, 0.1]) / 100.0
DBE_NAMES = ["chain alkane", "1-ring naphthene", "2-ring naphthene", "3-ring naphthene",
             "alkylbenzene", "indane/tetralin", "indene", "naphthalene",
             "acenaphthene", "acenaphthylene", "tricyclic aromatic"]

# the nine aromatic ring bits of the 27-bit block
AROMATIC_BITS = ["ring_aromatic_6_84480431", "ring_aromatic_10_01f31866",
                 "ring_aromatic_14_541c53d1", "ring_aromatic_10_c1d2687b",
                 "ring_aromatic_6_8b792288", "ring_aromatic_14_a1e6ec18",
                 "ring_aromatic_18_1622d282", "ring_aromatic_9_afce84c1",
                 "ring_aromatic_5_9eb77984"]


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    return data, spec, zcols


# --------------------------------------------------------------- part A
def part_a(data, spec, zcols) -> None:
    print("=" * 78)
    print("A. single compounds")
    print("=" * 78)

    def uniq(mat: pd.DataFrame) -> tuple[int, int]:
        key = mat.round(3).astype(str).agg("|".join, axis=1)
        sizes = np.array(list(Counter(key).values()))
        return len(sizes), int(sum(v for v in sizes if v > 1))

    ri = pd.Series(np.round(data["desc_airi_ri"].to_numpy(), 1)).astype(str)
    desc20 = [c for c in zcols if c.startswith("desc_")]
    bits27 = [c for c in zcols if not c.startswith("desc_")]
    variants = {
        "19 descriptors": data[desc20],
        "27 substructure": data[bits27],
        "47 Z": data[zcols],
    }
    print("\nidentification power (unique fingerprints / molecules in a collision group):")
    for k, v in variants.items():
        u, coll = uniq(v)
        print(f"  {k:18s} unique={u:5d}   collided={coll:5d} ({100*coll/len(data):5.1f}%)")
    key47 = data[zcols].round(3).astype(str).agg("|".join, axis=1)
    u, coll = uniq(pd.DataFrame({"v": key47 + "@" + ri}))
    print(f"  {'47 Z + RI':18s} unique={u:5d}   collided={coll:5d} ({100*coll/len(data):5.1f}%)")
    print("  -> the 27 bits add 794 distinguishable molecules, RI adds a further 1412")

    b = pd.read_csv(BASE / "models" / "qspr_routes" / "property_block_ablation.csv", index_col=0)
    print("\nproperty prediction (mean over 12 properties, test R2):")
    print(f"  full 47 Z + RI   {b['All 47 + RI'].mean():.4f}")
    print(f"  full 47 Z        {b['All 47 (no RI)'].mean():.4f}   (RI gain {b['All 47 (no RI)'].mean()-b['All 47 + RI'].mean():+.4f})")
    print(f"  19 descriptors   {b['20 desc only'].mean():.4f}   (27-bit gain over it {b['All 47 (no RI)'].mean()-b['20 desc only'].mean():+.4f})")
    print(f"  27 substructures {b['27 sub only'].mean():.4f}   (RI gain over it {b['27 sub + RI'].mean()-b['27 sub only'].mean():+.4f})")
    print(f"  RI only          {b['RI only'].mean():.4f}")

    z = pd.read_csv(BASE / "models" / "z_routes_evaluation" / "z_bit_metrics.csv")
    z = z[z.route == "EI_RI"].set_index("bit")
    el = pd.read_csv(BASE / "models" / "z_routes_evaluation" / "z_bit_metrics.csv")
    e = el[el.route == "EI"].set_index("bit")
    v = pd.DataFrame({"EI": e.r2.fillna(e.auc), "EI_RI": z.r2.fillna(z.auc)})
    v["kind"] = z["kind"]
    print("\nZ recovery: EI only vs EI+RI")
    for kind in ["continuous", "binary"]:
        s = v[v.kind == kind]
        print(f"  {kind:11s} n={len(s):2d}  EI {s.EI.mean():.3f} -> EI+RI {s.EI_RI.mean():.3f}"
              f"  (delta {s.EI_RI.mean()-s.EI.mean():+.3f})")


# --------------------------------------------------------------- part B
def part_b(data, spec, zcols) -> None:
    print("=" * 78)
    print("B. composition - does RI help, and what do the aromatic bits add?")
    print("=" * 78)

    # ---- B1: aromatic flag vs aromatic ring bits (library level) ----
    flag = data["desc_is_aromatic"].to_numpy(float) == 1.0
    struct = (data[AROMATIC_BITS].to_numpy(float).sum(1) > 0)
    print("\nB1 aromatic attribution on the 5227-molecule library")
    print(f"  desc_is_aromatic = 1                : {flag.sum():5d}")
    print(f"  contains an aromatic ring (27 bits) : {struct.sum():5d}")
    print(f"  flagged but no aromatic ring        : {(flag & ~struct).sum():5d}"
          f"  ({100*(flag & ~struct).sum()/flag.sum():.1f}% of the flagged)")
    print(f"  aromatic ring but not flagged       : {(struct & ~flag).sum():5d}")

    # ---- B2: DBE-Ridge input ablation ----
    rng = np.random.default_rng(2026)
    pure_X = np.hstack([data[spec].to_numpy(float), data["desc_airi_ri"].to_numpy(float)[:, None]])
    pure_Y = dbe_one_hot(data)
    Xm, Ym = make_dbe_mixtures(data, spec, 20000, rng)

    def ridge():
        return Pipeline([("i", SimpleImputer()), ("s", StandardScaler()), ("r", Ridge(alpha=1.0))])

    sel = select_virtual_diesel(data)
    names = DBE_NAMES
    print("\nB2 composition by DBE-one-hot Ridge: spectrum only vs spectrum + RI")
    out_rows = []
    for tag, keep_ri in (("spec", False), ("spec+RI", True)):
        def cut(X):
            return X if keep_ri else X[:, :-1]
        m = ridge()
        m.fit(np.vstack([cut(pure_X), cut(Xm)]), np.vstack([pure_Y, Ym]))
        # virtual diesel, 5 abundance settings
        maes, true_agg, pred_agg = [], [], []
        for _ in range(5):
            vm, _w = build_slices(sel, data, spec, rng)
            Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
            aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
            P = np.clip(m.predict(cut(Xv)), 0, None)
            P = P / P.sum(1, keepdims=True)
            T = vm[[f"_true_dbe_{k}" for k in range(11)]].to_numpy(float)
            maes.append(np.abs(P - T).mean())
            true_agg.append((T * aw[:, None]).sum(0))
            pred_agg.append((P * aw[:, None]).sum(0))
        T = np.array(true_agg).mean(0)
        P = np.array(pred_agg).mean(0)
        maes = float(np.mean(maes))
        agg_mae = float(np.abs(P - T).mean())
        # real diesel
        real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
        Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
        rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
        R = np.clip(m.predict(cut(Xr)), 0, None)
        R = R / R.sum(1, keepdims=True)
        comp = (R * rw[:, None]).sum(0)
        out_rows.append(dict(tag=tag, diesel_slice_mae=maes, diesel_agg_mae=agg_mae,
                             real_mae=float(np.abs(comp - SH0606).mean()),
                             **{f"real_{k}": float(comp[k]) for k in range(11)}))
        print(f"\n  [{tag}]  virtual-diesel DBE class MAE {maes*100:.2f}% (per slice) / "
              f"{agg_mae*100:.2f}% (bulk composition)   "
              f"real-diesel MAE vs SH/T0606 {np.abs(comp-SH0606).mean()*100:.2f}%")
        if tag == "spec+RI":
            print(f"    {'DBE':>3} {'type':<20} {'true':>7} {'SH/T0606':>9} {'pred':>7}")
            for k in range(11):
                print(f"    {k:>3} {names[k]:<20} {T[k]*100:6.1f}% {SH0606[k]*100:8.1f}% {comp[k]*100:6.1f}%")
    pd.DataFrame(out_rows).to_csv(OUT / "ri_contribution_composition.csv", index=False)


def build_slices(sel: pd.DataFrame, data: pd.DataFrame, spec: list[str],
                 rng: np.random.Generator, sigma: float = 8.0, step: float = 5.0,
                 ri_lo: float = 850.0, ri_hi: float = 3100.0):
    """Same slices as run_type_composition_validations.build_diesel_slices.

    The grid spans the whole working retention-index window by default
    (850-3100), so a mixture's slices cover the RI range of its components.
    """
    ri = sel["desc_airi_ri"].to_numpy(float)
    cf = np.ones(len(sel))
    lib = sum_normalize(sel[spec].to_numpy(float))
    dbe = np.clip(np.rint(sel["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    class_frac = sel["_class_frac"].to_numpy(float)
    count = sel.groupby("_dbe")["identifier"].transform("count").to_numpy(float)
    # deterministic slice weights: the design mass fractions themselves
    # (``class_frac / count`` == ``_w``).  The mixture ground truth is computed
    # from the same design fractions, so the spectra and the truth describe the
    # same object; the earlier ``* rng.uniform(0.5, 2.0, ...)`` "within-group
    # abundance jitter" was removed on 2026-10-13 because it put a ~0.64 pp
    # irreducible offset (design vs slice composition) into every reported
    # composition error.  ``rng`` is kept for API compatibility.
    w = class_frac / count
    w = w / w.sum()
    grid = np.arange(ri_lo, ri_hi, step)
    rows = []
    for r in grid:
        g = np.exp(-0.5 * ((r - ri) / sigma) ** 2)
        mass = w * g
        if mass.sum() <= 0:
            continue
        contrib = np.zeros(len(spec))
        for i in range(len(sel)):
            contrib += (mass[i] / cf[i]) * lib[i]
        if contrib.sum() <= 0:
            continue                      # degenerate slice (all weights ~0)
        contrib /= contrib.sum()
        row = {"rt": r / 100.0, "ri": float(r), "area": float(mass.sum())}
        for j, c in enumerate(spec):
            row[c] = contrib[j]
        for k in range(11):
            row[f"_true_dbe_{k}"] = float(mass[dbe == k].sum() / mass.sum())
        rows.append(row)
    return pd.DataFrame(rows), w


# --------------------------------------------------------------- part C
def part_c(data, spec, zcols) -> None:
    print("=" * 78)
    print("C. mixtures - Z model fed EI only vs EI + RI")
    print("=" * 78)
    rng = np.random.default_rng(2026)
    X_mix_tr, Y_mix_tr, _ = make_mixtures(data, spec, zcols, 20000, 3, 40.0, rng)
    X_mix_te, Y_mix_te, _ = make_mixtures(data, spec, zcols, 2000, 3, 40.0, rng)
    pure_X = np.hstack([data[spec].to_numpy(float),
                        data["desc_airi_ri"].to_numpy(float)[:, None]])
    pure_Y = data[zcols].to_numpy(float)

    sel = select_virtual_diesel(data)
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    models, report = {}, {}
    for tag, keep_ri in (("EI", False), ("EI+RI", True)):
        cut = (lambda X: X) if keep_ri else (lambda X: X[:, :-1])
        m = Pipeline([("i", SimpleImputer()),
                      ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                                min_samples_leaf=2, n_jobs=1))])
        m.fit(np.vstack([cut(pure_X), cut(X_mix_tr)]), np.vstack([pure_Y, Y_mix_tr]))
        models[tag] = (m, cut)
        P = m.predict(cut(X_mix_te))
        report[tag] = {"synth_mix_z_mae": float(np.abs(P - Y_mix_te).mean())}
        print(f"  [{tag:5s}] synthetic co-eluting mixture Z MAE {report[tag]['synth_mix_z_mae']:.4f}")

    # virtual diesel: both models must see exactly the same slices
    pos = data.index.get_indexer(sel.index)
    Z_sel = data[zcols].to_numpy(float)[pos]
    comps = {t: [] for t in models}
    zmaes = {t: [] for t in models}
    for _ in range(5):
        vm, w = build_slices(sel, data, spec, rng)
        Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
        aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
        Ztrue = (Z_sel * w[:, None]).sum(0)          # global mass-weighted truth
        for tag, (m, cut) in models.items():
            Z = pd.DataFrame(m.predict(cut(Xv)), columns=zcols)
            zmaes[tag].append(np.abs((Z.to_numpy(float) * aw[:, None]).sum(0) - Ztrue).mean())
            comps[tag].append(broad(Z, aw))
    for tag in models:
        report[tag]["diesel_z_mae"] = float(np.mean(zmaes[tag]))
        report[tag]["diesel_comp"] = {k: float(np.mean([c[k] for c in comps[tag]]))
                                      for k in comps[tag][0]}
    true_comp = true_diesel_composition(sel)
    print("\n  virtual diesel composition (mass fractions):")
    print(f"    {'':14s} {'chain':>7} {'naphthene':>10} {'aromatic':>9} {'avg DBE':>8} {'avg ring':>9}")
    print(f"    {'true':14s} {true_comp['chain']:7.3f} {true_comp['naphthene']:10.3f}"
          f" {true_comp['aromatic']:9.3f} {true_comp['avg_DBE']:8.2f} {true_comp['avg_ring']:9.2f}")
    for tag in models:
        c = report[tag]["diesel_comp"]
        print(f"    {tag:14s} {c['chain']:7.3f} {c['naphthene']:10.3f} {c['aromatic']:9.3f}"
              f" {c['avg_DBE']:8.2f} {c['avg_ring']:9.2f}")

    # real diesel composition
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    print("\n  real diesel composition:")
    print(f"    {'':14s} {'chain':>7} {'naphthene':>10} {'aromatic':>9} {'avg DBE':>8} {'avg ring':>9}")
    print(f"    {'SH/T0606':14s} {SH0606[0]:7.3f} {SH0606[1:4].sum():10.3f}"
          f" {SH0606[4:].sum():9.3f} {'—':>8} {'—':>9}")
    for tag, (m, cut) in models.items():
        Z = pd.DataFrame(m.predict(cut(Xr)), columns=zcols)
        c = broad(Z, rw)
        report[tag]["real_comp"] = c
        print(f"    {tag:14s} {c['chain']:7.3f} {c['naphthene']:10.3f} {c['aromatic']:9.3f}"
              f" {c['avg_DBE']:8.2f} {c['avg_ring']:9.2f}")

    rows = []
    for tag in models:
        r = report[tag]
        rows.append({"variant": tag, "synth_mix_z_mae": r["synth_mix_z_mae"],
                     "diesel_z_mae": r["diesel_z_mae"],
                     **{f"diesel_{k}": v for k, v in r["diesel_comp"].items()},
                     **{f"real_{k}": v for k, v in r["real_comp"].items()}})
    pd.DataFrame(rows).to_csv(OUT / "ri_contribution_mixtures.csv", index=False)
    print("\n  saved:", OUT / "ri_contribution_mixtures.csv")


# --------------------------------------------------------------- part E
MEASURED = {"C_mass_frac": 0.8607, "H_mass_frac": 0.1334,
            "density_g_ml": 0.833}          # 33# hydrotreated diesel


# --------------------------------------------------------------- part F
def part_f(data, spec, zcols) -> None:
    """Diesel properties: Z model fed EI only vs EI+RI (same recipe/seed)."""
    print("=" * 78)
    print("F. diesel properties: EI -> Z vs EI+RI -> Z (property model on Z + RI)")
    print("=" * 78)
    import joblib
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(2026)
    X_mix_tr, Y_mix_tr, _ = make_mixtures(data, spec, zcols, 20000, 3, 40.0, rng)
    pure_X = np.hstack([data[spec].to_numpy(float),
                        data["desc_airi_ri"].to_numpy(float)[:, None]])
    pure_Y = data[zcols].to_numpy(float)
    models = {"EI+RI": (joblib.load(OUT / "augmented_z_model.joblib"), lambda X: X)}
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(np.vstack([pure_X[:, :-1], X_mix_tr[:, :-1]]), np.vstack([pure_Y, Y_mix_tr]))
    models["EI"] = (m, lambda X: X[:, :-1])

    probs = ["PubChem_MolecularWeight", "chem_Tb_K", "density_label_g_ml",
             "chem_Hfus_J_mol", "chem_Hfg_J_mol"]
    train, _ = train_test_split(data, test_size=0.2, random_state=2026)
    pm = {}
    for col in probs:
        tr = train[train[col].notna()]
        pm[col] = Pipeline([("i", SimpleImputer()),
                            ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                                      min_samples_leaf=2, n_jobs=1))]).fit(
            tr[zcols + ["desc_airi_ri"]], tr[col].to_numpy(float))

    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    truth = {c: float((sel[c].to_numpy(float) * sw).sum()) for c in probs}
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    print("\n  virtual diesel, area-weighted mixture property:")
    print(f"    {'property':28s} {'true':>10} {'EI':>10} {'EI+RI':>10}")
    for col in probs:
        vals = {}
        for tag, (zm, cut) in models.items():
            Z = pd.DataFrame(zm.predict(cut(Xv)), columns=zcols)
            Z["desc_airi_ri"] = vm["ri"].to_numpy(float)
            vals[tag] = float(np.sum(pm[col].predict(Z[zcols + ["desc_airi_ri"]]) * aw))
        print(f"    {col:28s} {truth[col]:10.3f} {vals['EI']:10.3f} {vals['EI+RI']:10.3f}")

    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    print("\n  real diesel:")
    print(f"    {'property':28s} {'measured':>10} {'EI':>10} {'EI+RI':>10}")
    for col in probs:
        vals = {}
        for tag, (zm, cut) in models.items():
            Z = pd.DataFrame(zm.predict(cut(Xr)), columns=zcols)
            Z["desc_airi_ri"] = real["ri"].to_numpy(float)
            vals[tag] = float(np.sum(pm[col].predict(Z[zcols + ["desc_airi_ri"]]) * rw))
        meas = MEASURED.get("density_g_ml", np.nan) if col == "density_label_g_ml" else np.nan
        print(f"    {col:28s} {meas:10.3f} {vals['EI']:10.3f} {vals['EI+RI']:10.3f}")
    print("\n  (boiling point of the real sample: D86 T50 = 253.2 C = 526.4 K, T90 = 333.8 C = 607.0 K)")


# --------------------------------------------------------------- part L
MIX_PROPS = ["PubChem_MolecularWeight", "chem_Tb_K", "density_label_g_ml",
             "chem_Hfus_J_mol", "chem_Hfg_J_mol"]


def make_mixtures_with_truth(data, spec, zcols, n, n_comp, ri_window, rng):
    """Like make_mixtures, but also returns the mass-weighted property truth."""
    ri = data["desc_airi_ri"].to_numpy(float)
    cf = np.ones(len(data))
    lib = sum_normalize(data[spec].to_numpy(float))
    Z = data[zcols].to_numpy(float)
    P = data[MIX_PROPS].to_numpy(float)
    X = np.zeros((n, len(spec) + 1))
    Y = np.zeros((n, len(zcols)))
    T = np.zeros((n, len(MIX_PROPS)))
    for i in range(n):
        anchor = rng.uniform(900.0, 2800.0)
        cand = np.where(np.abs(ri - anchor) <= ri_window)[0]
        if len(cand) < n_comp:
            cand = np.argsort(np.abs(ri - anchor))[:max(n_comp, 20)]
        chosen = rng.choice(cand, size=n_comp, replace=False)
        w = rng.dirichlet(np.ones(n_comp))
        contrib = np.zeros(len(spec))
        for j, idx in enumerate(chosen):
            contrib += (w[j] / cf[idx]) * lib[idx]
        contrib /= contrib.sum()
        X[i, : len(spec)] = contrib
        X[i, len(spec)] = float(np.sum(w * ri[chosen]))
        Y[i] = (Z[chosen] * w[:, None]).sum(axis=0)
        T[i] = (P[chosen] * w[:, None]).sum(axis=0)
    return X, Y, T


def part_l(data, spec, zcols) -> None:
    """Co-eluting mixtures: not only Z MAE but also the property error."""
    print("=" * 78)
    print("L. co-eluting mixtures: Z recovery AND property prediction")
    print("=" * 78)
    import joblib
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(2026)
    X_mix_tr, Y_mix_tr, _ = make_mixtures_with_truth(data, spec, zcols, 20000, 3, 40.0, rng)
    X_mix_te, Y_mix_te, T_te = make_mixtures_with_truth(data, spec, zcols, 2000, 3, 40.0, rng)
    pure_X = np.hstack([data[spec].to_numpy(float),
                        data["desc_airi_ri"].to_numpy(float)[:, None]])
    pure_Y = data[zcols].to_numpy(float)
    models = {"EI+RI": (joblib.load(OUT / "augmented_z_model.joblib"), lambda X: X)}
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(np.vstack([pure_X[:, :-1], X_mix_tr[:, :-1]]), np.vstack([pure_Y, Y_mix_tr]))
    models["EI"] = (m, lambda X: X[:, :-1])

    train, _ = train_test_split(data, test_size=0.2, random_state=2026)
    pm = {}
    for col in MIX_PROPS:
        tr = train[train[col].notna()]
        pm[col] = Pipeline([("i", SimpleImputer()),
                            ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                                      min_samples_leaf=2, n_jobs=1))]).fit(
            tr[zcols + ["desc_airi_ri"]], tr[col].to_numpy(float))

    print(f"  {'variant':7s} {'Z MAE':>8}  " +
          "  ".join(f"{c.replace('PubChem_','').replace('fuel_','')[:14]:>16}" for c in MIX_PROPS))
    for tag, (zm, cut) in models.items():
        Zp = zm.predict(cut(X_mix_te))
        zmae = float(np.abs(Zp - Y_mix_te).mean())
        errs = []
        Zdf = pd.DataFrame(Zp, columns=zcols)
        Zdf["desc_airi_ri"] = X_mix_te[:, -1]
        for col in MIX_PROPS:
            pred = pm[col].predict(Zdf[zcols + ["desc_airi_ri"]])
            errs.append(float(np.abs(pred - T_te[:, MIX_PROPS.index(col)]).mean()))
        print(f"  {tag:7s} {zmae:8.4f}  " + "  ".join(f"{e:16.3f}" for e in errs))
    print("  (property truth = mass-weighted average of the components; "
          "prediction uses the mass-weighted-average Z the model inferred)")


def part_e(data, spec, zcols) -> None:
    """Mixture level: is the slice's RI needed next to the predicted Z?"""
    print("=" * 78)
    print("E. second stage at mixture level: predicted Z vs predicted Z + RI")
    print("=" * 78)
    import joblib
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    zcols47 = zcols          # the 46-position Z used in the paper
    zm = joblib.load(OUT / "augmented_z_model.joblib")
    zc = list(np.load(OUT / "zcols.npy", allow_pickle=True))
    assert list(zc) == list(zcols47)

    probs = ["PubChem_MolecularWeight", "chem_Tb_K", "density_label_g_ml",
             "chem_Hfus_J_mol", "chem_Hfg_J_mol"]
    props = {}
    train, _ = train_test_split(data, test_size=0.2, random_state=2026)
    for col in probs:
        tr = train[train[col].notna()]
        for tag, feats in (("Z", zcols47), ("Z+RI", zcols47 + ["desc_airi_ri"])):
            props[(col, tag)] = Pipeline([("i", SimpleImputer()),
                                          ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                                                    min_samples_leaf=2, n_jobs=1))]).fit(
                tr[feats], tr[col].to_numpy(float))

    sel = select_virtual_diesel(data)
    sel_w = sel["_w"].to_numpy(float)
    truth = {c: float((sel[c].to_numpy(float) * sel_w).sum()) for c in probs}
    truth["C_mass_frac"] = float((12.011 * sel["desc_C"] * sel_w).sum()
                                 / (12.011 * sel["desc_C"] + 1.008 * sel["desc_H"]).to_numpy().dot(sel_w))
    truth["H_mass_frac"] = float((1.008 * sel["desc_H"] * sel_w).sum()
                                 / (12.011 * sel["desc_C"] + 1.008 * sel["desc_H"]).to_numpy().dot(sel_w))

    rng = np.random.default_rng(2026)
    vm, _w = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Z = pd.DataFrame(zm.predict(Xv), columns=zcols47)
    Z["desc_airi_ri"] = vm["ri"].to_numpy(float)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    print("\n  virtual diesel (true = mass-weighted compounds, pred = area-weighted slices)")
    print(f"    {'property':26s} {'true':>10} {'Z':>10} {'Z+RI':>10}")
    for col in probs:
        p1 = float(np.sum(props[(col, "Z")].predict(Z[zcols47]) * aw))
        p2 = float(np.sum(props[(col, "Z+RI")].predict(Z[zcols47 + ["desc_airi_ri"]]) * aw))
        print(f"    {col:26s} {truth[col]:10.4f} {p1:10.4f} {p2:10.4f}")
    mwp = 12.011 * Z["desc_C"].to_numpy(float) + 1.008 * Z["desc_H"].to_numpy(float)
    cf = 12.011 * Z["desc_C"].to_numpy(float) / mwp
    print(f"    {'C mass fraction':26s} {truth['C_mass_frac']:10.4f} "
          f"{float((cf*aw).sum()):10.4f} {float((cf*aw).sum()):10.4f}")

    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols47)
    Zr["desc_airi_ri"] = real["ri"].to_numpy(float)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    print("\n  real diesel (measured values from the 33# sample)")
    print(f"    {'property':26s} {'measured':>10} {'Z':>10} {'Z+RI':>10}")
    for col, meas in MEASURED.items():
        if col == "C_mass_frac":
            mwp_r = 12.011 * Zr["desc_C"].to_numpy(float) + 1.008 * Zr["desc_H"].to_numpy(float)
            v = float((12.011 * Zr["desc_C"].to_numpy(float) / mwp_r * rw).sum())
        elif col == "H_mass_frac":
            mwp_r = 12.011 * Zr["desc_C"].to_numpy(float) + 1.008 * Zr["desc_H"].to_numpy(float)
            v = float((1.008 * Zr["desc_H"].to_numpy(float) / mwp_r * rw).sum())
        else:
            c = "density_label_g_ml"
            v1 = float(np.sum(props[(c, "Z")].predict(Zr[zcols47]) * rw))
            v2 = float(np.sum(props[(c, "Z+RI")].predict(Zr[zcols47 + ["desc_airi_ri"]]) * rw))
            print(f"    {col:26s} {meas:10.4f} {v1:10.4f} {v2:10.4f}")
            continue
        print(f"    {col:26s} {meas:10.4f} {v:10.4f} {v:10.4f}")


def broad(Z: pd.DataFrame, w: np.ndarray) -> dict:
    chain = Z["desc_is_chain_alkane"].to_numpy(float)
    arom = Z["desc_is_aromatic"].to_numpy(float)
    return {"chain": float(np.sum(chain * w)),
            "aromatic": float(np.sum(arom * w)),
            "naphthene": float(np.sum(np.clip(1.0 - chain - arom, 0, None) * w)),
            "avg_DBE": float(np.sum(Z["desc_DBE"].to_numpy(float) * w)),
            "avg_ring": float(np.sum(Z["desc_ring_count"].to_numpy(float) * w))}


# --------------------------------------------------------------- part D
PROPS = {
    "PubChem_MolecularWeight": "molecular weight", "chem_Vc_m3_mol": "critical volume",
    "PubChem_XLogP": "XLogP", "chem_Pc_Pa": "critical pressure",
    "PubChem_Complexity": "complexity", "PubChem_RotatableBondCount": "rotatable bonds",
    "chem_Tb_K": "boiling point", "chem_Tc_K": "critical temperature",
    "chem_Tm_K": "melting point", "chem_Hfus_J_mol": "enthalpy of fusion",
    "chem_Hfg_J_mol": "enthalpy of formation", "density_label_g_ml": "density",
}


def part_d(data, spec, zcols) -> None:
    """Is RI needed *on top of* Z? Stratify by whether Z is ambiguous."""
    print("=" * 78)
    print("D. second stage: given Z, does RI still help? (stratified by Z collisions)")
    print("=" * 78)
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    key = data[zcols].round(3).astype(str).agg("|".join, axis=1)
    sizes = key.map(Counter(key))
    data = data.assign(_collided=(sizes >= 2).to_numpy())
    train, test = train_test_split(data, test_size=0.2, random_state=2026)
    print(f"  molecules sharing their Z with someone: {int((sizes>=2).sum())} "
          f"({100*(sizes>=2).mean():.1f}%)   test set: {int(test._collided.sum())} collided")

    rows, spreads = [], {}
    for col, lab in PROPS.items():
        tr, te = train[train[col].notna()], test[test[col].notna()]
        z_only = tr[zcols]
        z_ri = tr[zcols + ["desc_airi_ri"]]
        m1 = Pipeline([("i", SimpleImputer()),
                       ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                                 min_samples_leaf=2, n_jobs=1))]).fit(
            z_only, tr[col].to_numpy(float))
        m2 = Pipeline([("i", SimpleImputer()),
                       ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                                 min_samples_leaf=2, n_jobs=1))]).fit(
            z_ri, tr[col].to_numpy(float))
        y = te[col].to_numpy(float)
        p1, p2 = m1.predict(te[zcols]), m2.predict(te[zcols + ["desc_airi_ri"]])
        coll = te._collided.to_numpy(bool)
        row = {"property": lab}
        for tag, mask in (("all", np.ones(len(te), bool)),
                          ("collided", coll), ("unique", ~coll)):
            if mask.sum() < 10:
                continue
            row[f"r2_Z_{tag}"] = r2_score(y[mask], p1[mask])
            row[f"r2_ZRI_{tag}"] = r2_score(y[mask], p2[mask])
        rows.append(row)
        # how much does the property vary inside a Z collision group?
        grp = data.groupby(key)[col].std()
        spreads[lab] = float(grp.dropna().mean())
    d = pd.DataFrame(rows)
    d["gain_all"] = d.r2_ZRI_all - d.r2_Z_all
    d["gain_collided"] = d.r2_ZRI_collided - d.r2_Z_collided
    d["gain_unique"] = d.r2_ZRI_unique - d.r2_Z_unique
    pd.set_option("display.width", 200)
    print("\n  R2 with true Z vs true Z + RI:")
    print(d[["property", "r2_Z_all", "r2_ZRI_all", "gain_all",
             "r2_Z_collided", "r2_ZRI_collided", "gain_collided",
             "r2_Z_unique", "r2_ZRI_unique", "gain_unique"]].round(4).to_string(index=False))
    print(f"\n  mean gain   all {d.gain_all.mean():+.4f}   "
          f"collided {d.gain_collided.mean():+.4f}   unique {d.gain_unique.mean():+.4f}")
    print(f"  mean within-Z-group property spread (what Z alone cannot resolve): "
          f"{np.mean(list(spreads.values())):.3f}  "
          f"(e.g. boiling point {spreads['boiling point']:.1f} K, density {spreads['density']:.4f} g/mL)")
    d.to_csv(OUT / "ri_contribution_z_plus_ri.csv", index=False)
    print("  saved:", OUT / "ri_contribution_z_plus_ri.csv")


def true_diesel_composition(sel: pd.DataFrame) -> dict:
    w = sel["_w"].to_numpy(float)
    chain = sel["desc_DBE"].to_numpy(float) == 0.0
    arom = sel["desc_is_aromatic"].to_numpy(float) == 1.0
    return {"chain": float(w[chain].sum()), "aromatic": float(w[arom].sum()),
            "naphthene": float(w[~chain & ~arom].sum()),
            "avg_DBE": float((sel["desc_DBE"].to_numpy(float) * w).sum()),
            "avg_ring": float((sel["desc_ring_count"].to_numpy(float) * w).sum())}


# --------------------------------------------------------------- part G
def part_g(data, spec, zcols) -> None:
    """How much of the 'uniqueness' of Z+RI is trivial (RI is continuous)?"""
    print("=" * 78)
    print("G. identification: RI alone vs Z alone, at several rounding precisions")
    print("=" * 78)

    def uniq(vals: np.ndarray) -> tuple[int, float]:
        key = pd.Series(np.round(vals, 3).astype(str))
        sizes = np.array(list(Counter(key).values()))
        n = len(key)
        pair = sum(v * (v - 1) / 2 for v in sizes) / (n * (n - 1) / 2) * 100
        return len(sizes), pair

    ri = data["desc_airi_ri"].to_numpy(float)
    key47 = data[zcols].round(3).astype(str).agg("|".join, axis=1)

    print(f"  {'variant':28s} {'unique':>7} {'pairwise collision %':>21}")
    for label, vals in [("RI alone (raw)", ri),
                        ("RI alone (round 1)", np.round(ri, 1)),
                        ("RI alone (round to 5 RI units)", np.round(ri / 5) * 5),
                        ("RI alone (round to 10 RI units)", np.round(ri / 10) * 10)]:
        u, p = uniq(vals)
        print(f"  {label:28s} {u:7d} {p:21.4f}")
    u, p = uniq(np.arange(len(data)))          # any continuous id -> trivial
    print(f"  {'(a meaningless serial number)':28s} {u:7d} {p:21.4f}")
    print("  " + "-" * 58)
    for label, vals in [("47 Z alone", None), ("47 Z + RI (round 1)", None)]:
        pass
    zsizes = np.array(list(Counter(key47).values()))
    print(f"  {'47 Z alone':28s} {len(zsizes):7d} "
          f"{sum(v*(v-1)/2 for v in zsizes)/(len(data)*(len(data)-1)/2)*100:21.4f}")
    for d in (1.0, 5.0, 10.0):
        key = key47 + "@" + pd.Series(np.round(ri / d) * d).astype(str)
        sizes = np.array(list(Counter(key).values()))
        pair = sum(v * (v - 1) / 2 for v in sizes) / (len(data) * (len(data) - 1) / 2) * 100
        print(f"  {'47 Z + RI (RI binned ' + f'{d:g}' + ' units)':28s} {len(sizes):7d} {pair:21.4f}")


# --------------------------------------------------------------- part H
def part_h(data, spec, zcols) -> None:
    """Inside a Z collision group: does RI track the property difference?"""
    print("=" * 78)
    print("H. inside Z-collision groups: is RI able to explain the property spread?")
    print("=" * 78)
    key = data[zcols].round(3).astype(str).agg("|".join, axis=1)
    data = data.assign(_g=key)
    sizes = data.groupby("_g")["identifier"].transform("size")
    grp = data[sizes >= 2]
    print(f"  {grp._g.nunique()} collision groups covering {len(grp)} molecules")
    rows = []
    for col, lab in PROPS.items():
        if col not in grp.columns:
            continue
        sub = grp[["_g", "desc_airi_ri", col]].dropna()
        within = sub.groupby("_g")
        n_g = within.size()
        keep = n_g[n_g >= 3].index
        sub = sub[sub._g.isin(keep)]
        if sub.empty:
            continue
        dev = sub.copy()
        for c in ["desc_airi_ri", col]:
            dev[c] = dev[c] - dev.groupby("_g")[c].transform("mean")
        r = float(np.corrcoef(dev["desc_airi_ri"], dev[col])[0, 1])
        rng_ = float((within[col].max() - within[col].min()).mean())
        rng_max = float((within[col].max() - within[col].min()).max())
        rows.append({"property": lab, "within-group r(RI, property)": round(r, 3),
                     "mean within-group range": round(rng_, 3),
                     "max within-group range": round(rng_max, 3),
                     "n groups": len(keep)})
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(d.to_string(index=False))
    d.to_csv(OUT / "ri_within_collision_group.csv", index=False)
    print("  saved:", OUT / "ri_within_collision_group.csv")


# --------------------------------------------------------------- part I
def part_i(data, spec, zcols) -> None:
    """Who are the colliding molecules whose properties differ a lot?"""
    print("=" * 78)
    print("I. the collision groups with the largest boiling-point spread")
    print("=" * 78)
    rec = pd.read_csv(BASE / "data" / "hydrocarbon_records.csv",
                      usecols=["identifier", "SMILES", "FORMULA"])
    d = data.merge(rec[["identifier", "SMILES"]], on="identifier", how="left")
    key = d[zcols].round(3).astype(str).agg("|".join, axis=1)
    d = d.assign(_g=key)
    cnt = d.groupby("_g")["identifier"].transform("size")
    d = d[cnt >= 2]
    stats = d.groupby("_g")["chem_Tb_K"].agg(["size", "min", "max", "std"])
    stats["range"] = stats["max"] - stats["min"]
    print(f"  694 collision groups, mean Tb range {stats.range.mean():.1f} K, "
          f"median {stats.range.median():.1f} K, max {stats.range.max():.1f} K")
    print(f"  groups with Tb range > 20 K: {(stats.range>20).sum()} of {len(stats)}")
    ri_gap = d.groupby("_g")["desc_airi_ri"].agg(lambda s: float(s.max() - s.min()))
    print(f"  RI spread inside a group: mean {ri_gap.mean():.1f} RI units, "
          f"median {ri_gap.median():.1f}")
    for g in stats.sort_values("range", ascending=False).head(3).index:
        m = d[d._g == g]
        print(f"\n  --- group with Tb range {stats.loc[g,'range']:.1f} K ({len(m)} molecules)")
        for r in m.sort_values("chem_Tb_K").itertuples():
            print(f"    C{r.desc_C:<3.0f} DBE{r.desc_DBE:<3.0f} branch{r.desc_branch_carbon_count:<3.0f} "
                  f"Tb={r.chem_Tb_K:7.1f} K  RI={r.desc_airi_ri:7.1f}  {r.SMILES}")


# --------------------------------------------------------------- part J
def part_j(data, spec, zcols) -> None:
    """Same collision analysis, but only on saturated + aromatic molecules."""
    print("=" * 78)
    print("J. collision groups inside the paper's chemical space (saturated + aromatic)")
    print("=" * 78)
    cls = pd.read_csv(BASE / "data" / "hydrocarbon_class_strict.csv",
                      usecols=["identifier", "class_strict", "SMILES"])
    d = data.merge(cls, on="identifier", how="left")
    keep = d["class_strict"] != "Unsaturated"
    print(f"  molecules: {len(d)}  ->  saturated + aromatic subset: {int(keep.sum())}")
    d = d[keep].copy()
    key = d[zcols].round(3).astype(str).agg("|".join, axis=1)
    d = d.assign(_g=key)
    sizes = d.groupby("_g")["identifier"].transform("size")
    d = d.assign(_n=sizes)
    collided = d[d._n >= 2]
    print(f"  unique Z {d._g.nunique()} / {len(d)}   "
          f"molecules in a collision group {len(collided)} ({100*len(collided)/len(d):.1f}%)")
    stats = collided.groupby("_g")["chem_Tb_K"].agg(["size", "min", "max"])
    stats["range"] = stats["max"] - stats["min"]
    print(f"  Tb range inside a group: mean {stats.range.mean():.1f} K, "
          f"median {stats.range.median():.1f} K, max {stats.range.max():.1f} K")
    print(f"  groups with Tb range > 20 K: {(stats.range>20).sum()} / {len(stats)}")
    ri_spread = collided.groupby("_g")["desc_airi_ri"].agg(lambda s: float(s.max() - s.min()))
    print(f"  RI spread inside a group: mean {ri_spread.mean():.1f}, median {ri_spread.median():.1f}")
    rows = []
    for col, lab in PROPS.items():
        sub = collided[["_g", "desc_airi_ri", col]].dropna()
        n_g = sub.groupby("_g").size()
        sub = sub[sub._g.isin(n_g[n_g >= 3].index)]
        if sub.empty:
            continue
        dev = sub.copy()
        for c in ["desc_airi_ri", col]:
            dev[c] = dev[c] - dev.groupby("_g")[c].transform("mean")
        r = float(np.corrcoef(dev["desc_airi_ri"], dev[col])[0, 1])
        rng_ = float((sub.groupby("_g")[col].max() - sub.groupby("_g")[col].min()).mean())
        rows.append({"property": lab, "r(RI, property) within group": round(r, 3),
                     "mean within-group range": round(rng_, 3)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    # a few example groups with the largest spread
    for g in stats.sort_values("range", ascending=False).head(2).index:
        m = d[d._g == g]
        print(f"\n  --- group with Tb range {stats.loc[g,'range']:.1f} K ({len(m)} molecules)")
        for r in m.sort_values("chem_Tb_K").itertuples():
            print(f"    C{r.desc_C:<3.0f} DBE{r.desc_DBE:<3.0f} Tb={r.chem_Tb_K:7.1f} K "
                  f"RI={r.desc_airi_ri:7.1f}  {r.SMILES}")


def part_k(data, spec, zcols) -> None:
    """Where Tb really differs inside a Z group, does RI track it?"""
    print("=" * 78)
    print("K. does RI track the property inside the groups where it matters?")
    print("=" * 78)
    import glob
    cls = pd.read_csv(BASE / "data" / "hydrocarbon_class_strict.csv",
                      usecols=["identifier", "class_strict"])
    d = data.merge(cls, on="identifier", how="left")
    d = d[d["class_strict"] != "Unsaturated"].copy()
    key = d[zcols].round(3).astype(str).agg("|".join, axis=1)
    d = d.assign(_g=key)
    d = d[d.groupby("_g")["identifier"].transform("size") >= 3]
    rs, pooled_x, pooled_y = [], [], []
    for _g, sub in d.groupby("_g"):
        s = sub[["desc_airi_ri", "chem_Tb_K"]].dropna()
        if len(s) < 3 or s.chem_Tb_K.max() - s.chem_Tb_K.min() < 20:
            continue
        rs.append(float(np.corrcoef(s.desc_airi_ri, s.chem_Tb_K)[0, 1]))
        pooled_x.append(s.desc_airi_ri - s.desc_airi_ri.mean())
        pooled_y.append(s.chem_Tb_K - s.chem_Tb_K.mean())
    rs = np.array(rs)
    pool_r = float(np.corrcoef(np.concatenate(pooled_x), np.concatenate(pooled_y))[0, 1]) \
        if rs.size else float("nan")
    print(f"  groups (n>=3) with Tb range > 20 K: {len(rs)}")
    print(f"  within-group r(RI, Tb): mean {rs.mean():.2f}, median {np.median(rs):.2f}, "
          f"|r|>0.8 in {int((np.abs(rs)>0.8).sum())} groups")
    print(f"  pooled within-group r(RI, Tb) over those groups: {pool_r:.2f}")

    files = sorted(glob.glob(str(BASE / "data" / "airi" / "*.csv")))
    print("\n  RI data files:", [Path(f).name for f in files])
    for f in files:
        t = pd.read_csv(f, nrows=5)
        cand = [c for c in t.columns if "ri" in c.lower() or "true" in c.lower()
                or "exp" in c.lower() or "pred" in c.lower()]
        if len(cand) >= 2:
            full = pd.read_csv(f)
            print(f"    {Path(f).name}: columns {list(full.columns)[:8]}")
            break


def part_m(data, spec, zcols) -> None:
    """What do the 27 substructure bits contribute to the *composition* task?"""
    print("=" * 78)
    print("M. the 27 substructure bits and the composition task")
    print("=" * 78)
    import joblib

    size = data[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    alip_bits = [c for c in zcols if c.startswith("ring_aliphatic_")]
    has_alip = data[alip_bits].to_numpy(float).sum(1) > 0
    flag = data["desc_is_aromatic"].to_numpy(float) > 0.5
    chain = data["desc_is_chain_alkane"].to_numpy(float) > 0.5
    dbe = np.clip(np.rint(data["desc_DBE"].to_numpy(float)).astype(int), 0, 10)

    print("\nM1 aromatic fraction on the library")
    print(f"  by descriptor flag (desc_is_aromatic)      : {flag.mean()*100:5.1f}%")
    print(f"  by aromatic ring bits (structural truth)   : {size.mean()*100:5.1f}%")
    print(f"  -> the flag over-reports by {(flag.mean()-size.mean())*100:.1f} percentage points")

    print("\nM2 which DBE class is actually aromatic? (structural, from the 27 bits)")
    print(f"  {'DBE':>3} {'n':>5} {'aromatic':>9} {'naphthenic':>11} {'acyclic':>8}")
    rows = []
    for k in range(11):
        m = dbe == k
        if m.sum() == 0:
            continue
        ar, na, ac = float(size[m].mean()), float((has_alip[m] & ~size[m]).mean()), float((~has_alip[m] & ~size[m]).mean())
        rows.append({"DBE": k, "n": int(m.sum()), "aromatic": ar, "naphthenic": na, "acyclic": ac})
        print(f"  {k:>3} {int(m.sum()):5d} {ar*100:8.1f}% {na*100:10.1f}% {ac*100:7.1f}%")
    pd.DataFrame(rows).to_csv(OUT / "dbe_class_identity.csv", index=False)

    cls = pd.read_csv(BASE / "data" / "hydrocarbon_class_strict.csv",
                      usecols=["identifier", "class_strict"])
    dd = data.merge(cls, on="identifier", how="left")
    dd = dd[dd["class_strict"] != "Unsaturated"]
    s2 = dd[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    a2 = dd[alip_bits].to_numpy(float).sum(1) > 0
    dbe2 = np.clip(np.rint(dd["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    print("\nM2b same table restricted to the 3133 saturated + aromatic molecules")
    print(f"  {'DBE':>3} {'n':>5} {'aromatic':>9} {'naphthenic':>11} {'acyclic':>8}")
    for k in range(11):
        m = dbe2 == k
        if m.sum() == 0:
            continue
        print(f"  {k:>3} {int(m.sum()):5d} {s2[m].mean()*100:8.1f}% "
              f"{(a2[m] & ~s2[m]).mean()*100:10.1f}% {(~a2[m] & ~s2[m]).mean()*100:7.1f}%")

    print("\nM3 virtual diesel: the chain / naphthene / aromatic split")
    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    sel_size = sel[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    sel_alip = sel[alip_bits].to_numpy(float).sum(1) > 0
    t_ar, t_na = float(sw[sel_size].sum()), float(sw[sel_alip & ~sel_size].sum())
    print(f"  true (structure):      chain {1-t_ar-t_na:.3f}  naphthene {t_na:.3f}  aromatic {t_ar:.3f}")
    zm = joblib.load(OUT / "augmented_z_model.joblib")
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Z = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    p_flag = Z["desc_is_aromatic"].to_numpy(float)
    p_size = Z[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    p_chain = Z["desc_is_chain_alkane"].to_numpy(float)
    for tag, ar in (("predicted by flag", p_flag), ("predicted by the 27 bits", p_size)):
        print(f"  {tag:22s} chain {float(np.sum(p_chain*aw)):.3f}  "
              f"naphthene {float(np.sum((1-p_chain-np.asarray(ar,dtype=float))*aw)):.3f}  "
              f"aromatic {float(np.sum(np.asarray(ar,dtype=float)*aw)):.3f}")

    print("\nM4 real diesel vs SH/T0606 (aromatic = 8.4+7.8+2.7+1.4+0.6+1.0+0.1 = 22.0%)")
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    rf = Zr["desc_is_aromatic"].to_numpy(float)
    rs = (Zr[AROMATIC_BITS].to_numpy(float).sum(1) > 0).astype(float)
    rc = Zr["desc_is_chain_alkane"].to_numpy(float)
    for tag, ar in (("descriptor flag", rf), ("27 aromatic bits", rs)):
        print(f"  {tag:18s} chain {float(np.sum(rc*rw)):.3f}  "
              f"naphthene {float(np.sum((1-rc-ar)*rw)):.3f}  aromatic {float(np.sum(ar*rw)):.3f}")


def part_n(data, spec, zcols) -> None:
    """Can the 9 aromatic bits be combined into a usable aromatic fraction?"""
    print("=" * 78)
    print("N. calibrated combination of the aromatic-ring bits vs the descriptor flag")
    print("=" * 78)
    import joblib

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    lib_X = np.hstack([data[spec].to_numpy(float),
                       data["desc_airi_ri"].to_numpy(float)[:, None]])
    Zlib = pd.DataFrame(zm.predict(lib_X), columns=zcols)
    truth = (data[AROMATIC_BITS].to_numpy(float).sum(1) > 0)
    score = Zlib[AROMATIC_BITS].to_numpy(float).sum(1)
    thr = float(np.quantile(score, 1 - truth.mean()))
    print(f"  library aromatic rate (structural)      : {truth.mean()*100:.1f}%")
    print(f"  predicted by desc_is_aromatic           : "
          f"{(Zlib['desc_is_aromatic'].to_numpy()>0.5).mean()*100:.1f}%")
    print(f"  predicted by OR of the 9 bits           : {(score>0.5).mean()*100:.1f}%")
    print(f"  predicted by sum-of-bits > {thr:.2f} (prevalence matched): "
          f"{(score>thr).mean()*100:.1f}%")
    from sklearn.metrics import roc_auc_score
    print(f"  AUC of the flag        : "
          f"{roc_auc_score(truth, Zlib['desc_is_aromatic']):.3f}")
    print(f"  AUC of the sum-of-bits : {roc_auc_score(truth, score):.3f}")

    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    s_sel = sel[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    true_ar = float(sw[s_sel].sum())
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()

    print("\n  aromatic fraction of the mixtures:")
    print(f"    {'estimator':34s} {'virtual diesel':>15} {'real diesel':>13}")
    print(f"    {'true (virtual) / SH-T0606 (real)':34s} {true_ar*100:14.1f}% {22.0:12.1f}%")
    est = [("desc_is_aromatic flag",
            Zv["desc_is_aromatic"].to_numpy(float) > 0.5,
            Zr["desc_is_aromatic"].to_numpy(float) > 0.5),
           ("OR of the 9 aromatic bits",
            Zv[AROMATIC_BITS].to_numpy(float).sum(1) > 0.5,
            Zr[AROMATIC_BITS].to_numpy(float).sum(1) > 0.5),
           ("sum-of-bits, prevalence matched",
            Zv[AROMATIC_BITS].to_numpy(float).sum(1) > thr,
            Zr[AROMATIC_BITS].to_numpy(float).sum(1) > thr)]
    for tag, a, b in est:
        print(f"    {tag:34s} {float(np.sum(a*aw))*100:14.1f}% {float(np.sum(b*rw))*100:12.1f}%")


def part_o(data, spec, zcols) -> None:
    """A calibrated 'aromatic ring present' estimator built from the 27 bits."""
    print("=" * 78)
    print("O. composition: aromatic fraction from a *calibrated* combination of bits")
    print("=" * 78)
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    lib_X = np.hstack([data[spec].to_numpy(float),
                       data["desc_airi_ri"].to_numpy(float)[:, None]])
    Zlib = pd.DataFrame(zm.predict(lib_X), columns=zcols)
    y_true = (data[AROMATIC_BITS].to_numpy(float).sum(1) > 0).astype(int)
    X9 = Zlib[AROMATIC_BITS].to_numpy(float)
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, random_state=2026)
    # calibrate so that the predicted probability matches an aromatic fraction
    lr = LogisticRegression(max_iter=2000).fit(X9[tr], y_true[tr])
    p_te, p_tr = lr.predict_proba(X9[te])[:, 1], lr.predict_proba(X9[tr])[:, 1]
    print(f"  'has an aromatic ring' from the 9 bits: AUC {roc_auc_score(y_true[te], p_te):.3f} "
          f"(train {roc_auc_score(y_true[tr], p_tr):.3f})")
    print(f"  mean probability on the library: train {p_tr.mean()*100:.1f}% / "
          f"test {p_te.mean()*100:.1f}%   (true {y_true[te].mean()*100:.1f}%)")
    print(f"  mean of the raw desc_is_aromatic output: "
          f"{Zlib['desc_is_aromatic'].to_numpy(float).mean()*100:.1f}%")

    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    true_ar = float(sw[(sel[AROMATIC_BITS].to_numpy(float).sum(1) > 0)].sum())
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()

    p_v = lr.predict_proba(Zv[AROMATIC_BITS].to_numpy(float))[:, 1]
    p_r = lr.predict_proba(Zr[AROMATIC_BITS].to_numpy(float))[:, 1]
    print("\n  aromatic mass fraction of the mixtures:")
    print(f"    {'estimator':40s} {'virtual':>9} {'real':>9}")
    print(f"    {'true (virtual) / SH/T0606 (real)':40s} {true_ar*100:8.1f}% {22.0:8.1f}%")
    print(f"    {'desc_is_aromatic, area-weighted mean':40s} "
          f"{float(np.sum(Zv['desc_is_aromatic'].to_numpy(float)*aw))*100:8.1f}% "
          f"{float(np.sum(Zr['desc_is_aromatic'].to_numpy(float)*rw))*100:8.1f}%")
    print(f"    {'9 aromatic bits, calibrated':40s} "
          f"{float(np.sum(p_v*aw))*100:8.1f}% {float(np.sum(p_r*rw))*100:8.1f}%")
    print("\n  -> DBE-based classes stay as they are; only the aromatic/naphthene split")
    print("     is re-estimated from the ring-system bits.")


def part_p(data, spec, zcols) -> None:
    """Train 'has an aromatic ring' (the OR of the 9 bits) as a mixture-trained
    target and use it for the composition instead of desc_is_aromatic."""
    print("=" * 78)
    print("P. composition: a mixture-trained 'aromatic ring' target from the 27 bits")
    print("=" * 78)
    from sklearn.ensemble import ExtraTreesRegressor

    y = (data[AROMATIC_BITS].to_numpy(float).sum(1) > 0).astype(float)
    rng = np.random.default_rng(2026)
    spec_np = sum_normalize(data[spec].to_numpy(float))
    ri = data["desc_airi_ri"].to_numpy(float)
    cf = np.ones(len(data))
    X = np.zeros((22000, len(spec) + 1))
    Y = np.zeros((22000, 1))
    for i in range(22000):
        anchor = rng.uniform(900.0, 2800.0)
        cand = np.where(np.abs(ri - anchor) <= 40.0)[0]
        if len(cand) < 3:
            cand = np.argsort(np.abs(ri - anchor))[:20]
        chosen = rng.choice(cand, size=3, replace=False)
        w = rng.dirichlet(np.ones(3))
        contrib = np.zeros(len(spec))
        for j, idx in enumerate(chosen):
            contrib += (w[j] / cf[idx]) * spec_np[idx]
        contrib /= contrib.sum()
        X[i, : len(spec)] = contrib
        X[i, len(spec)] = float(np.sum(w * ri[chosen]))
        Y[i, 0] = float(np.sum(w * y[chosen]))
    pure_X = np.hstack([spec_np, ri[:, None]])
    m = ExtraTreesRegressor(n_estimators=120, random_state=2026,
                            min_samples_leaf=2, n_jobs=1)
    m.fit(np.vstack([pure_X, X]), np.vstack([y[:, None], Y]))
    import joblib
    joblib.dump(m, OUT / "aromatic_ring_bit_model.joblib")

    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    true_ar = float(sw[(sel[AROMATIC_BITS].to_numpy(float).sum(1) > 0)].sum())
    rng2 = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng2)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()
    print(f"  new target = OR of the 9 aromatic bits; library rate {y.mean()*100:.1f}%")
    print("\n  aromatic mass fraction of the mixtures:")
    print(f"    {'estimator':44s} {'virtual':>9} {'real':>9}")
    print(f"    {'true (virtual) / SH/T0606 (real)':44s} {true_ar*100:8.1f}% {22.0:8.1f}%")
    zm = joblib.load(OUT / "augmented_z_model.joblib")
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    print(f"    {'desc_is_aromatic (current pipeline)':44s} "
          f"{float(np.sum(Zv['desc_is_aromatic'].to_numpy(float)*aw))*100:8.1f}% "
          f"{float(np.sum(Zr['desc_is_aromatic'].to_numpy(float)*rw))*100:8.1f}%")
    print(f"    {'new aromatic-ring target (27 bits)':44s} "
          f"{float(np.sum(np.clip(m.predict(Xv),0,1)*aw))*100:8.1f}% "
          f"{float(np.sum(np.clip(m.predict(Xr),0,1)*rw))*100:8.1f}%")


def part_q(data, spec, zcols) -> None:
    """Do the 27 bits deliver a ring-system-resolved composition?"""
    print("=" * 78)
    print("Q. ring-system level composition from the 27 bits")
    print("=" * 78)
    import joblib
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()

    rows = []
    for c in zcols:
        if c.startswith("desc_"):
            continue
        true = float((sel[c].to_numpy(float) * sw).sum())
        pred = float((np.clip(Zv[c].to_numpy(float), 0, 1) * aw).sum())
        rows.append({"bit": short(c), "code": c, "true": true, "pred": pred,
                     "real": float((np.clip(Zr[c].to_numpy(float), 0, 1) * rw).sum())})
    t = pd.DataFrame(rows).sort_values("true", ascending=False)
    print(f"  {'ring system':22s} {'true %':>8} {'predicted %':>12} {'real diesel %':>14}")
    for r in t.head(10).itertuples():
        print(f"  {r.bit:22s} {r.true*100:7.1f}% {r.pred*100:11.1f}% {r.real*100:13.1f}%")
    print(f"\n  mean |error| over the 27 bits: "
          f"{np.abs(t.true - t.pred).mean()*100:.2f} percentage points")
    print("  mass fraction scale: each row is the share of the sample's mass that")
    print("  contains that ring system (overlaps allowed, so columns do not sum to 1)")
    t.to_csv(OUT / "ring_system_composition.csv", index=False)
    print("  saved:", OUT / "ring_system_composition.csv")


def part_r(data, spec, zcols) -> None:
    """Full 27-bit evidence table: AUC + mixture MAE + composition errors."""
    print("=" * 110)
    print("R. the 27 substructure bits - complete evidence table")
    print("=" * 110)
    import joblib
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    bits = [c for c in zcols if not c.startswith("desc_")]
    auc = pd.read_csv(BASE / "models" / "z_routes_evaluation" / "z_bit_metrics.csv")
    auc = auc[(auc.route == "EI_RI") & (auc.kind == "binary")].set_index("bit")["auc"]
    mix = pd.read_csv(OUT / "augmented" / "z_bit_mixture_mae.csv").set_index("bit")

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    # per-slice MAE against the true mass-weighted bit value
    Pv = np.clip(Zv[bits].to_numpy(float), 0, 1)
    Tv = vm[[f"_true_dbe_{k}" for k in range(11)]].to_numpy(float)  # only DBE fractions
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()

    rows = []
    for c in bits:
        true = float((sel[c].to_numpy(float) * sw).sum())
        pred = float((Pv[:, bits.index(c)] * aw).sum())
        rows.append({
            "bit": short(c), "code": c,
            "library %": float(data[c].mean() * 100),
            "single-compound AUC": float(auc[c]),
            "co-eluting mixture MAE": float(mix.loc[c, "aug_mae"]),
            "diesel true %": true * 100, "diesel pred %": pred * 100,
            "diesel |err| pp": abs(pred - true) * 100,
            "real diesel %": float((np.clip(Zr[c].to_numpy(float), 0, 1) * rw).sum() * 100),
        })
    t = pd.DataFrame(rows).sort_values("library %", ascending=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  mean single-compound AUC      : {t['single-compound AUC'].mean():.3f}"
          f"   (median {t['single-compound AUC'].median():.3f})")
    print(f"  mean co-eluting-mixture MAE   : {t['co-eluting mixture MAE'].mean():.3f}")
    print(f"  mean |error| on virtual diesel: {t['diesel |err| pp'].mean():.2f} pp"
          f"   (max {t['diesel |err| pp'].max():.2f} pp)")
    t.to_csv(OUT / "substructure_bit_evidence.csv", index=False)
    print("  saved:", OUT / "substructure_bit_evidence.csv")


def part_s(data, spec, zcols) -> None:
    """Assemble the motif-composition metric table for all four levels."""
    print("=" * 110)
    print("S. motif composition: AUC / MAE / MAPE at every level")
    print("=" * 110)
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    bits = [c for c in zcols if not c.startswith("desc_")]
    # --- single-compound level: same recipe as evaluate_z_routes.py
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, random_state=2026)
    X = np.hstack([data[spec].to_numpy(float), data["desc_airi_ri"].to_numpy(float)[:, None]])
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(X[tr], data[zcols].to_numpy(float)[tr])
    P = m.predict(X[te])
    single_mae = np.abs(P - data[zcols].to_numpy(float)[te]).mean(0)
    single_mae = dict(zip(zcols, single_mae))

    auc = pd.read_csv(BASE / "models" / "z_routes_evaluation" / "z_bit_metrics.csv")
    auc = auc[(auc.route == "EI_RI") & (auc.kind == "binary")].set_index("bit")["auc"]
    mix = pd.read_csv(OUT / "augmented" / "z_bit_mixture_mae.csv").set_index("bit")

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    real = pd.read_csv(BASE / "data" / "exp" / "virtual_molecules.csv")
    Xr = np.hstack([real[spec].to_numpy(float), real["ri"].to_numpy(float)[:, None]])
    Zr = pd.DataFrame(zm.predict(Xr), columns=zcols)
    rw = real["area"].to_numpy(float) / real["area"].to_numpy(float).sum()

    rows = []
    for c in bits:
        true = float((sel[c].to_numpy(float) * sw).sum())
        pred = float((np.clip(Zv[c].to_numpy(float), 0, 1) * aw).sum())
        rows.append({"bit": short(c), "code": c,
                     "library %": float(data[c].mean() * 100),
                     "AUC (single)": float(auc[c]),
                     "MAE (single)": float(single_mae[c]),
                     "MAE (co-eluting)": float(mix.loc[c, "aug_mae"]),
                     "diesel true %": true * 100, "diesel pred %": pred * 100,
                     "diesel MAE pp": abs(pred - true) * 100,
                     "real diesel %": float((np.clip(Zr[c].to_numpy(float), 0, 1) * rw).sum() * 100)})
    t = pd.DataFrame(rows).sort_values("library %", ascending=False)
    big = t[t["diesel true %"] >= 1.0].copy()
    big["diesel MAPE %"] = (big["diesel MAE pp"] / big["diesel true %"]) * 100
    t = t.merge(big[["bit", "diesel MAPE %"]], on="bit", how="left")
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  single compounds : mean AUC {t['AUC (single)'].mean():.3f}   "
          f"mean MAE {t['MAE (single)'].mean():.3f}")
    print(f"  co-eluting mixes : mean MAE {t['MAE (co-eluting)'].mean():.3f}")
    print(f"  virtual diesel   : mean MAE {t['diesel MAE pp'].mean():.2f} pp   "
          f"MAPE {big['diesel MAPE %'].mean():.1f}% (bits with true >= 1%, n={len(big)})")
    t.to_csv(OUT / "motif_composition_metrics.csv", index=False)
    print("  saved:", OUT / "motif_composition_metrics.csv")


def part_t(data, spec, zcols) -> None:
    """Are the molecules Z cannot separate actually much more similar?"""
    print("=" * 78)
    print("T. same-formula pairs: same Z (not separated) vs different Z (separated)")
    print("=" * 78)
    from itertools import combinations
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    RDLogger.DisableLog("rdApp.*")

    rec = pd.read_csv(BASE / "data" / "hydrocarbon_records.csv",
                      usecols=["identifier", "SMILES", "FORMULA"])
    d = data.merge(rec[["identifier", "SMILES", "FORMULA"]], on="identifier", how="left")
    key = d[zcols].round(3).astype(str).agg("|".join, axis=1).to_numpy()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in d["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    props = ["chem_Tb_K", "fuel_density_g_ml_298K", "chem_Hfg_J_mol",
             "chem_Tc_K", "PubChem_XLogP"]
    P = d[props].to_numpy(float)
    groups = d.groupby("FORMULA").indices          # formula -> positional indices

    rows_same, rows_diff, n_same, n_diff = [], [], 0, 0
    for _f, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            if fps[i] is None or fps[j] is None:
                continue
            tan = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            dp = P[i] - P[j]
            rec_row = (tan, *np.abs(dp))
            if key[i] == key[j]:
                rows_same.append(rec_row); n_same += 1
            else:
                rows_diff.append(rec_row); n_diff += 1
    cols = ["Tanimoto"] + [f"|d{p}|" for p in props]
    S = pd.DataFrame(rows_same, columns=cols)
    D = pd.DataFrame(rows_diff, columns=cols)
    print(f"  same-formula pairs: same Z = {n_same},  different Z = {n_diff}"
          f"   ({100*n_diff/(n_same+n_diff):.1f}% of isomer pairs are separated by Z)")
    out = []
    for c in cols:
        out.append({"metric": c, "same Z (median)": S[c].median(), "same Z (mean)": S[c].mean(),
                    "different Z (median)": D[c].median(), "different Z (mean)": D[c].mean()})
    o = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # a random-pair null
    rng = np.random.default_rng(2026)
    ia = rng.integers(0, len(d), 20000); ib = rng.integers(0, len(d), 20000)
    ok = (ia != ib) & np.array([f is not None for f in fps])[ia] & np.array([f is not None for f in fps])[ib]
    ia, ib = ia[ok], ib[ok]
    tan_r = np.array([DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in zip(ia, ib)])
    dtb_r = np.abs(P[ia, 0] - P[ib, 0])
    print(f"\n  reference (random pairs, n={len(ia)}): Tanimoto median {np.median(tan_r):.3f}, "
          f"|dTb| median {np.median(dtb_r):.1f} K")
    o.to_csv(OUT / "isomer_pair_similarity.csv", index=False)
    print("  saved:", OUT / "isomer_pair_similarity.csv")
    # pair-level data for the figure (all same-Z pairs + a random sample of the rest)
    rng_p = np.random.default_rng(7)
    idx_same = np.arange(len(rows_same))
    idx_diff = rng_p.choice(len(rows_diff), size=min(20000, len(rows_diff)), replace=False)
    pair_df = pd.DataFrame(
        np.vstack([np.asarray(rows_same)[idx_same], np.asarray(rows_diff)[idx_diff]]),
        columns=cols)
    pair_df["group"] = ["same Z (not separated)"] * len(idx_same) + \
                       ["different Z (separated)"] * len(idx_diff)
    pair_df.to_csv(OUT / "isomer_pair_scatter.csv", index=False)
    rng_r = np.random.default_rng(11)
    ia2 = rng_r.integers(0, len(d), 8000); ib2 = rng_r.integers(0, len(d), 8000)
    ok2 = (ia2 != ib2) & np.array([f is not None for f in fps])[ia2] & \
          np.array([f is not None for f in fps])[ib2]
    ia2, ib2 = ia2[ok2], ib2[ok2]
    rand_df = pd.DataFrame({
        "Tanimoto": [float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                     for i, j in zip(ia2, ib2)],
        "|dchem_Tb_K|": np.abs(P[ia2, 0] - P[ib2, 0])})
    rand_df["group"] = "random pairs"
    rand_df.to_csv(OUT / "random_pair_scatter.csv", index=False)

    cls = pd.read_csv(BASE / "data" / "hydrocarbon_class_strict.csv",
                      usecols=["identifier", "class_strict"])
    keep = set(cls.loc[cls["class_strict"] != "Unsaturated", "identifier"])
    in_keep = d["identifier"].isin(keep).to_numpy()
    rows_same2, rows_diff2 = [], []
    for _f, idxs in groups.items():
        idxs = [i for i in idxs if in_keep[i]]
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            tan = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            rec_row = (tan, *np.abs(P[i] - P[j]))
            (rows_same2 if key[i] == key[j] else rows_diff2).append(rec_row)
    S2 = pd.DataFrame(rows_same2, columns=cols)
    D2 = pd.DataFrame(rows_diff2, columns=cols)
    print(f"\n  restricted to saturated + aromatic molecules:")
    print(f"    same-Z pairs {len(S2)}, different-Z pairs {len(D2)}")
    o2 = pd.DataFrame([{"metric": c,
                        "same Z (median)": S2[c].median(),
                        "different Z (median)": D2[c].median()}
                       for c in cols])
    print(o2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    o2.to_csv(OUT / "isomer_pair_similarity_strict.csv", index=False)


def part_u(data, spec, zcols) -> None:
    """Every motif bit x every metric x every level (AUC / R2 / MAE / MAPE)."""
    print("=" * 120)
    print("U. motif bits: full metric matrix for individual compounds / co-eluting mixtures / virtual diesel")
    print("=" * 120)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    bits = [c for c in zcols if not c.startswith("desc_")]
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, random_state=2026)
    X = np.hstack([data[spec].to_numpy(float), data["desc_airi_ri"].to_numpy(float)[:, None]])
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    Yall = data[zcols].to_numpy(float)
    m.fit(X[tr], Yall[tr])
    P_ind = m.predict(X[te])

    truth_mix = np.load(OUT / "augmented" / "truth.npy")
    pred_mix = np.load(OUT / "augmented" / "augmented_pred.npy")
    jj = [zcols.index(c) for c in bits]

    zm = joblib.load(OUT / "augmented_z_model.joblib") if False else None
    import joblib
    zm = joblib.load(OUT / "augmented_z_model.joblib")
    sel = select_virtual_diesel(data)
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    # per-slice ground truth for each bit = mass-weighted average of the components
    mass = np.zeros((len(vm), len(sel)))
    ri_sel = sel["desc_airi_ri"].to_numpy(float)
    grid = vm["ri"].to_numpy(float)
    w_sel = np.array([sel["_w"].to_numpy(float)])  # recomputed below from class fractions
    # rebuild the same weights as build_slices
    cf = np.ones(len(sel))
    lib = sum_normalize(sel[spec].to_numpy(float))
    rng2 = np.random.default_rng(2026)
    _ = rng2.uniform(0, 0, 0)
    # the slice builder already produced the spectra; recover per-slice weights by
    # re-running the same random draws as build_slices for this rep
    rr = np.random.default_rng(2026)
    dbe = np.clip(np.rint(sel["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    class_frac = sel["_class_frac"].to_numpy(float)
    count = sel.groupby("_dbe")["identifier"].transform("count").to_numpy(float)
    # same weights as ``build_slices`` (no jitter, see the note there)
    _ = rr
    w = class_frac / count
    w = w / w.sum()
    Tbit = np.zeros((len(vm), len(bits)))
    for k, g in enumerate(grid):
        gauss = np.exp(-0.5 * ((g - ri_sel) / 8.0) ** 2)
        mass_k = w * gauss
        if mass_k.sum() > 0:
            mass_k = mass_k / mass_k.sum()
        Tbit[k] = mass_k @ sel[bits].to_numpy(float)
    Pbit = np.clip(Zv[bits].to_numpy(float), 0, 1)

    rows = []
    for k, c in enumerate(bits):
        y_ind, p_ind = Yall[te, zcols.index(c)], P_ind[:, zcols.index(c)]
        y_mix, p_mix = truth_mix[:, zcols.index(c)], pred_mix[:, zcols.index(c)]
        y_die, p_die = Tbit[:, k], Pbit[:, k]

        def auc(y, p):
            return roc_auc_score((y > 0.5).astype(int), p) if len(np.unique(y > 0.5)) > 1 else np.nan

        def mape(y, p, floor):
            mm = y >= floor
            return float(np.abs(p[mm] - y[mm]).mean() / y[mm].mean() * 100) if mm.sum() >= 3 else np.nan

        rows.append({
            "bit": short(c), "code": c,
            "ind AUC": auc(y_ind, p_ind), "ind R2": r2_score(y_ind, p_ind),
            "ind MAE": float(np.abs(p_ind - y_ind).mean()),
            "ind MAPE%": mape(y_ind, p_ind, 0.5),
            "mix AUC": auc(y_mix, p_mix), "mix R2": r2_score(y_mix, p_mix),
            "mix MAE": float(np.abs(p_mix - y_mix).mean()),
            "mix MAPE%": mape(y_mix, p_mix, 0.05),
            "die AUC": auc(y_die, p_die), "die R2": r2_score(y_die, p_die),
            "die MAE pp": float(np.abs(p_die - y_die).mean() * 100),
            "die MAPE%": mape(y_die, p_die, 0.01),
        })
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  column means:")
    means = t.drop(columns=["bit", "code"]).mean()
    print(means.round(3).to_string())
    t.to_csv(OUT / "motif_metric_matrix.csv", index=False)
    print("  saved:", OUT / "motif_metric_matrix.csv")


def part_v(data, spec, zcols) -> None:
    """Detection + quantification metrics for the motif bits (no R2 for 0/1)."""
    print("=" * 110)
    print("V. motif bits: detection (AUC / F1) and quantification (composition MAE / MAPE)")
    print("=" * 110)
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    bits = [c for c in zcols if not c.startswith("desc_")]
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, random_state=2026)
    X = np.hstack([data[spec].to_numpy(float), data["desc_airi_ri"].to_numpy(float)[:, None]])
    Y = data[zcols].to_numpy(float)
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=80, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(X[tr], Y[tr])
    P_ind = np.clip(m.predict(X[te]), 0, 1)
    truth_mix = np.load(OUT / "augmented" / "truth.npy")
    pred_mix = np.clip(np.load(OUT / "augmented" / "augmented_pred.npy"), 0, 1)

    zm = joblib.load(OUT / "augmented_z_model.joblib")
    sel = select_virtual_diesel(data)
    sw = sel["_w"].to_numpy(float)
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()

    rows = []
    for c in bits:
        j = zcols.index(c)
        y_ind = (Y[te, j] > 0.5).astype(int)
        p_ind = P_ind[:, j]
        y_mix = (truth_mix[:, j] > 0.5).astype(int)
        p_mix = pred_mix[:, j]
        true_c = float((sel[c].to_numpy(float) * sw).sum())
        pred_c = float((np.clip(Zv[c].to_numpy(float), 0, 1) * aw).sum())

        def det(y, p):
            yh = (p > 0.5).astype(int)
            if len(np.unique(y)) < 2:
                return np.nan, np.nan, np.nan, np.nan
            return (roc_auc_score(y, p), f1_score(y, yh, zero_division=0),
                    precision_score(y, yh, zero_division=0),
                    recall_score(y, yh, zero_division=0))

        a1, f1v, pr1, rc1 = det(y_ind, p_ind)
        a2, f2v, pr2, rc2 = det(y_mix, p_mix)
        rows.append({"bit": short(c), "code": c,
                     "ind AUC": a1, "ind F1": f1v,
                     "mix AUC": a2, "mix F1": f2v,
                     "comp true %": true_c * 100, "comp pred %": pred_c * 100,
                     "comp error pp": abs(pred_c - true_c) * 100,
                     "comp RE %": (abs(pred_c - true_c) / true_c * 100) if true_c >= 0.01 else np.nan})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  means:")
    print(t.drop(columns=["bit", "code"]).mean(numeric_only=True).round(3).to_string())
    big = t[t["comp true %"] >= 1.0]
    print(f"\n  aggregate composition: MAE {t['comp error pp'].mean():.2f} pp  "
          f"MAPE {big['comp RE %'].mean():.1f}% (motifs >= 1%)")
    t.to_csv(OUT / "motif_detection_quantification.csv", index=False)
    print("  saved:", OUT / "motif_detection_quantification.csv")


TYPE_TARGET = {0: 0.397, 1: 0.235, 2: 0.116, 3: 0.032, 4: 0.084,
               5: 0.078, 6: 0.027, 7: 0.014, 8: 0.006, 9: 0.010, 10: 0.001}

# ring systems that are actually plausible in a middle distillate; a selected
# molecule must contain at least one of them (this removes cyclononane,
# cyclodecane, cyclooctane, [18]annulene and similar library curiosities)
COMMON_RINGS = ["ring_aromatic_6_84480431", "ring_aliphatic_6_9970801b",
                "ring_aliphatic_5_e5038db0", "ring_aromatic_10_01f31866",
                "ring_aliphatic_10_f09f6a8f", "ring_aromatic_10_c1d2687b",
                "ring_aromatic_9_afce84c1", "ring_aromatic_14_541c53d1",
                "ring_aromatic_14_a1e6ec18", "ring_aliphatic_9_f0ab0e88",
                "ring_aliphatic_3_49aa76c6", "ring_aliphatic_4_18004bfa"]

# the core of a middle distillate: benzene / cyclohexane / cyclopentane rings,
# fused naphthenes and 1-3 ring aromatics.  Everything else (small strained
# rings, medium/large naphthenes, annulenes) is admitted only as a low-weight
# long tail, so that the evaluation still exercises those motifs without
# pretending they are abundant.
CORE_RINGS = ["ring_aromatic_6_84480431", "ring_aliphatic_6_9970801b",
              "ring_aliphatic_5_e5038db0", "ring_aromatic_10_01f31866",
              "ring_aliphatic_10_f09f6a8f", "ring_aromatic_10_c1d2687b",
              "ring_aromatic_9_afce84c1", "ring_aromatic_14_541c53d1",
              "ring_aromatic_14_a1e6ec18", "ring_aliphatic_9_f0ab0e88"]

# broad (chain, naphthene, aromatic) mass fractions for the ten virtual-diesel
# selections: a grid that spans the plausible composition space, so the motif
# contents cover the whole 0-100 % range instead of clustering at a few values
# A *ladder* of broad (chain, naphthene, aromatic) compositions: the aromatic
# fraction sweeps 5 % -> 75 % and the non-aromatic part is split between chain
# alkanes and naphthenes in two variants.  This guarantees that every
# concentration level of the aromatic / naphthenic motifs is populated, so the
# composition scatter (figure 37a) covers the whole mass-fraction range.
BROAD_GRID = []
for _arom in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
    for _f in (0.35, 0.65):
        _chain = (1.0 - _arom) * _f
        BROAD_GRID.append((round(_chain, 3), round(1.0 - _arom - _chain, 3), _arom))
BROAD_GRID += [(0.10, 0.80, 0.10), (0.80, 0.10, 0.10),
               (0.05, 0.45, 0.50), (0.45, 0.05, 0.50)]


def select_virtual_diesel_realistic(data: pd.DataFrame, seed: int = 2026,
                                    n_total: int = 120,
                                    vary_types: bool = True,
                                    diversify_rings: bool = True) -> pd.DataFrame:
    """Pick a *plausible* virtual diesel instead of sampling by DBE alone.

    Constraints (per SH/T 0606 compound types):
      * carbon number 9-24 (middle distillate), no alkynes;
      * DBE 0          -> alkanes (no ring);
      * DBE 1-3        -> naphthenes (ring, no aromatic ring);
      * DBE 4-10       -> aromatic series (>=1 aromatic ring bit);
      * every ring-containing molecule must carry at least one *common* ring
        (cyclopentane / cyclohexane / benzene / decalin / indane / tetralin /
         naphthalene / hydrindane / phenanthrene / anthracene / small rings).
    Class sizes follow the SH/T 0606 example composition.
    """
    rng = np.random.default_rng(seed)
    aro = data[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    alip = data[[c for c in data.columns if c.startswith("ring_aliphatic_")]].to_numpy(float).sum(1) > 0
    common = data[COMMON_RINGS].to_numpy(float).sum(1) > 0
    alkyne = data["alkyne_any"].to_numpy(float) > 0 if "alkyne_any" in data else np.zeros(len(data), bool)
    dbe = np.clip(np.rint(data["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    C = data["desc_C"].to_numpy(float)
    ri = data["desc_airi_ri"].to_numpy(float)
    # NOTE: no carbon-number limit.  The SH/T 0606 example is used only for the
    # *relative* class proportions; a real sample may well fall outside C9-C24.
    base = (~alkyne) & (ri >= 860) & (ri <= 3000)
    core = data[CORE_RINGS].to_numpy(float).sum(1) > 0
    ring_cols = [c for c in data.columns if c.startswith("ring_")]
    ring_mat = data[ring_cols].to_numpy(float) > 0
    pools, quotas = {}, {}
    for k, frac in TYPE_TARGET.items():
        if k == 0:
            pool = np.where(base & (dbe == 0) & (~aro) & (~alip))[0]
        elif k <= 3:
            pool = np.where(base & (dbe == k) & (~aro) & alip & common)[0]
        else:
            pool = np.where(base & (dbe == k) & aro & common)[0]
        n = max(2, int(round(frac * n_total)))
        if len(pool) == 0:
            continue
        pools[k] = pool
        quotas[k] = min(n, len(pool))

    # ---- greedy motif cover: make sure every motif that the plausible pools
    #      can supply appears at least once, scarcest motifs first -------------
    sel, is_tail, chosen = [], {}, set()
    ring_bits = [c for c in data.columns if c.startswith("ring_")] + \
                ["tertiary_carbon", "quaternary_carbon"]
    for c in sorted(ring_bits, key=lambda c: sum(int((data[c].to_numpy(float)[p] > 0).sum())
                                            for p in pools.values())):
        cands = [(k, int(i)) for k, p in pools.items()
                 for i in p if quotas[k] > 0 and int(i) not in chosen
                 and data[c].to_numpy(float)[i] > 0]
        if not cands:
            continue
        k, i = cands[rng.integers(len(cands))]
        chosen.add(i)
        quotas[k] -= 1
        sel.append(i)
        is_tail[i] = (not bool(core[i])) and int(dbe[i]) > 0

    for k, frac in TYPE_TARGET.items():
        if k not in pools:
            continue
        pool = pools[k]
        n = quotas[k]
        if n <= 0:
            continue
        pool = np.array([i for i in pool if int(i) not in chosen])
        if len(pool) == 0:
            continue
        core_mask = core[pool]                     # core ring systems?
        take = []
        if diversify_rings:
            n_core = int(round(0.75 * min(n, len(pool))))
            core_pool = pool[core_mask]
            tail_pool = pool[~core_mask]
            # round-robin over distinct ring-system signatures, so that each
            # motif present in the plausible pool is represented at least once
            def spread(cands, k):
                buckets: dict[tuple, list[int]] = {}
                for i in cands:
                    buckets.setdefault(tuple(np.where(ring_mat[i])[0]), []).append(i)
                out, keys = [], list(buckets)
                while len(out) < k and keys:
                    for s in list(keys):
                        if not buckets.get(s):
                            keys.remove(s)
                            continue
                        out.append(int(buckets[s].pop(rng.integers(len(buckets[s])))))
                        if len(out) >= k:
                            break
                return out
            take = spread(core_pool, min(n_core, len(core_pool)))
            rest = min(n, len(pool)) - len(take)
            if rest > 0 and len(tail_pool):
                take += spread(tail_pool, min(rest, len(tail_pool)))
            elif rest > 0:
                extra = [i for i in pool if i not in set(take)]
                rng.shuffle(extra)
                take += extra[:rest]
        else:
            take = rng.choice(pool, size=min(n, len(pool)), replace=False)
        sel.extend(int(i) for i in take)
        for i in take:
            # acyclic alkanes have no ring at all -> they are never "tail"
            is_tail[int(i)] = (not bool(core[i])) and int(dbe[i]) > 0
    out = data.iloc[sorted(set(sel))].copy()
    dbe_o = np.clip(np.rint(out["desc_DBE"].to_numpy(float)).astype(int), 0, 10)
    out["_dbe"] = dbe_o
    if vary_types:
        # Each virtual diesel gets its own type fractions.  The three broad
        # families are drawn from plausible ranges and then split inside the
        # family following the SH/T 0606 example, so that the motif contents
        # span a wide range of levels (including the 30-70 % band) instead of
        # clustering at the example values.
        chain_t, naph_t, arom_t = BROAD_GRID[int(seed) % len(BROAD_GRID)]
        jit_b = rng.normal(0.0, 0.03, 3)
        chain_t = max(0.05, chain_t + jit_b[0])
        naph_t = max(0.05, naph_t + jit_b[1])
        arom_t = max(0.05, arom_t + jit_b[2])
        s = chain_t + naph_t + arom_t
        chain_t, naph_t, arom_t = chain_t / s, naph_t / s, arom_t / s
        fam = {}
        for k in TYPE_TARGET:
            fam[k] = "chain" if k == 0 else ("naph" if k <= 3 else "arom")
        base = {f: sum(TYPE_TARGET[k] for k in TYPE_TARGET if fam[k] == f)
                for f in ("chain", "naph", "arom")}
        jit = {f: np.exp(rng.normal(0.0, 0.25, 3)) for f in ("chain", "naph", "arom")}
        tot = {"chain": chain_t, "naph": naph_t, "arom": arom_t}
        frac_map = {k: tot[fam[k]] * (TYPE_TARGET[k] / base[fam[k]])
                    for k in TYPE_TARGET}
    else:
        frac_map = dict(TYPE_TARGET)
    out["_class_frac"] = out["_dbe"].map(frac_map)
    cnt = out.groupby("_dbe")["identifier"].transform("count").to_numpy(float)
    tail_factor = np.array([0.35 if is_tail.get(int(i), False) else 1.0
                            for i in out.index], dtype=float)
    # renormalise inside each DBE class so that the class total still equals the
    # requested type fraction (down-weighting the long tail only redistributes
    # mass inside the class)
    f = pd.Series(tail_factor, index=out.index)
    f = f / f.groupby(out["_dbe"]).transform("mean")
    w = (out["_class_frac"].to_numpy(float) / cnt) * f.to_numpy(float) \
        * rng.uniform(0.5, 2.0, len(out))
    out["_w"] = w / w.sum()
    return out


def part_w(data, spec, zcols) -> None:
    """Check the plausibility and the composition error of the new selection."""
    print("=" * 100)
    print("W. realistic virtual diesel (SH/T 0606 types, plausible ring systems)")
    print("=" * 100)
    import joblib
    sys.path.insert(0, str(BASE / "src"))
    from z_labels import short

    sel = select_virtual_diesel_realistic(data)
    print(f"  selected {len(sel)} compounds   C {sel.desc_C.min():.0f}-{sel.desc_C.max():.0f}   "
          f"RI {sel.desc_airi_ri.min():.0f}-{sel.desc_airi_ri.max():.0f}")
    print("  per DBE class:", sel.groupby("_dbe").size().to_dict())
    aro = sel[AROMATIC_BITS].to_numpy(float).sum(1) > 0
    alip = sel[[c for c in sel.columns if c.startswith("ring_aliphatic_")]].to_numpy(float).sum(1) > 0
    w = sel["_w"].to_numpy(float)
    print(f"  mass fractions: chain {w[(sel._dbe == 0).to_numpy()].sum():.3f}  "
          f"naphthene {w[(alip & ~aro)].sum():.3f}  aromatic {w[aro].sum():.3f}")
    print("  ring inventory (mass fraction of molecules containing it):")
    for c in COMMON_RINGS:
        if c in sel.columns:
            v = float((sel[c].to_numpy(float) * w).sum())
            if v > 0.005:
                print(f"    {short(c):16s} {v:.3f}")

    # composition error with the existing augmented Z model
    zm = joblib.load(OUT / "augmented_z_model.joblib")
    rng = np.random.default_rng(2026)
    vm, _ = build_slices(sel, data, spec, rng)
    Xv = np.hstack([vm[spec].to_numpy(float), vm["ri"].to_numpy(float)[:, None]])
    Zv = pd.DataFrame(zm.predict(Xv), columns=zcols)
    aw = vm["area"].to_numpy(float) / vm["area"].to_numpy(float).sum()
    bits = [c for c in zcols if not c.startswith("desc_")]
    err = []
    for c in bits:
        true = float((sel[c].to_numpy(float) * w).sum())
        pred = float((np.clip(Zv[c].to_numpy(float), 0, 1) * aw).sum())
        err.append((short(c), true * 100, pred * 100, abs(pred - true) * 100))
    e = pd.DataFrame(err, columns=["bit", "true %", "pred %", "|err| pp"]).sort_values(
        "|err| pp", ascending=False)
    print(e.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  mean |error| {e['|err| pp'].mean():.2f} pp   "
          f"within 1 pp: {(e['|err| pp'] <= 1).sum()}/{len(e)}")
    sel.to_csv(OUT / "virtual_diesel_realistic_compounds.csv", index=False)
    e.to_csv(OUT / "virtual_diesel_realistic_composition.csv", index=False)
    print("  saved: virtual_diesel_realistic_compounds.csv / _composition.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
                             "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "all"])
    args = ap.parse_args()
    data, spec, zcols = load()
    if args.part in ("A", "all"):
        part_a(data, spec, zcols)
    if args.part in ("B", "all"):
        part_b(data, spec, zcols)
    if args.part in ("C", "all"):
        part_c(data, spec, zcols)
    if args.part == "D":
        part_d(data, spec, zcols)
    if args.part == "E":
        part_e(data, spec, zcols)
    if args.part == "F":
        part_f(data, spec, zcols)
    if args.part == "G":
        part_g(data, spec, zcols)
    if args.part == "H":
        part_h(data, spec, zcols)
    if args.part == "I":
        part_i(data, spec, zcols)
    if args.part == "J":
        part_j(data, spec, zcols)
    if args.part == "K":
        part_k(data, spec, zcols)
    if args.part == "L":
        part_l(data, spec, zcols)
    if args.part == "M":
        part_m(data, spec, zcols)
    if args.part == "N":
        part_n(data, spec, zcols)
    if args.part == "O":
        part_o(data, spec, zcols)
    if args.part == "P":
        part_p(data, spec, zcols)
    if args.part == "Q":
        part_q(data, spec, zcols)
    if args.part == "R":
        part_r(data, spec, zcols)
    if args.part == "S":
        part_s(data, spec, zcols)
    if args.part == "T":
        part_t(data, spec, zcols)
    if args.part == "U":
        part_u(data, spec, zcols)
    if args.part == "V":
        part_v(data, spec, zcols)
    if args.part == "W":
        part_w(data, spec, zcols)


if __name__ == "__main__":
    main()
