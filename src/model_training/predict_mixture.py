#!/usr/bin/env python
"""Train EI+RI -> Z -> property models, then predict mixture properties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from z_fingerprint_model import MZ_LO, MZ_HI, fit_z_model, predict_z_model, sum_normalize
from mixing_rules import density_mix, mass_to_mole

DROP_TARGETS = {
    "fuel_density_kg_m3_298K", "fuel_API_gravity_298K", "fuel_flash_point_C", "fuel_LHV_MJ_kg",
    "fuel_log10_dynamic_viscosity_Pa_s_298K", "fuel_log10_kinematic_viscosity_cSt_298K", "fuel_flash_point_K",
    "chem_Hfl_J_mol", "fuel_HHV_MJ_kg", "chem_S0g_J_molK", "chem_omega", "fuel_Hvap_J_kg",
    "fuel_specific_gravity_298K", "fuel_cetane_index_D976_Tb_proxy", "chem_MW", "PubChem_ExactMass",
    "PubChem_MonoisotopicMass", "PubChem_HeavyAtomCount", "PubChem_TPSA",
    # removed from the target set on 2026-10-10 (no longer reported anywhere):
    # cross-zero free energies and 2D count descriptors that are not physical
    # properties of a mixture
    "chem_Hfg_J_mol", "chem_Hfus_J_mol", "PubChem_Complexity", "PubChem_RotatableBondCount",
}

# The mixture layer reports the *same* eight target properties as the
# individual-compound layer: Tb / Tm / Tc / Pc / Vc / MW / XLogP / density.
# Free energies (Delta-Hf, Delta-Hfus) and the 2D count descriptors
# (complexity, rotatable bonds) were dropped from the project.
KEEP_TARGETS = [
    "chem_Tb_K", "chem_Tm_K", "chem_Tc_K", "chem_Pc_Pa", "chem_Vc_m3_mol",
    "PubChem_MolecularWeight", "PubChem_XLogP", "fuel_density_g_ml_298K",
]

DROP_DESC = {
    "desc_MW", "desc_exact_mass", "desc_nominal_mass", "desc_carbon_mass_fraction",
    "desc_hydrogen_mass_fraction", "desc_airi_ri", "desc_airi_ri_stddev", "desc_airi_ri_confidence_interval",
    "desc_dbe_ge_1", "desc_carbon_skeleton_diameter", "desc_component_count",
    "desc_c_5_10", "desc_c_11_16", "desc_c_17_24", "desc_c_25_40",
    "desc_is_normal_alkane", "desc_triple_bond_count", "desc_max_carbon_degree", "desc_is_cycloalkane",
    "desc_quaternary_carbon_count", "desc_branching_index", "desc_is_alkene",
}

REMOVE_FP = {"alkyne_any", "alkene_any", "alkene_terminal"}

# target column each property model is trained on when it differs from the
# property name: density trains on its completed label set (library values +
# group-contribution completion), matching the individual-compound policy.
TRAIN_COLUMN = {"fuel_density_g_ml_298K": "density_label_g_ml"}

# production EI+RI -> Z translator (5,225 individuals + 30,000 whole-library
# co-eluting spectra, see train_uniform_augmented_translator.py).  Kept as a
# file path so that this module -- and therefore the real-sample pipeline --
# uses exactly the same translator as the co-eluting and virtual-mixture layers.
PRODUCTION_Z_MODEL = (
    BASE / "models" / "simulated_mixture_validation"
    / "augmented_z_model.joblib"
)


def load_z_model(data: pd.DataFrame, zcols: list[str]):
    """Return (translator, binary_columns).

    The paper's main route translates every kind of spectrum with one model
    trained on the individual compounds *plus* the synthetic co-eluting
    spectra; a per-layer translator would make the layers incomparable.  Only
    if the production file is missing does this fall back to a pure
    individual-compound fit.
    """
    binary = [c for c in zcols if set(pd.unique(data[c].dropna())) <= {0.0, 1.0}]
    if PRODUCTION_Z_MODEL.exists():
        return joblib.load(PRODUCTION_Z_MODEL), binary
    return fit_z_model(data, zcols)


def reg(seed: int = 2026) -> Pipeline:
    return Pipeline([
        ("i", SimpleImputer()),
        ("m", ExtraTreesRegressor(n_estimators=80, random_state=seed, min_samples_leaf=2, n_jobs=1)),
    ])


def load_training_data() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    base = BASE
    data = pd.read_csv(base / "data" / "modeling" / "modeling_table.csv")
    schema = json.loads((base / "data" / "modeling" / "modeling_schema.json").read_text(encoding="utf-8"))
    fp = pd.read_csv(base / "data" / "fingerprints" / "hydrocarbon_substructure_fingerprints.csv")
    job = pd.read_csv(base / "data" / "modeling" / "hybrid_properties.csv")
    cn = pd.read_csv(base / "data" / "modeling" / "cetane_clean.csv")
    corr = pd.read_csv(base / "data" / "modeling" / "corrected_Tb_K.csv")
    thermo = pd.read_csv(base / "data" / "modeling" / "completed_thermo_properties.csv")
    pub = pd.read_csv(base / "data" / "modeling" / "completed_pubchem_properties.csv")
    den = pd.read_csv(base / "data" / "modeling" / "completed_density_labels.csv")
    data = data.merge(fp, on="identifier", how="inner").merge(job, on="identifier", how="left")
    data = data.merge(cn[["identifier", "cetane_number_clean", "cetane_number_gc"]], on="identifier", how="left")
    data = data.merge(corr[["identifier", "Tb_K"]], on="identifier", how="left")
    data = data.merge(
        thermo[["identifier", "Hfus_J_mol", "Hfus_source", "density_g_ml", "density_source"]],
        on="identifier", how="left",
    )
    # full-coverage MW / XLogP: PubChem where it exists, exact-formula MW and
    # calibrated-Crippen XLogP elsewhere (only 118 molecules, see
    # complete_pubchem_labels.py).  The PubChem-only columns are left untouched
    # so the mixture pipeline keeps its current inputs.
    data = data.merge(
        pub[["identifier", "MW_full_g_mol", "MW_source", "XLogP_full", "XLogP_source"]],
        on="identifier", how="left",
    )
    # density: 2,088 library values + a group-contribution completion of the rest
    # (the Rackett completion of ``density_full_g_ml`` is kept for reference only)
    data = data.merge(
        den[["identifier", "density_lib_g_ml", "density_gc_g_ml",
             "density_label_g_ml", "density_label_source"]],
        on="identifier", how="left",
    )
    data["chem_Tb_K"] = data["Tb_K"]
    for s, d in [("joback_Tm_K", "chem_Tm_K"), ("joback_Tc_K", "chem_Tc_K"),
                 ("joback_Pc_Pa", "chem_Pc_Pa"), ("joback_Vc_m3_mol", "chem_Vc_m3_mol"), ("joback_Hfg_J_mol", "chem_Hfg_J_mol")]:
        data[d] = data[d].fillna(data[s])
    # Full-coverage Hfus (exp + Marrero-Gani) for mixture ground truth.
    data["Hfus_full_J_mol"] = data["Hfus_J_mol"]
    # Density: the *library* value where it exists and the group-contribution
    # completion elsewhere (see complete_density_labels.py).  This is the density
    # used everywhere downstream - mixture ground truth, per-slice models - and
    # it replaces the Rackett fill (``density_g_ml``), which blew up to
    # 1.4-2.5 g/mL for C>28 and cost 0.10 R2 when used as a label.  The Rackett
    # column is kept as ``density_rackett_g_ml`` for reference only.
    data["density_rackett_g_ml"] = data["density_g_ml"]
    data["density_full_g_ml"] = data["density_label_g_ml"]
    spec = [c for c in schema["spectrum_columns"] if MZ_LO <= int(c.split("_")[1]) <= MZ_HI]
    desc = [c for c in schema["descriptor_columns"] if c not in DROP_DESC]
    fpcols = [c for c in fp.columns if c != "identifier" and c not in REMOVE_FP]
    zcols = desc + fpcols
    # keep the frozen eight-target order (see KEEP_TARGETS), intersected with
    # the columns actually present in the modelling table
    props = [p for p in KEEP_TARGETS if p in data.columns and p not in DROP_TARGETS]
    return data, spec, zcols, props


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm", default=str(BASE / "data/exp/virtual_molecules.csv"))
    parser.add_argument("--out", default=str(BASE / "data/exp/mixture_properties.csv"))
    args = parser.parse_args()

    data, spec, zcols, props = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    mw_term = 12.011 * data["desc_C"] + 1.008 * data["desc_H"]
    data["C_mass_frac"] = 12.011 * data["desc_C"] / mw_term
    data["H_mass_frac"] = 1.008 * data["desc_H"] / mw_term
    data["HC_ratio"] = data["desc_H"] / data["desc_C"]
    props = props + ["C_mass_frac", "H_mass_frac", "HC_ratio"]
    xcols = spec + ["desc_airi_ri"]

    # 中间层模型：EI+RI -> Z（生产 translator：单体 + 3 万条全库共流出谱）
    z_model, binary = load_z_model(data, zcols)

    # 性质模型：C 路线（Z+RI -> property）与 G 路线（EI+RI+Z -> property）
    prop_models = {}
    g_models = {}
    # property model input = the fingerprint Z alone (RI is only used upstream
    # in the EI+RI -> Z translator; see the ablation in figures 23/24/33)
    prop_input = list(zcols)
    for p in props:
        col = TRAIN_COLUMN.get(p, p)
        sub = data[data[col].notna()]
        if len(sub) < 20:
            continue
        m = reg()
        m.fit(sub[prop_input], sub[col].to_numpy(float))
        prop_models[p] = m
        g_train = pd.concat(
            [sub[xcols].reset_index(drop=True), sub[zcols].reset_index(drop=True)],
            axis=1,
        )
        gm = reg()
        gm.fit(g_train, sub[col].to_numpy(float))
        g_models[p] = gm

    # ---- non-additive bulk properties: mixture-augmented models -----------
    # Density and boiling point cannot be read off the mean fingerprint; the
    # model must be trained on (mean fingerprint -> mixed value) pairs, exactly
    # like every other mixture-augmented property model.  Both routes (C: mean
    # Z -> property, G: spectrum + RI + mean Z -> property) are augmented with
    # the same synthetic co-eluting spectra.
    from mixture_sampling import draw_components, N_MIX
        from z_fingerprint_model import mass_columns
    _spec_cols = mass_columns(data)
    _lib = sum_normalize(data[_spec_cols].to_numpy(float))
    _cf = np.ones(len(data))
    _ri_lib = data["desc_airi_ri"].to_numpy(float)
    _idx, _wm = draw_components(data, np.random.default_rng(2026), N_MIX)
    _Xmix_spec = np.einsum("ij,ijk->ik", _wm / _cf[_idx], _lib[_idx])
    _Xmix_spec /= _Xmix_spec.sum(1, keepdims=True)
    _Xmix_ri = (_wm * _ri_lib[_idx]).sum(1)
    _Zbar = np.einsum("ij,ijk->ik", _wm, data[zcols].to_numpy(float)[_idx])
    _G_aug = np.column_stack([_Xmix_spec, _Xmix_ri, _Zbar])
    for p, rule in (("fuel_density_g_ml_298K", "reciprocal"),
                    ("chem_Tb_K", "mean")):
        col = TRAIN_COLUMN.get(p, p)
        v = data[col].to_numpy(float)
        if not np.isfinite(v).any():
            continue
        safe = np.where(np.isfinite(v[_idx]), v[_idx], 0.0)
        y = (1.0 / np.sum(_wm / np.clip(safe, 1e-9, None), axis=1)
             if rule == "reciprocal" else np.sum(_wm * safe, axis=1))
        keep = np.isfinite(v[_idx]).all(1)
        sub = data[data[col].notna()]
        m = reg()
        m.fit(np.vstack([sub[prop_input].to_numpy(float), _Zbar[keep]]),
              np.concatenate([v[sub.index], y[keep]]))
        prop_models[p] = m
        gm = reg()
        g_train = pd.concat([sub[xcols].reset_index(drop=True),
                             sub[zcols].reset_index(drop=True)], axis=1)
        gm.fit(np.vstack([g_train.to_numpy(float), _G_aug[keep]]),
               np.concatenate([v[sub.index], y[keep]]))
        g_models[p] = gm

    # 虚拟分子推理
    vm = pd.read_csv(args.vm)
    vm_ri = vm["ri"].to_numpy(float)
    Z = predict_z_model(z_model, vm, zcols, binary)
    Z["desc_airi_ri"] = vm_ri

    area = vm["area"].to_numpy(float)
    w = area / area.sum()

    # 每个虚拟分子的性质预测 + 按该量的正确混合规则聚合
    #   mass-additive 量（Tb/Tm/XLogP）-> 质量分数加权平均
    #   density                        -> 比容混合 1/rho = sum(w_i/rho_i)
    #   MW                             -> 数均 1/sum(w_i/M_i)
    #   Tc/Pc/Vc                       -> Kay 法则（摩尔分数加权）
    KAY = {"chem_Tc_K", "chem_Pc_Pa", "chem_Vc_m3_mol"}
    rows = []
    mixture = {}
    mixture_g = {}
    vm_spec = pd.DataFrame(
        sum_normalize(vm[[f"mz_{i}" for i in range(MZ_LO, MZ_HI + 1)]].to_numpy(float)),
        columns=spec,
    )
    g_test = pd.concat([
        vm_spec.reset_index(drop=True),
        vm["ri"].rename("desc_airi_ri").reset_index(drop=True),
        Z[zcols].reset_index(drop=True),
    ], axis=1)
    mw_pred = 12.011 * Z["desc_C"].to_numpy(float) + 1.008 * Z["desc_H"].to_numpy(float)
    z_frac = mass_to_mole(w, mw_pred)
    for p, m in prop_models.items():
        pred = m.predict(Z[prop_input])
        pred_g = g_models[p].predict(g_test)
        if p in KAY:
            mixture[p] = float(np.sum(z_frac * pred))
            mixture_g[p] = float(np.sum(z_frac * pred_g))
        elif p == "fuel_density_g_ml_298K":
            # non-additive: specific-volume mixing, 1/rho = sum(w_i / rho_i)
            mixture[p] = density_mix(w, pred)
            mixture_g[p] = density_mix(w, pred_g)
        elif p == "PubChem_MolecularWeight":
            mixture[p] = float(1.0 / np.sum(w / np.clip(pred, 1e-9, None)))
            mixture_g[p] = float(1.0 / np.sum(w / np.clip(pred_g, 1e-9, None)))
        else:
            mixture[p] = float(np.sum(pred * w))
            mixture_g[p] = float(np.sum(pred_g * w))
        rows.append({"property": p, "C_EI_RI_to_Z": mixture[p], "G_EI_RI_plus_predZ": mixture_g[p]})
    # MW 也可以直接从翻译出的指纹读出（12.011 C + 1.008 H，数均）：作为
    # "指纹 <-> 性质模型" 的一致性检验一并报告（两条路线共用同一指纹，故相同）
    mw_from_z = float(1.0 / np.sum(w / mw_pred))
    rows.append({"property": "MW_from_Z", "C_EI_RI_to_Z": mw_from_z,
                 "G_EI_RI_plus_predZ": mw_from_z})
    # H/C is a ratio of two additive quantities: the mixture value must be
    # (sum of H atoms) / (sum of C atoms), NOT the mass-weighted mean of the
    # per-slice H/C values (which carries a Jensen bias).  Recompute it from
    # the aggregated elemental mass fractions.
    for d in (mixture, mixture_g):
        d["HC_ratio"] = (d["H_mass_frac"] / 1.008) / (d["C_mass_frac"] / 12.011)
        for r in rows:
            if r["property"] == "HC_ratio":
                key = "C_EI_RI_to_Z" if d is mixture else "G_EI_RI_plus_predZ"
                r[key] = d["HC_ratio"]
    # C route: the elemental composition of the mixture is *read off the
    # translated Z* (desc_C / desc_H) and mass-weighted -- no separate property
    # model is needed, because Z already carries the carbon and hydrogen count.
    c_z = Z["desc_C"].to_numpy(float)
    h_z = Z["desc_H"].to_numpy(float)
    mw_z = 12.011 * c_z + 1.008 * h_z
    cm_z = float(np.sum(12.011 * c_z / mw_z * w))
    hm_z = float(np.sum(1.008 * h_z / mw_z * w))
    for name, val in (("C_mass_frac", cm_z), ("H_mass_frac", hm_z),
                      ("HC_ratio", (hm_z / 1.008) / (cm_z / 12.011))):
        mixture[name] = val
        for r in rows:
            if r["property"] == name:
                r["C_EI_RI_to_Z"] = val
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(args.out, index=False)

    # ---- per-slice table for the distillation (TBP) curve ------------------
    # One row per 0.1-min slice: its mass weight and the boiling point read from
    # the translated fingerprint.  The curve itself is produced by
    # distillation_curve.py, so the real-sample numbers always come from this
    # same run.
    if "chem_Tb_K" in prop_models:
        Zsl = Z[["desc_C", "desc_H", "desc_DBE", "desc_ring_count",
                 "desc_is_chain_alkane", "desc_is_aromatic"]].copy()
        pd.DataFrame({
            "rt": vm["rt"].to_numpy(float),
            "ri": vm_ri,
            "mass_frac": w,
            "Tb_K": prop_models["chem_Tb_K"].predict(Z[prop_input]),
            "Tb_G_K": g_models["chem_Tb_K"].predict(g_test),
        }).join(Zsl.reset_index(drop=True)).to_csv(
            Path(args.out).with_name("real_slices_tb.csv"), index=False)

    # 加权聚合结构表征
    Z_weighted = (Z[zcols].to_numpy(float) * w[:, None]).sum(axis=0)
    print(f"虚拟分子数={len(vm)}, 性质数={len(prop_models)}")
    print("混合物性质预测（面积加权平均）:")
    for p, v in sorted(mixture.items(), key=lambda x: x[0]):
        print(f"  {p}: C={v:.4f}, G={mixture_g[p]:.4f}")


if __name__ == "__main__":
    main()
