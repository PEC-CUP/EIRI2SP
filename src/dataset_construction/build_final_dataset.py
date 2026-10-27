#!/usr/bin/env python
"""Build the final, provenance-aware MS2Prop dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROPERTY_META = {
    "chem_MW": ("chemicals", "A"),
    "chem_Tb_K": ("chemicals", "B"),
    "chem_Tm_K": ("chemicals", "B"),
    "chem_Tc_K": ("chemicals", "B"),
    "chem_Pc_Pa": ("chemicals", "B"),
    "chem_Vc_m3_mol": ("chemicals", "B"),
    "chem_omega": ("chemicals", "B"),
    "chem_Hfg_J_mol": ("chemicals", "B"),
    "chem_Hfl_J_mol": ("chemicals", "B"),
    "chem_Hfus_J_mol": ("chemicals", "B"),
    "chem_S0g_J_molK": ("chemicals", "B"),
    "PubChem_Complexity": ("PubChem computed", "A"),
    "PubChem_ExactMass": ("PubChem computed", "A"),
    "PubChem_HeavyAtomCount": ("PubChem computed", "A"),
    "PubChem_MolecularWeight": ("PubChem computed", "A"),
    "PubChem_MonoisotopicMass": ("PubChem computed", "A"),
    "PubChem_RotatableBondCount": ("PubChem computed", "A"),
    "PubChem_TPSA": ("PubChem computed", "A"),
    "PubChem_XLogP": ("PubChem computed", "A"),
    "fuel_HHV_MJ_kg": ("thermo", "B"),
    "fuel_density_g_ml_298K": ("thermo", "B"),
    "fuel_flash_point_K": ("thermo", "C"),
    "cetane_number_gc": ("Kubic 2016 group-contribution ANN", "B"),
    "RON_gc": ("Kubic 2016 group-contribution ANN", "B"),
    "MON_gc": ("Kubic 2016 group-contribution ANN", "B"),
}


def parse_temp_c(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.replace("\u00b0", " deg ")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:deg)?\s*([CF])", s)
    if not m:
        return None
    v = float(m.group(1))
    return (v - 32) * 5 / 9 + 273.15 if m.group(2) == "F" else v + 273.15


def parse_density(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    if "relative" in s.lower() or "water" in s.lower():
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*g", s, re.I)
    return float(m.group(1)) if m else None


def main() -> None:
    base = BASE
    out_dir = base / "data" / "modeling"
    modeling = pd.read_csv(out_dir / "modeling_table.csv")
    fp = pd.read_csv(base / "data" / "fingerprints" / "hydrocarbon_substructure_fingerprints.csv")
    cn = pd.read_csv(base / "data" / "properties_experimental" / "group_contribution_cn.csv")
    nist = pd.read_csv(base / "data" / "properties_experimental" / "nist_phase_long.csv")

    nist = nist[nist["source_type"] == "experimental"]
    nist["value"] = pd.to_numeric(nist["value"], errors="coerce")
    nist_wide = nist.pivot_table(index="identifier", columns="property_name", values="value", aggfunc="median").reset_index()

    pub = pd.read_csv(base / "data" / "properties_experimental" / "pubchem_experimental_long.csv")
    pub["exp_flash_K"] = pub["raw_text"].where(pub["property_name"] == "Flash Point").map(parse_temp_c)
    pub["exp_density_gml"] = pub["raw_text"].where(pub["property_name"] == "Density").map(parse_density)
    pub_exp = (
        pub.dropna(subset=["exp_flash_K", "exp_density_gml"], how="all")
        .groupby("identifier")[["exp_flash_K", "exp_density_gml"]]
        .median()
        .reset_index()
    )

    data = modeling.merge(fp, on="identifier", how="inner").merge(cn, on="identifier", how="left")
    data = data.merge(nist_wide, on="identifier", how="left").merge(pub_exp, on="identifier", how="left")

    keep_props = list(PROPERTY_META)
    cols = ["identifier", "SMILES", "FORMULA", "hydrocarbon_class", "NAME", "CASNO"]
    cols += [c for c in modeling.columns if c.startswith("mz_")]
    cols += [c for c in ["desc_airi_ri"] if c in data.columns]
    desc_cols = [c for c in modeling.columns if c.startswith("desc_") and c not in {
        "desc_MW", "desc_exact_mass", "desc_nominal_mass", "desc_carbon_mass_fraction",
        "desc_hydrogen_mass_fraction", "desc_airi_ri", "desc_airi_ri_stddev",
        "desc_airi_ri_confidence_interval", "desc_dbe_ge_1", "desc_carbon_skeleton_diameter",
        "desc_component_count", "desc_c_5_10", "desc_c_11_16", "desc_c_17_24", "desc_c_25_40",
    }]
    fp_cols = [c for c in fp.columns if c != "identifier"]
    cols += desc_cols + fp_cols
    cols += keep_props
    cols += [c for c in nist_wide.columns if c.startswith("exp_") and c != "identifier"]
    cols += ["exp_flash_K", "exp_density_gml"]
    cols = [c for c in cols if c in data.columns]

    final = data[cols].copy()
    final.to_csv(out_dir / "final_dataset.csv", index=False)

    schema = {
        "n_molecules": int(len(final)),
        "n_spectrum_features": int(sum(1 for c in final.columns if c.startswith("mz_"))),
        "ri_columns": ["desc_airi_ri"],
        "desc_columns": desc_cols,
        "fingerprint_columns": fp_cols,
        "n_Z": len(desc_cols) + len(fp_cols),
        "property_meta": PROPERTY_META,
        "experimental_columns": [c for c in final.columns if c.startswith("exp_")],
    }
    (out_dir / "final_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in schema.items() if k not in ("desc_columns", "fingerprint_columns")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
