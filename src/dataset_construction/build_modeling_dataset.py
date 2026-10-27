#!/usr/bin/env python
"""Build the unified S, D, Y dataset for EI property modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import chemicals
except Exception:  # pragma: no cover
    chemicals = None


CHEMICALS_PROPS = {
    "chem_MW": "MW",
    "chem_Tb_K": "Tb",
    "chem_Tm_K": "Tm",
    "chem_Tc_K": "Tc",
    "chem_Pc_Pa": "Pc",
    "chem_Vc_m3_mol": "Vc",
    "chem_omega": "omega",
    "chem_Hfg_J_mol": "Hfg",
    "chem_Hfl_J_mol": "Hfl",
    "chem_Hfs_J_mol": "Hfs",
    "chem_Hfus_J_mol": "Hfus",
    "chem_S0g_J_molK": "S0g",
    "chem_S0l_J_molK": "S0l",
}


EXCLUDED_PROPERTY_TARGETS = {
    # Identifiers are useful for joins/provenance but are not physical targets.
    "PubChem_CID",
    # Raw viscosity spans many orders of magnitude and contains extreme
    # state/correlation outliers.  Use the log10 fuel viscosity targets instead.
    "fuel_dynamic_viscosity_Pa_s_298K",
    "fuel_kinematic_viscosity_m2_s_298K",
    "fuel_kinematic_viscosity_cSt_298K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--vectors", default=str(BASE / "data/eision_vectors.csv"))
    parser.add_argument("--descriptors", default=str(BASE / "data/descriptor_targets.csv"))
    parser.add_argument("--property-long", default=str(BASE / "data/properties/property_long.csv"))
    parser.add_argument("--fuel-property-matrix", default="", help="Optional wide fuel-property table with identifier and fuel_* columns.")
    parser.add_argument("--ri-table", default="", help="Optional measured or AIRI-predicted RI table with identifier and RI-like numeric columns.")
    parser.add_argument("--out-dir", default=str(BASE / "data/modeling"))
    parser.add_argument("--min-nonnull", type=int, default=200, help="Minimum non-null labels to keep a property target.")
    return parser.parse_args()


def normalize_cas(raw) -> str:
    text = str(raw or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if "-" in text:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 3:
        return ""
    cas = f"{int(digits[:-3])}-{digits[-3:-1]}-{digits[-1]}"
    return cas if valid_cas(cas) else ""


def valid_cas(cas: str) -> bool:
    digits = [int(ch) for ch in cas.replace("-", "")]
    if len(digits) < 3:
        return False
    checksum = sum((i + 1) * d for i, d in enumerate(reversed(digits[:-1]))) % 10
    return checksum == digits[-1]


def chemicals_properties(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    if chemicals is None:
        return pd.DataFrame({"identifier": records["identifier"]}), pd.DataFrame()

    for rec in records.itertuples(index=False):
        identifier = getattr(rec, "identifier")
        cas = normalize_cas(getattr(rec, "CASNO", ""))
        row = {"identifier": identifier, "CASRN": cas}
        if cas:
            for out_col, func_name in CHEMICALS_PROPS.items():
                func = getattr(chemicals, func_name)
                try:
                    value = func(cas)
                except Exception:
                    value = None
                row[out_col] = value
                if value is not None and not pd.isna(value):
                    long_rows.append(
                        {
                            "identifier": identifier,
                            "CASNO": getattr(rec, "CASNO", ""),
                            "CASRN": cas,
                            "SMILES": getattr(rec, "SMILES", ""),
                            "property_name": out_col,
                            "value": value,
                            "unit": unit_for(out_col),
                            "source_database": "chemicals",
                            "source_type": "open_database_or_correlation",
                            "source_reference": f"chemicals.{func_name}({cas})",
                            "quality_flag": "",
                        }
                    )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def unit_for(prop: str) -> str:
    if prop.endswith("_K"):
        return "K"
    if prop.endswith("_Pa"):
        return "Pa"
    if prop.endswith("_m3_mol"):
        return "m^3/mol"
    if prop.endswith("_J_mol"):
        return "J/mol"
    if prop.endswith("_J_molK"):
        return "J/mol/K"
    return ""


def load_external_property_matrix(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    long_df = pd.read_csv(p)
    if long_df.empty or "identifier" not in long_df.columns:
        return pd.DataFrame()
    numeric = long_df.copy()
    numeric["value_numeric"] = pd.to_numeric(numeric.get("value"), errors="coerce")
    matrix = numeric.dropna(subset=["value_numeric"]).pivot_table(
        index="identifier",
        columns="property_name",
        values="value_numeric",
        aggfunc="first",
    )
    return matrix.reset_index()


def load_wide_property_matrix(path: str, prefix: str = "") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty or "identifier" not in df.columns:
        return pd.DataFrame()
    keep = ["identifier"]
    for col in df.columns:
        if col == "identifier":
            continue
        if prefix and not col.startswith(prefix):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        keep.append(col)
    return df[keep]


def load_ri_matrix(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame()
    if "identifier" not in df.columns and "id_original" in df.columns:
        df = df.rename(columns={"id_original": "identifier"})
    if "identifier" not in df.columns:
        return pd.DataFrame()

    source_cols = {
        "predicted_ri": "desc_airi_ri",
        "predicted_ri_stddev": "desc_airi_ri_stddev",
        "confidence_interval": "desc_airi_ri_confidence_interval",
    }
    keep = ["identifier"]
    for col, out_col in source_cols.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        df[out_col] = numeric
        keep.append(out_col)
    return df[keep] if len(keep) > 1 else pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.records)
    vectors = pd.read_csv(args.vectors)
    descriptors = pd.read_csv(args.descriptors)

    chem_matrix, chem_long = chemicals_properties(records)
    external_matrix = load_external_property_matrix(args.property_long)
    fuel_matrix = load_wide_property_matrix(args.fuel_property_matrix, prefix="fuel_") if args.fuel_property_matrix else pd.DataFrame()
    ri_matrix = load_ri_matrix(args.ri_table) if args.ri_table else pd.DataFrame()
    if not ri_matrix.empty:
        descriptors = descriptors.merge(ri_matrix, on="identifier", how="left")

    prop = records[["identifier", "NAME", "CASNO", "SMILES", "FORMULA", "hydrocarbon_class"]].copy()
    for table in [chem_matrix, external_matrix, fuel_matrix]:
        if not table.empty:
            prop = prop.merge(table, on="identifier", how="left")

    reserved = {"identifier", "NAME", "CASNO", "SMILES", "FORMULA", "hydrocarbon_class", "CASRN"}
    prop_cols = []
    for col in prop.columns:
        if col in reserved:
            continue
        if col in EXCLUDED_PROPERTY_TARGETS:
            continue
        prop[col] = pd.to_numeric(prop[col], errors="coerce")
        if prop[col].notna().sum() >= args.min_nonnull:
            prop_cols.append(col)

    prop_keep = prop[[*reserved.intersection(prop.columns), *prop_cols]]
    modeling = vectors.merge(descriptors, on="identifier", how="inner").merge(
        prop_keep[["identifier", *prop_cols]], on="identifier", how="inner"
    )

    prop_keep.to_csv(out_dir / "property_matrix.csv", index=False)
    modeling.to_csv(out_dir / "modeling_table.csv", index=False)
    if not chem_long.empty:
        chem_long.to_csv(out_dir / "chemicals_property_long.csv", index=False)

    schema = {
        "spectrum_columns": [c for c in vectors.columns if c.startswith("mz_")],
        "descriptor_columns": [c for c in descriptors.columns if c.startswith("desc_")],
        "property_columns": prop_cols,
        "n_molecules": int(len(modeling)),
        "property_nonnull_counts": {c: int(prop[c].notna().sum()) for c in prop_cols},
    }
    (out_dir / "modeling_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
