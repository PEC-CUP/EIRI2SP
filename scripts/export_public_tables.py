"""Create the public compound, fingerprint and property tables.

The input modelling tables contain licensed EI spectra and are therefore
expected only in a local, git-ignored directory. This script exports the
public structure, fingerprint, split and property tables used in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from z_labels import ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dataset", default=str(ROOT / "data" / "modeling" / "final_dataset.csv"))
    parser.add_argument("--class-table", default=str(ROOT / "data" / "library" / "hydrocarbon_class_strict.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    return parser.parse_args()


def split_assignment(n: int) -> np.ndarray:
    train, _ = train_test_split(np.arange(n), test_size=0.2, random_state=2026)
    split = np.full(n, "test", dtype=object)
    split[train] = "train"
    return split


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    data = pd.read_csv(args.final_dataset)
    classes = pd.read_csv(args.class_table)[["identifier", "class_strict"]]
    modeling = ROOT / "data" / "modeling"
    joback = pd.read_csv(modeling / "hybrid_properties.csv")
    corrected_tb = pd.read_csv(modeling / "corrected_Tb_K.csv")
    pubchem = pd.read_csv(modeling / "completed_pubchem_properties.csv")
    density = pd.read_csv(modeling / "completed_density_labels.csv")
    data = data.merge(joback, on="identifier", how="left")
    data = data.merge(
        corrected_tb[["identifier", "Tb_K", "Tb_K_source"]],
        on="identifier",
        how="left",
    )
    data = data.merge(
        pubchem[["identifier", "MW_full_g_mol", "MW_source", "XLogP_full", "XLogP_source"]],
        on="identifier",
        how="left",
    )
    data = data.merge(
        density[["identifier", "density_label_g_ml", "density_label_source"]],
        on="identifier",
        how="left",
    )
    for source, target in (
        ("joback_Tm_K", "chem_Tm_K"),
        ("joback_Tc_K", "chem_Tc_K"),
        ("joback_Pc_Pa", "chem_Pc_Pa"),
        ("joback_Vc_m3_mol", "chem_Vc_m3_mol"),
    ):
        data[target] = data[target].fillna(data[source])

    library = pd.DataFrame({
        "identifier": data["identifier"],
        "SMILES": data["SMILES"],
        "split": split_assignment(len(data)),
    }).merge(classes, on="identifier", how="left")
    library.to_csv(out / "library" / "compound_library.csv", index=False)

    fingerprint = data[["identifier"] + ORDER].merge(
        library[["identifier", "split"]], on="identifier", how="left"
    )
    fingerprint.to_csv(out / "fingerprints" / "fingerprints_47bit.csv", index=False)

    properties = pd.DataFrame({
        "identifier": data["identifier"],
        "MW_g_mol": data["MW_full_g_mol"],
        "MW_source": data["MW_source"],
        "Tb_K": data["Tb_K"],
        "Tb_source": data["Tb_K_source"],
        "XLogP": data["XLogP_full"],
        "XLogP_source": data["XLogP_source"],
        "Vc_cm3_mol": data["chem_Vc_m3_mol"] * 1e6,
        "Vc_source": np.where(data["chem_Vc_m3_mol"].notna(), "library_or_Joback", "missing"),
        "Pc_MPa": data["chem_Pc_Pa"] / 1e6,
        "Pc_source": np.where(data["chem_Pc_Pa"].notna(), "library_or_Joback", "missing"),
        "density_g_mL": data["density_label_g_ml"],
        "density_source": data["density_label_source"],
        "Tc_K": data["chem_Tc_K"],
        "Tc_source": np.where(data["chem_Tc_K"].notna(), "library_or_Joback", "missing"),
        "Tm_K": data["chem_Tm_K"],
        "Tm_source": np.where(data["chem_Tm_K"].notna(), "library_or_Joback", "missing"),
    }).merge(library[["identifier", "split"]], on="identifier", how="left")
    properties.to_csv(out / "properties" / "property_labels_8.csv", index=False)

    print("molecules:", len(library))
    print("split:", library["split"].value_counts().to_dict())
    print("fingerprint positions:", len(ORDER))


if __name__ == "__main__":
    main()
