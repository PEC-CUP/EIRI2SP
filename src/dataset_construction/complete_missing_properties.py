#!/usr/bin/env python
"""Fill missing pure-compound thermo labels by pure-substance equations.

- Hfus (enthalpy of fusion): Marrero-Gani (2001) group contribution
    Hfus[kJ/mol] = Hfus0 + sum_g N_g * Hfus1_g ,  Hfus0 = -2.806 kJ/mol
    (first/second/third-order groups share one flat coefficient table `hm`).
- density (298 K, g/mL): Rackett equation from Joback critical properties
    Vsat = Vc * Zc^((1 - T/Tc)^(2/7)),  Zc = Pc*Vc/(R*Tc)
    rho = MW / Vsat.

Experimental values always win; missing values are filled by the equations and
labelled `marrero_gani` / `rackett`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_marrero_gani_properties import count_groups, COEFF


def mg_hfus_jmol(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = count_groups(mol)
    h = -2.806  # Hfus0, kJ/mol
    for g, n in counts.items():
        h += n * COEFF["hm"][g]
    return h * 1000.0  # J/mol


def rackett_density(row) -> float | None:
    Tc = row["joback_Tc_K"]
    Pc = row["joback_Pc_Pa"]
    Vc = row["joback_Vc_m3_mol"]
    sm = row["SMILES"]
    if not (Tc > 0 and Pc > 0 and Vc > 0):
        return None
    R = 8.314
    T = 298.15
    Zc = Pc * Vc / (R * Tc)
    Tr = T / Tc
    if Tr >= 1 or Zc <= 0 or Zc >= 1:
        return None
    Vs = Vc * Zc ** ((1 - Tr) ** (2 / 7))
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        return None
    C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    H = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
    MW = 12.011 * C + 1.008 * H
    return (MW / Vs) * 1e-6  # g/mL


def main() -> None:
    base = BASE
    data = pd.read_csv(base / "data" / "modeling" / "modeling_table.csv")
    job = pd.read_csv(base / "data" / "modeling" / "hybrid_properties.csv")
    rec = pd.read_csv(base / "data" / "hydrocarbon_records.csv")
    df = data.merge(job, on="identifier").merge(rec[["identifier", "SMILES"]], on="identifier")

    df["Hfus_mg_J_mol"] = df["SMILES"].map(mg_hfus_jmol)
    df["rho_rackett_g_ml"] = df.apply(rackett_density, axis=1)

    # validate against experimental where both exist
    sub_h = df.dropna(subset=["chem_Hfus_J_mol", "Hfus_mg_J_mol"])
    sub_r = df.dropna(subset=["fuel_density_g_ml_298K", "rho_rackett_g_ml"])
    print(f"Hfus  MG vs exp: n={len(sub_h)}  MAE={(sub_h.Hfus_mg_J_mol-sub_h.chem_Hfus_J_mol).abs().mean():.0f} J/mol  "
          f"MAPE={((sub_h.Hfus_mg_J_mol-sub_h.chem_Hfus_J_mol).abs()/sub_h.chem_Hfus_J_mol*100).mean():.1f}%")
    print(f"density Rackett vs exp: n={len(sub_r)}  MAE={(sub_r.rho_rackett_g_ml-sub_r.fuel_density_g_ml_298K).abs().mean():.3f} g/mL  "
          f"MAPE={((sub_r.rho_rackett_g_ml-sub_r.fuel_density_g_ml_298K).abs()/sub_r.fuel_density_g_ml_298K*100).mean():.1f}%")

    out = pd.DataFrame({
        "identifier": df["identifier"],
        "Hfus_exp_J_mol": df["chem_Hfus_J_mol"],
        "Hfus_calc_J_mol": df["Hfus_mg_J_mol"],
        "density_exp_g_ml": df["fuel_density_g_ml_298K"],
        "density_calc_g_ml": df["rho_rackett_g_ml"],
    })
    out["Hfus_J_mol"] = out["Hfus_exp_J_mol"].fillna(out["Hfus_calc_J_mol"])
    out["Hfus_source"] = np.where(out["Hfus_exp_J_mol"].notna(), "experimental", "marrero_gani")
    out["density_g_ml"] = out["density_exp_g_ml"].fillna(out["density_calc_g_ml"])
    out["density_source"] = np.where(out["density_exp_g_ml"].notna(), "experimental", "rackett")
    out.to_csv(base / "data" / "modeling" / "completed_thermo_properties.csv", index=False)
    print(f"saved {len(out)} rows -> data/modeling/completed_thermo_properties.csv")
    print("Hfus final coverage:", out["Hfus_J_mol"].notna().sum(), "/", len(out))
    print("density final coverage:", out["density_g_ml"].notna().sum(), "/", len(out))


if __name__ == "__main__":
    main()
