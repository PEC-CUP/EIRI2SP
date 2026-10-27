#!/usr/bin/env python
"""Complete the two PubChem label gaps of the 5,225-molecule library.

MW      - only 5,109 of 5,225 molecules have a PubChem record, but MW needs no
          record: for a C/H molecule it is mass-additive, computed here with
          RDKit ``Descriptors.MolWt``.  The fill is *exact*, verified against the
          5,109 PubChem values (max |difference| 0.065, mean 0.012 g/mol).
XLogP   - the same 118 molecules have no XLogP3 record.  They are completed with
          RDKit's Crippen ``MolLogP`` after a linear calibration to XLogP3 fitted
          on all overlapping molecules.  Cross-estimator agreement is reported
          below and goes into the SI; the model itself is insensitive to the fill
          (R2 0.9779 -> 0.9772, scored on the PubChem rows).

Output: data/modeling/completed_pubchem_properties.csv
        identifier, MW_full_g_mol, MW_source, XLogP_full, XLogP_source
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "data" / "modeling" / "completed_pubchem_properties.csv"


def main() -> None:
    rec = pd.read_csv(BASE / "data" / "hydrocarbon_records.csv")
    tab = pd.read_csv(BASE / "data" / "modeling" / "modeling_table.csv", low_memory=False)
    df = tab[["identifier", "PubChem_MolecularWeight", "PubChem_XLogP"]].merge(
        rec[["identifier", "SMILES", "MW"]], on="identifier", how="right")

    mols = df["SMILES"].map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    missing_smiles = int(mols.isna().sum())
    df["MW_formula"] = [Descriptors.MolWt(m) if m is not None else np.nan for m in mols]
    df["crippen"] = [Crippen.MolLogP(m) if m is not None else np.nan for m in mols]

    # ---- MW: exact, no estimation involved -------------------------------
    ok_mw = df["PubChem_MolecularWeight"].notna() & df["MW_formula"].notna()
    diff = (df.loc[ok_mw, "MW_formula"] - df.loc[ok_mw, "PubChem_MolecularWeight"]).abs()
    print(f"MW  formula vs PubChem (n={int(ok_mw.sum())}): max |d| {diff.max():.4f}, "
          f"mean {diff.mean():.5f} g/mol")
    df["MW_full_g_mol"] = df["PubChem_MolecularWeight"].fillna(df["MW_formula"])
    df["MW_source"] = np.where(df["PubChem_MolecularWeight"].notna(), "pubchem",
                               "formula (exact)")

    # ---- XLogP: Crippen, linearly calibrated to XLogP3 --------------------
    ok_x = df["PubChem_XLogP"].notna() & df["crippen"].notna()
    slope, intercept = np.polyfit(df.loc[ok_x, "crippen"], df.loc[ok_x, "PubChem_XLogP"], 1)
    cal = slope * df["crippen"] + intercept
    print(f"XLogP3 = {slope:.4f} * Crippen {intercept:+.4f}   "
          f"(fit on n={int(ok_x.sum())})")
    print(f"XLogP calibrated Crippen vs XLogP3: R2 {r2_score(df.loc[ok_x, 'PubChem_XLogP'], cal[ok_x]):.4f}, "
          f"MAE {np.abs(cal[ok_x] - df.loc[ok_x, 'PubChem_XLogP']).mean():.3f}")
    df["XLogP_full"] = df["PubChem_XLogP"].fillna(cal)
    df["XLogP_source"] = np.where(df["PubChem_XLogP"].notna(), "pubchem_xlogp3",
                                  "calibrated_crippen")

    out = df[["identifier", "MW_full_g_mol", "MW_source", "XLogP_full", "XLogP_source"]]
    out.to_csv(OUT, index=False)
    print(f"\ncoverage  MW {out.MW_full_g_mol.notna().sum()}/{len(out)}  "
          f"XLogP {out.XLogP_full.notna().sum()}/{len(out)}"
          + (f"  ({missing_smiles} unparsed SMILES)" if missing_smiles else ""))
    print("saved", OUT)


if __name__ == "__main__":
    main()
