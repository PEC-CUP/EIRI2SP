#!/usr/bin/env python
"""Strict, structure-based (RDKit) classification of the 5,225-hydrocarbon library.

The legacy ``hydrocarbon_class`` column in ``data/hydrocarbon_records.csv`` was
produced with SMILES-string heuristics and is known to be wrong for a number of
molecules (e.g. alkynes counted as cycloalkanes, rings missed).  This script
re-derives an exclusive, complete four-way class from the molecular graph:

    aromatic            : molecule contains at least one aromatic ring
    unsaturated         : no aromatic ring, but contains a non-aromatic C=C or C#C
    saturated cyclic    : no aromatic ring, no C=C/C#C, contains at least one ring
    saturated acyclic   : no aromatic ring, no C=C/C#C, no ring

Precedence is aromatic > unsaturated > saturated cyclic > saturated acyclic, so
every molecule lands in exactly one class.  The output also records the carbon
number and the DBE used by the dataset Sankey figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

BASE = Path(__file__).resolve().parents[2]
RECORDS = BASE / "data" / "hydrocarbon_records.csv"
OUT = BASE / "data" / "hydrocarbon_class_strict.csv"

ALKENE = Chem.MolFromSmarts("[#6]=[#6]")
ALKYNE = Chem.MolFromSmarts("[#6]#[#6]")

CLASS_ORDER = ["Saturated acyclic", "Saturated cyclic", "Unsaturated", "Aromatic"]


def classify(mol: Chem.Mol) -> str:
    has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    if has_aromatic:
        return "Aromatic"
    if mol.HasSubstructMatch(ALKENE) or mol.HasSubstructMatch(ALKYNE):
        return "Unsaturated"
    if mol.GetRingInfo().NumRings() > 0:
        return "Saturated cyclic"
    return "Saturated acyclic"


def main() -> None:
    rec = pd.read_csv(RECORDS)
    classes = []
    for smi in rec["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            classes.append("Unparsable")
            continue
        classes.append(classify(mol))

    # DBE from formula-style relation: DBE = C - H/2 + 1 for CxHy hydrocarbons
    dbe = rec["C"] - rec["H"] / 2.0 + 1.0
    out = pd.DataFrame({
        "identifier": rec["identifier"],
        "SMILES": rec["SMILES"],
        "C": rec["C"],
        "H": rec["H"],
        "DBE": dbe.astype(float),
        "class_strict": classes,
    })
    out.to_csv(OUT, index=False)

    counts = out["class_strict"].value_counts()
    print(f"molecules: {len(out)}")
    for c in CLASS_ORDER:
        print(f"  {c:<20} {int(counts.get(c, 0)):>5}")
    other = counts.drop(index=[c for c in CLASS_ORDER if c in counts.index], errors="ignore")
    if len(other):
        print("  other:", dict(other))
    print("saved", OUT)


if __name__ == "__main__":
    main()
