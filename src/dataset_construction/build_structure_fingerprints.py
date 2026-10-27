#!/usr/bin/env python
"""Build RDKit molecular fingerprints/descriptors for the screened molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


RDKIT_DESCRIPTOR_FUNCS = {
    "rdkit_MolWt": Descriptors.MolWt,
    "rdkit_ExactMolWt": Descriptors.ExactMolWt,
    "rdkit_MolLogP": Descriptors.MolLogP,
    "rdkit_TPSA": Descriptors.TPSA,
    "rdkit_NumRotatableBonds": Descriptors.NumRotatableBonds,
    "rdkit_RingCount": Descriptors.RingCount,
    "rdkit_NumAromaticRings": Descriptors.NumAromaticRings,
    "rdkit_NumAliphaticRings": Descriptors.NumAliphaticRings,
    "rdkit_FractionCSP3": Descriptors.FractionCSP3,
    "rdkit_HeavyAtomCount": Descriptors.HeavyAtomCount,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--out-dir", default=str(BASE / "data/fingerprints"))
    parser.add_argument("--morgan-bits", type=int, default=512)
    parser.add_argument("--radius", type=int, default=2)
    return parser.parse_args()


def bits_to_dict(prefix: str, bitstring: str) -> dict[str, int]:
    return {f"{prefix}_{i}": int(bit) for i, bit in enumerate(bitstring)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records)
    morgan_gen = GetMorganGenerator(radius=args.radius, fpSize=args.morgan_bits)

    fp_rows = []
    desc_rows = []
    errors = []
    for rec in records.itertuples(index=False):
        identifier = getattr(rec, "identifier")
        smiles = getattr(rec, "SMILES")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            errors.append({"identifier": identifier, "SMILES": smiles, "error": "MolFromSmiles failed"})
            continue
        morgan = morgan_gen.GetFingerprint(mol).ToBitString()
        maccs = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        fp_row = {"identifier": identifier}
        fp_row.update(bits_to_dict("morgan", morgan))
        fp_row.update(bits_to_dict("maccs", maccs))
        fp_rows.append(fp_row)

        desc_row = {"identifier": identifier}
        for name, func in RDKIT_DESCRIPTOR_FUNCS.items():
            try:
                desc_row[name] = func(mol)
            except Exception:
                desc_row[name] = None
        desc_rows.append(desc_row)

    fp = pd.DataFrame(fp_rows)
    desc = pd.DataFrame(desc_rows)
    fp.to_csv(out_dir / "structure_fingerprints.csv", index=False)
    desc.to_csv(out_dir / "rdkit_descriptors.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "fingerprint_errors.csv", index=False)
    schema = {
        "morgan_bits": args.morgan_bits,
        "radius": args.radius,
        "maccs_bits": 167,
        "n_molecules": int(len(fp)),
        "fingerprint_columns": [c for c in fp.columns if c != "identifier"],
        "rdkit_descriptor_columns": [c for c in desc.columns if c != "identifier"],
    }
    (out_dir / "fingerprint_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in schema.items() if not k.endswith("_columns")}, indent=2))


if __name__ == "__main__":
    main()
