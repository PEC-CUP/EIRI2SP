#!/usr/bin/env python
"""Build interpretable hydrocarbon substructure fingerprints.

The fingerprint is key-based: every bit corresponds to one explicit SMARTS
substructure (ring system, unsaturation, or branching pattern) and has a
stable, human-readable name.  Candidate substructures are generated
automatically from the hydrocarbon library, then filtered by prevalence so
that only informative bits are kept.

This is deliberately different from hashed Morgan bits, which are not
interpretable.  It is intended to serve as the interpretable part of the
EI+RI -> structural-probability-fingerprint -> property QSPR pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


# Human-named, hydrocarbon-relevant substructures that are hard to obtain by
# ring-system enumeration alone.  Every entry maps a stable name to a SMARTS.
MANUAL_SUBSTRUCTURES = {
    "alkene_any": "[#6]=[#6]",
    "alkene_terminal": "[#6]=[#6;H2]",
    "alkyne_any": "[#6]#[#6]",
    "alkyne_terminal": "[#6]#[#6;H1]",
    "quaternary_carbon": "[CX4;H0]([#6])([#6])([#6])[#6]",
    "tertiary_carbon": "[CX4;H1]([#6])([#6])[#6]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--out-dir", default=str(BASE / "data/fingerprints"))
    parser.add_argument("--min-prevalence", type=float, default=0.02)
    parser.add_argument("--max-prevalence", type=float, default=0.98)
    parser.add_argument("--min-ring-system-size", type=int, default=3)
    return parser.parse_args()


def ring_system_submol(mol: Chem.Mol, atom_indices: set[int]) -> Chem.Mol:
    """Return a submolecule containing only the ring-system atoms/bonds."""
    emol = Chem.EditableMol(Chem.Mol())
    idx_map: dict[int, int] = {}
    for a in sorted(atom_indices):
        idx_map[a] = emol.AddAtom(mol.GetAtomWithIdx(a))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in idx_map and j in idx_map:
            emol.AddBond(idx_map[i], idx_map[j], bond.GetBondType())
    sub = emol.GetMol()
    Chem.SanitizeMol(
        sub,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
    )
    return sub


def enumerate_ring_systems(mol: Chem.Mol) -> list[tuple[str, str]]:
    """Enumerate fused ring systems as (smarts, kind) pairs."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    merged: list[set[int]] = []
    for ring in rings:
        hit = None
        for group in merged:
            if ring & group:
                hit = group
                break
        if hit is None:
            merged.append(set(ring))
        else:
            hit |= ring

    results: list[tuple[str, str]] = []
    for group in merged:
        if len(group) < 1:
            continue
        sub = ring_system_submol(mol, group)
        canonical = Chem.MolToSmiles(sub)
        is_aromatic = any(sub.GetAtomWithIdx(i).GetIsAromatic() for i in range(sub.GetNumAtoms()))
        kind = "aromatic" if is_aromatic else "aliphatic"
        results.append((canonical, kind))
    return results


def canonical_name(kind: str, smiles: str) -> str:
    """Stable, collision-resistant name for a ring system."""
    n = subgraph_n_atoms(smiles)
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:8]
    return f"ring_{kind}_{n}_{digest}"


def subgraph_n_atoms(smarts: str) -> int:
    """Count carbon atoms in a carbon-only ring SMARTS string."""
    try:
        mol = Chem.MolFromSmarts(smarts)
        return mol.GetNumAtoms() if mol is not None else 0
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.records)
    mols: dict[str, Chem.Mol] = {}
    parse_errors: list[str] = []
    for rec in records.itertuples(index=False):
        identifier = str(getattr(rec, "identifier"))
        smiles = str(getattr(rec, "SMILES"))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            parse_errors.append(identifier)
            continue
        mols[identifier] = mol

    # Collect candidate ring systems automatically.
    ring_candidates: dict[str, str] = {}
    for mol in mols.values():
        for smarts, kind in enumerate_ring_systems(mol):
            ring_candidates.setdefault(smarts, kind)

    # Merge manual templates with auto ring systems.
    candidates: dict[str, dict] = {}
    for name, smarts in MANUAL_SUBSTRUCTURES.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        candidates[smarts] = {"name": name, "type": "manual"}

    for smarts, kind in ring_candidates.items():
        candidates[smarts] = {
            "name": canonical_name(kind, smarts),
            "type": f"ring_{kind}",
        }

    # Match every candidate against the library and build the 0/1 matrix.
    identifiers = sorted(mols)
    n_total = len(identifiers)
    bit_names: list[str] = []
    bit_smarts: list[str] = []
    bit_meta: list[dict] = []
    columns: dict[str, list[int]] = {}

    for smarts, meta in candidates.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        col: list[int] = []
        n_pos = 0
        for identifier in identifiers:
            hit = int(mols[identifier].HasSubstructMatch(pattern))
            col.append(hit)
            n_pos += hit
        prevalence = n_pos / n_total if n_total else 0.0
        if prevalence < args.min_prevalence or prevalence > args.max_prevalence:
            continue
        bit_names.append(meta["name"])
        bit_smarts.append(smarts)
        bit_meta.append(
            {
                "name": meta["name"],
                "smarts": smarts,
                "type": meta["type"],
                "prevalence": prevalence,
                "n_positive": n_pos,
            }
        )
        columns[meta["name"]] = col

    fp = pd.concat(
        [pd.DataFrame({"identifier": identifiers}), pd.DataFrame(columns)],
        axis=1,
    )
    fp.to_csv(out_dir / "hydrocarbon_substructure_fingerprints.csv", index=False)

    schema = {
        "n_molecules": int(len(identifiers)),
        "n_candidate_substructures": int(len(candidates)),
        "n_kept_bits": int(len(bit_names)),
        "min_prevalence": args.min_prevalence,
        "max_prevalence": args.max_prevalence,
        "n_parse_errors": int(len(parse_errors)),
        "bits": bit_meta,
    }
    (out_dir / "hydrocarbon_substructure_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in schema.items() if k != "bits"},
        ensure_ascii=False,
        indent=2,
    ))
    if bit_meta:
        prev = [b["prevalence"] for b in bit_meta]
        print("kept prevalence min/median/max:",
              round(min(prev), 4), round(pd.Series(prev).median(), 4), round(max(prev), 4))


if __name__ == "__main__":
    main()
