#!/usr/bin/env python
"""Marrero-Gani (2001) group-contribution properties for hydrocarbons."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import pandas as pd
from rdkit import Chem


COEFF = json.load(open(str(BASE / "data/marrero_gani_coeff.json"), encoding="utf-8"))


def count_groups(mol: Chem.Mol) -> dict[int, int]:
    counts: dict[int, int] = {}
    ri = mol.GetRingInfo()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6:
            continue
        idx = a.GetIdx()
        in_ring = ri.NumAtomRings(idx) > 0
        aromatic = a.GetIsAromatic()
        hs = a.GetTotalNumHs()
        n_double = sum(1 for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        n_triple = sum(1 for b in a.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
        if aromatic:
            if hs == 1:
                g = 14
            else:
                n_ring = ri.NumAtomRings(idx)
                g = 15 if n_ring >= 3 else (16 if n_ring == 2 else 17)
        elif in_ring:
            if n_double:
                g = 170 if hs == 1 else 172
            elif hs == 2:
                g = 167
            elif hs == 1:
                g = 168
            else:
                g = 169
        elif n_triple:
            g = 12 if hs == 1 else 13
        elif n_double:
            g = 4 if hs == 2 else (5 if hs == 1 else 8)
        else:
            g = 0 if hs == 3 else (1 if hs == 2 else (2 if hs == 1 else 3))
        counts[g] = counts.get(g, 0) + 1
    counts.update(_aromatic_ring_groups(mol))
    counts.update(_branching_groups(mol))
    return counts


def _aromatic_ring_groups(mol: Chem.Mol) -> dict[int, int]:
    out: dict[int, int] = {}
    ri = mol.GetRingInfo()
    aromatic_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic()}
    seen: set[int] = set()
    systems: list[list[int]] = []
    for start in aromatic_atoms:
        if start in seen:
            continue
        comp: list[int] = []
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                if nb.GetIdx() in aromatic_atoms:
                    stack.append(nb.GetIdx())
        systems.append(comp)
    for sys_atoms in systems:
        ring_count = len({tuple(sorted(r)) for r in ri.AtomRings() if set(r).issubset(set(sys_atoms)) and len(r) >= 5})
        if ring_count >= 2:
            if ring_count == 2:
                out[354] = out.get(354, 0) + 1
            elif ring_count == 3:
                out[361] = out.get(361, 0) + 1
            elif ring_count == 4:
                max_memb = max(ri.NumAtomRings(a) for a in sys_atoms)
                g = 362 if max_memb <= 2 else 365
                out[g] = out.get(g, 0) + 1
        elif ring_count == 1:
            subs = [a for a in sys_atoms if mol.GetAtomWithIdx(a).GetTotalNumHs() == 0]
            if len(subs) == 2:
                d = _ring_distance(mol, subs[0], subs[1], sys_atoms)
                g = 285 if d == 1 else (286 if d == 2 else 287)
                out[g] = out.get(g, 0) + 1
            elif len(subs) == 3:
                out[288] = out.get(288, 0) + 1
    return out


def _ring_distance(mol: Chem.Mol, a: int, b: int, atoms: list[int]) -> int:
    s = set(atoms)
    q = deque([(a, 0)])
    seen = {a}
    while q:
        u, d = q.popleft()
        if u == b:
            return d
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            if nb.GetIdx() in s and nb.GetIdx() not in seen:
                seen.add(nb.GetIdx())
                q.append((nb.GetIdx(), d + 1))
    return 99


def _branching_groups(mol: Chem.Mol) -> dict[int, int]:
    out: dict[int, int] = {}
    pats = {
        182: "[CH1]([CH3])([CH3])[#6]",
        183: "[CH0]([CH3])([CH3])([CH3])[#6]",
    }
    for g, smarts in pats.items():
        p = Chem.MolFromSmarts(smarts)
        if p is not None:
            out[g] = len(mol.GetSubstructMatches(p))
    return out


def marrero_gani(smiles: str) -> dict[str, float | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    counts = count_groups(mol)
    tb = tc = pc = 0.0
    vc = 7.95
    gf = -34.967
    hf = 5.549
    for g, n in counts.items():
        tb += n * COEFF["tb"][g]
        tc += n * COEFF["tc"][g]
        pc += n * COEFF["pc"][g]
        vc += n * COEFF["vc"][g]
        gf += n * COEFF["gf"][g]
        hf += n * COEFF["hf"][g]
    mw = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6) * 12.011 + sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1) * 1.008
    return {
        "Tb": 222.543 * math.log(tb) if tb > 0 else None,
        "Tc": 231.239 * math.log(tc) if tc > 0 else None,
        "Pc": (5.9827 + (pc + 0.108998) ** -2) * 1e5 if (pc + 0.108998) > 0 else None,
        "Vc": (vc / mw) * 1e-3 if mw > 0 else None,
        "Hfg": (hf / mw) * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--out", default=str(BASE / "data/modeling/marrero_gani_properties.csv"))
    args = parser.parse_args()
    rec = pd.read_csv(args.records)
    rows = []
    for r in rec.itertuples(index=False):
        p = marrero_gani(str(getattr(r, "SMILES")))
        rows.append({
            "identifier": getattr(r, "identifier"),
            "mg_Tb_K": p.get("Tb"),
            "mg_Tc_K": p.get("Tc"),
            "mg_Pc_Pa": p.get("Pc"),
            "mg_Vc_m3_mol": p.get("Vc"),
            "mg_Hfg_J_mol": p.get("Hfg"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
