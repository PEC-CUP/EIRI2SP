"""Canonical definitions and **frozen order** of the 47 fingerprint bits Z.

Single source of truth for figures, evaluation plots and the Excel export of
Table S1/S2, so that the names *and the order* used in plots always match the
order of the bits in Z itself.

``ORDER`` is the bit order of the trained Z vector: the 19 descriptor columns
(in the order of ``data/modeling/modeling_schema.json``,
``descriptor_columns`` minus the dropped ones) followed by the 27 substructure
bits (in the column order of
``data/fingerprints/hydrocarbon_substructure_fingerprints.csv``: the 2
atom-environment bits and then the 25 enumerated ring bits, alkene/alkyne
columns removed).

That equals ``predict_mixture.load_training_data()[2]`` (``zcols``) and is the
order drawn in ``figures/16_Z_bit_schematic.svg``.  Every per-bit figure
(12 / 15 / 32 / 47) must iterate ``ORDER`` / ``DESCRIPTORS`` / ``BITS`` rather
than re-sorting the rows, otherwise the figures no longer line up with the
definition tables.  Ordering that is explicitly a *ranking* (e.g. "sorted by
library prevalence" in figure 26/34/41, or "sorted by score" in figure 12) is
fine as long as it is stated; what must not drift is the definition order.

  BLOCK   "descriptor"  -> the 19 descriptor bits (14 continuous + 5 flags)
          "ring"        -> the 25 enumerated ring-system bits
          "atom"        -> the 2 added atom-environment bits
"""

BLOCK: dict[str, str] = {}
SHORT: dict[str, str] = {}
LONG: dict[str, str] = {}


def _add(code: str, block: str, short: str, long: str) -> None:
    BLOCK[code] = block
    SHORT[code] = short
    LONG[code] = long


# ---------------------------------------------------------------- descriptors
_add("desc_C", "descriptor", "C number", "Carbon number")
_add("desc_H", "descriptor", "H number", "Hydrogen number")
_add("desc_DBE", "descriptor", "DBE", "Double-bond equivalent")
_add("desc_H_C", "descriptor", "H/C", "H/C atomic ratio")
_add("desc_is_chain_alkane", "descriptor", "chain alkane", "Chain-alkane flag")
_add("desc_is_aromatic", "descriptor", "aro / unsat.",
     "Aromatic / highly unsaturated ring flag")
_add("desc_dbe_ge_4", "descriptor", "DBE \u2265 4", "DBE \u2265 4 flag")
_add("desc_carbon_bond_count", "descriptor", "C\u2013C bonds", "C\u2013C bond count")
_add("desc_ring_count", "descriptor", "ring count", "Ring count")
_add("desc_double_bond_count", "descriptor", "C=C bonds", "C=C bond count")
_add("desc_branch_carbon_count", "descriptor", "branch C", "Branch-carbon count")
_add("desc_terminal_carbon_count", "descriptor", "terminal C", "Terminal-carbon count")
_add("desc_mean_carbon_degree", "descriptor", "mean C degree", "Mean carbon degree")
_add("desc_longest_path_length_estimate", "descriptor", "longest path",
     "Longest-path length")
_add("desc_terminal_fraction", "descriptor", "terminal frac",
     "Terminal-carbon fraction")
_add("desc_ring_to_carbon_ratio", "descriptor", "ring / C", "Ring-to-carbon ratio")
_add("desc_is_isoalkane", "descriptor", "isoalkane", "Iso-alkane flag")
_add("desc_has_ring", "descriptor", "has ring", "Ring-containing flag")
_add("desc_aromatic_ring_count_estimate", "descriptor", "aromatic rings",
     "Aromatic-ring count")
_add("desc_unsaturation_per_carbon", "descriptor", "DBE / C", "DBE per carbon")

# -------------------------------------------------------- atom-environment bits
_add("quaternary_carbon", "atom", "quat-C", "Quaternary C (atom environment)")
_add("tertiary_carbon", "atom", "tert-C", "Tertiary CH (atom environment)")

# ------------------------------------------------------------ ring-system bits
# order = the column order of hydrocarbon_substructure_fingerprints.csv
_RING = [
    ("ring_aliphatic_3_49aa76c6", "cyclopropane"),
    ("ring_aliphatic_5_92df7aa0", "cyclopentene"),
    ("ring_aliphatic_4_18004bfa", "cyclobutane"),
    ("ring_aliphatic_5_e5038db0", "cyclopentane"),
    ("ring_aromatic_6_84480431", "benzene"),
    ("ring_aliphatic_6_36a45fa1", "cyclohexene"),
    ("ring_aliphatic_6_9970801b", "cyclohexane"),
    ("ring_aliphatic_7_87580d2a", "norbornane"),
    ("ring_aliphatic_7_44c793b5", "cycloheptane"),
    ("ring_aliphatic_8_9e484da7", "cyclooctene"),
    ("ring_aliphatic_8_dc3e5431", "cyclooctane"),
    ("ring_aromatic_9_afce84c1", "indane"),
    ("ring_aliphatic_9_150adda3", "cyclononene"),
    ("ring_aliphatic_9_f0ab0e88", "hydrindane"),
    ("ring_aromatic_10_01f31866", "naphthalene"),
    ("ring_aromatic_10_c1d2687b", "tetralin"),
    ("ring_aliphatic_10_1fe4ff10", "cyclodecene"),
    ("ring_aliphatic_10_f09f6a8f", "decalin"),
    ("ring_aliphatic_10_9edcd2d2", "cyclodecane"),
    ("ring_aromatic_5_9eb77984", "arene + 5-ring"),
    ("ring_aromatic_14_a1e6ec18", "anthracene"),
    ("ring_aromatic_14_541c53d1", "phenanthrene"),
    ("ring_aromatic_6_8b792288", "arene + 6-ring"),
    ("ring_aliphatic_9_8df094a4", "cyclononane"),
    ("ring_aromatic_18_1622d282", "[18]annulene"),
]
# A few ring-system names carry their own qualifier and read badly with the
# generic "<name> ring system" suffix.
_RING_LONG = {
    "ring_aromatic_5_9eb77984":
        "Arene fused to a 5-membered ring (indane-type fragment)",
    "ring_aromatic_6_8b792288":
        "Arene fused to a 6-membered ring (tetralin-type fragment)",
}
for _code, _name in _RING:
    _add(_code, "ring", _name, _RING_LONG.get(_code, _name + " ring system"))

# ---------------------------------------------------------------- frozen order
ORDER: list[str] = [
    # 19 descriptors
    "desc_C", "desc_H", "desc_DBE", "desc_H_C",
    "desc_is_chain_alkane", "desc_is_aromatic", "desc_dbe_ge_4",
    "desc_carbon_bond_count", "desc_ring_count", "desc_double_bond_count",
    "desc_branch_carbon_count", "desc_terminal_carbon_count",
    "desc_mean_carbon_degree", "desc_longest_path_length_estimate",
    "desc_terminal_fraction", "desc_ring_to_carbon_ratio",
    "desc_is_isoalkane", "desc_has_ring",
    "desc_aromatic_ring_count_estimate", "desc_unsaturation_per_carbon",
    # 27 substructure bits
    "quaternary_carbon", "tertiary_carbon",
    "ring_aliphatic_3_49aa76c6", "ring_aliphatic_5_92df7aa0",
    "ring_aliphatic_4_18004bfa", "ring_aliphatic_5_e5038db0",
    "ring_aromatic_6_84480431", "ring_aliphatic_6_36a45fa1",
    "ring_aliphatic_6_9970801b", "ring_aliphatic_7_87580d2a",
    "ring_aliphatic_7_44c793b5", "ring_aliphatic_8_9e484da7",
    "ring_aliphatic_8_dc3e5431", "ring_aromatic_9_afce84c1",
    "ring_aliphatic_9_150adda3", "ring_aliphatic_9_f0ab0e88",
    "ring_aromatic_10_01f31866", "ring_aromatic_10_c1d2687b",
    "ring_aliphatic_10_1fe4ff10", "ring_aliphatic_10_f09f6a8f",
    "ring_aliphatic_10_9edcd2d2", "ring_aromatic_5_9eb77984",
    "ring_aromatic_14_a1e6ec18", "ring_aromatic_14_541c53d1",
    "ring_aromatic_6_8b792288", "ring_aliphatic_9_8df094a4",
    "ring_aromatic_18_1622d282",
]

DESCRIPTORS: list[str] = [c for c in ORDER if BLOCK[c] == "descriptor"]
RING_BITS: list[str] = [c for c in ORDER if BLOCK[c] == "ring"]
ATOM_BITS: list[str] = [c for c in ORDER if BLOCK[c] == "atom"]
# the 27 substructure bits in the frozen order of Z (the 2 atom-environment
# bits come first, then the 25 ring bits)
BITS: list[str] = [c for c in ORDER if BLOCK[c] != "descriptor"]

assert len(ORDER) == 47 and set(ORDER) == set(BLOCK)
assert len(DESCRIPTORS) == 20 and len(BITS) == 27


def short(code: str) -> str:
    return SHORT.get(code, code)


def long(code: str) -> str:
    return LONG.get(code, code)


def block(code: str) -> str:
    return BLOCK.get(code, "descriptor" if code.startswith("desc_") else "ring")


def z_position(code: str) -> int:
    """1-based position of a bit inside the 47-dim Z vector (frozen order)."""
    return ORDER.index(code) + 1
