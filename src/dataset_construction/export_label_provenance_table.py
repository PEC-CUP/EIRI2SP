"""Table S2: where every property label comes from, and what the completion of
the missing labels costs.

Reads models/qspr_routes/label_provenance_comparison.csv and writes a formatted
workbook (one sheet, text copyable into Word) with, per property:
  * the label source and how many of the 5,225 library molecules carry a value,
  * the number of training molecules each of the two label sets provides,
  * test R2 / MAE for both, on the *same* held-out molecules,
  * the completion cost dR2, and whether the property is a target at all.

Retained targets: the completion is free for Tb/Tm/Tc/Pc/Vc (2.5x the training
labels, same accuracy) but costs 0.10 R2 for density, so density keeps the
uncompleted library labels - that is the one property where the two columns
disagree, and the reason the earlier "completion is harmless" statement had to
be corrected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "models" / "qspr_routes" / "label_provenance_comparison.csv"
OUT = BASE / "data" / "fingerprints" / "Table_S_label_provenance.xlsx"

LABEL = {"chem_Tb_K": "Tb (K)", "chem_Tm_K": "Tm (K)", "chem_Tc_K": "Tc (K)",
         "chem_Pc_Pa": "Pc (Pa)", "chem_Vc_m3_mol": "Vc (m3/mol)",
         "chem_Hfg_J_mol": "dHf (J/mol)", "chem_Hfus_J_mol": "dHfus (J/mol)",
         "PubChem_Complexity": "complexity", "PubChem_MolecularWeight": "MW (g/mol)",
         "PubChem_RotatableBondCount": "rotatable bonds", "PubChem_XLogP": "XLogP",
         "fuel_density_g_ml_298K": "density (g/mL)"}
# Vc is ~1e-4 m3/mol, so four decimals would print as 0.0000; the MAE columns for
# Vc are therefore reported in cm3/mol (1e-6 m3/mol).
MAE_SCALE = {"chem_Vc_m3_mol": 1e6, "chem_Pc_Pa": 1e-6,
             "chem_Hfg_J_mol": 1e-3, "chem_Hfus_J_mol": 1e-3}
MAE_UNIT = {"chem_Vc_m3_mol": "cm3/mol", "chem_Pc_Pa": "MPa",
            "chem_Hfg_J_mol": "kJ/mol", "chem_Hfus_J_mol": "kJ/mol"}
TITLEU = {"chem_Tb_K": "K", "chem_Tm_K": "K", "chem_Tc_K": "K",
          "PubChem_Complexity": "-", "PubChem_MolecularWeight": "g/mol",
          "PubChem_RotatableBondCount": "-",
          # logP is a log10 ratio, so its error unit is "log units", not "-"
          "PubChem_XLogP": "log units",
          "fuel_density_g_ml_298K": "g/mL"}


# retained targets, in the definition order of the paper, then the rejected ones
ORDER = ["chem_Tb_K", "chem_Tm_K", "chem_Tc_K", "chem_Pc_Pa", "chem_Vc_m3_mol",
         "PubChem_MolecularWeight", "PubChem_XLogP", "fuel_density_g_ml_298K",
         "chem_Hfg_J_mol", "chem_Hfus_J_mol", "PubChem_Complexity",
         "PubChem_RotatableBondCount"]
TARGETS = ORDER[:8]
SOURCE = {
    "chem_Tb_K": "measured / database + Joback completion",
    "chem_Tm_K": "measured / database + Joback completion",
    "chem_Tc_K": "measured / database + Joback completion",
    "chem_Pc_Pa": "measured / database + Joback completion",
    "chem_Vc_m3_mol": "measured / database + Joback completion",
    "PubChem_MolecularWeight": "PubChem + exact formula value for the 118 gaps",
    "PubChem_XLogP": "PubChem XLogP3 + calibrated Crippen for the 118 gaps",
    "fuel_density_g_ml_298K": "thermo library value + group-contribution completion of the gaps",
    "chem_Hfg_J_mol": "Joback completion",
    "chem_Hfus_J_mol": "measured only (Marrero-Gani fill not used)",
    "PubChem_Complexity": "PubChem",
    "PubChem_RotatableBondCount": "PubChem",
}
ROLE = {p: "target" for p in TARGETS}
ROLE.update({
    "chem_Hfg_J_mol": "not a target - completion too inaccurate (R2 0.74)",
    "chem_Hfus_J_mol": "not a target - completion too inaccurate",
    "PubChem_Complexity": "not a target - 2D structure descriptor, not a property",
    "PubChem_RotatableBondCount": "not a target - small integer count, % error meaningless",
})
# the two PubChem gaps are completed, so the raw column coverage understates them
COVERAGE = {
    "PubChem_MolecularWeight": "5,109 + 118 exact formula -> 5,225 (100 %)",
    "PubChem_XLogP": "5,109 + 118 calibrated Crippen -> 5,225 (100 %)",
    "fuel_density_g_ml_298K": "2,088 library + 3,139 group-contribution -> 5,225 (100 %)",
}


def main() -> None:
    t = pd.read_csv(SRC).set_index("property").reindex(ORDER)
    lib = pd.read_csv(BASE / "data" / "modeling" / "modeling_table.csv")
    n_lib = len(lib)

    t["library coverage"] = [
        f"{int(lib[p].notna().sum())}/{n_lib} ({100 * lib[p].notna().mean():.1f} %)"
        if p in lib.columns else "-" for p in t.index]
    for p, txt in COVERAGE.items():
        t.loc[p, "library coverage"] = txt
    t["dR2 (all - measured)"] = t["R2 all labels"] - t["R2 measured only"]
    # every retained target now uses its completed label set
    t["used as target"] = [p in TARGETS for p in t.index]
    t["final R2"] = [r["R2 all labels"] for _, r in t.iterrows()]
    t["final train n"] = [r["n train, all labels"] for _, r in t.iterrows()]
    t["note"] = [
        "completion accepted: +0.006 R2 (a Rackett completion was tested too and costs 0.10)"
        if p == "fuel_density_g_ml_298K" else
        ("labels complete - the two models are identical"
         if r["n train, all labels"] == r["n train, measured only"] else
         f"+{r['n train, all labels'] - r['n train, measured only']} completed training "
         f"labels for dR2 = {r['dR2 (all - measured)']:+.4f}")
        for p, r in t.iterrows()]

    out = pd.DataFrame({
        "Property": [LABEL.get(p, p) for p in t.index],
        "Bit (code)": list(t.index),
        "Role": [ROLE[p] for p in t.index],
        "Label source / policy": [SOURCE[p] for p in t.index],
        "Values in the library": t["library coverage"],
        "Train n (label set used)": t["final train n"],
        "Test n": t["n test (real)"],
        "R2 (label set used)": t["final R2"].round(4),
        "R2 with the completion": t["R2 all labels"].round(4),
        "dR2 (completed - measured)": [r["dR2 (all - measured)"] for _, r in t.iterrows()],
        "MAE": [round(r["MAE all labels"] * MAE_SCALE.get(p, 1.0), 4)
                for p, r in t.iterrows()],
        "MAPE %": [round(r["MAPE all labels"], 2) for _, r in t.iterrows()],
        "MAE unit": [MAE_UNIT.get(p, TITLEU.get(p, "-")) for p in t.index],
        "Note": t["note"]})
    mean_used = t.loc[t["used as target"], "final R2"].mean()
    mean_all = t.loc[t["used as target"], "R2 all labels"].mean()
    note = (f"Route A (input = the true 46-position fingerprint Z), random 80/20 split (seed 2026).  "
            f"Both label sets are scored on the same held-out molecules - those carrying a "
            f"database value (test n).  Over the 8 retained targets the completed label set used "
            f"here gives mean R2 {mean_used:.4f}.  Completion is free for Tb/Tm/Tc/Pc/Vc (2.5x the "
            f"training molecules, dR2 -0.025..+0.004), exact for MW, neutral for XLogP and mildly "
            f"positive for density (+0.006), whose 3,139 gaps are filled by a ridge fit of the "
            f"2,088 library values (5-fold CV: R2 0.925, MAE 0.021 g/mL, MAPE 2.4 %).  The first "
            f"density completion, Rackett from the Joback critical properties, was tested the same "
            f"way and rejected: it deviates by 10 % (median) and up to 70 % for heavy molecules and "
            f"costs 0.10 R2.  Density values outside 0.55-1.25 g/mL - the condensed-phase window of "
            f"a hydrocarbon at 298 K - are treated as missing and refilled.  MW is the one target "
            f"whose label is not inferred from the fingerprint at all (PubChem / exact molecular "
            f"formula); because it is mass-additive it can also be computed from Z, and that "
            f"cross-check (models/qspr_routes/mw_consistency.csv, figure si/55) agrees with the "
            f"property model to 0.41 g/mol on average.")

    with pd.ExcelWriter(OUT) as w:
        out.to_excel(w, sheet_name="label provenance", index=False, startrow=2)
        ws = w.sheets["label provenance"]
        ws["A1"] = ("Table S2.  Property labels: source, coverage and the cost of completing "
                    "the missing values")
        ws["A1"].font = Font(bold=True, size=12)
        ws["A2"] = note
        ws["A2"].font = Font(size=9)
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(out.columns))
        ws.row_dimensions[2].height = 58
        thin = Side(style="thin", color="D0D5DD")
        head = PatternFill("solid", fgColor="EDF1F7")
        for j, col in enumerate(out.columns, start=1):
            c = ws.cell(row=3, column=j)
            c.font = Font(bold=True, size=9)
            c.fill = head
            c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            width = max(11, min(30, int(col.__len__() * 1.15) + 4))
            ws.column_dimensions[get_column_letter(j)].width = width
        for i in range(4, 4 + len(out)):
            for j in range(1, len(out.columns) + 1):
                c = ws.cell(row=i, column=j)
                c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                c.font = Font(size=9)
                c.alignment = Alignment(vertical="top",
                                        wrap_text=j in (3, len(out.columns)))
        ws.freeze_panes = "A4"
    out.to_csv(OUT.with_suffix(".csv"), index=False)
    print(out.to_string(index=False))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
