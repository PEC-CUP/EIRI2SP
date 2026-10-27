"""Build the 200 virtual complex mixtures (40 selections x 5 abundance settings).

Uses the single construction in ``select_virtual_mixture`` (2026-10-11 revision:
**every selection has its own sampling bell**):

    * molecules from the **whole library** - a virtual mixture is a new object
      (a chromatogram with overlapping, abundance-weighted components), so all
      5,225 compounds are available to build it;
    * four families with realistic mass fractions (paraffins / naphthenes /
      aromatics / olefinic - the acyclic unsaturated part, a few percent);
    * 40 selections = 20 composition cells x 2 boiling-range halves; selection
      ``j`` draws the mass from **its own bell** over the retention index
      (centre 1150 -> 2600 RI, sigma 280 -> 620 RI), so the suite spans
      different cuts instead of parallel shifts of one curve;
    * 124 molecules per selection, one Gaussian slice train per mixture
      (sigma = 8 RI, 5 RI step, RI 850-3100).

The co-eluting *augmentation* sampler is untouched and still uses one fixed bell
(see ``select_virtual_mixture.draw_local_windows``), so the frozen models stay
valid: the bell only shapes the test bed.

Outputs (models/simulated_mixture_validation/virtual_mixture/)
  selections.csv    one row per (group, molecule): family, mass fraction
  summary.csv       one row per group: composition, average structure, and the
                    mixture properties computed from the components
  slices.npz        X (spectra + RI), area, group id, true per-slice fingerprint
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data                          # noqa: E402
from z_fingerprint_model import sum_normalize                           # noqa: E402
from validate_ri_and_substructure_contribution import build_slices      # noqa: E402
from select_virtual_mixture import (select_virtual_mixture, apply_abundance,
                                    FAMILY_NAMES, family_of, N_SELECTIONS,
                                    bell_of_cell, selection_cell)       # noqa: E402

OUT = BASE / "models" / "simulated_mixture_validation" / "virtual_mixture"
PERCENTILES = list(range(5, 96, 5))
N_SETTINGS = 5
SPLIT_SEED = 2026


def curve(vals, weights):
    o = np.argsort(vals)
    return np.interp(np.array(PERCENTILES, float) / 100.0,
                     np.cumsum(weights[o]), vals[o])


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    print(f"molecule pool: the whole library ({len(data)} compounds)")

    fam_all = family_of(data)
    mw_lib = 12.011 * data["desc_C"].to_numpy(float) + \
        1.008 * data["desc_H"].to_numpy(float)
    tb_lib = data["chem_Tb_K"].to_numpy(float)
    rho_lib = data["density_label_g_ml"].to_numpy(float)

    srows, sel_rows = [], []
    Xs, areas, gids, Ztrues, ris = [], [], [], [], []
    for s in range(N_SELECTIONS):
        base = select_virtual_mixture(data, seed=3000 + s)
        for a in range(1, N_SETTINGS + 1):
            sel = apply_abundance(base, a)
            w = sel["_w"].to_numpy(float)
            fam_sel = sel["_family"].to_numpy(int)
            grp = s * N_SETTINGS + a
            for i, r in enumerate(sel.itertuples()):
                sel_rows.append({"group": grp, "setting": a, "family_index": s,
                                 "identifier": r.identifier,
                                 "family": FAMILY_NAMES[fam_sel[i]],
                                 "mass_fraction": w[i]})
            mass = {n: float(w[fam_sel == k].sum())
                    for k, n in enumerate(FAMILY_NAMES)}
            mws = mw_lib[sel.index.to_numpy()]
            n_mol = w / mws
            x = n_mol / n_mol.sum()
            rho_mix = float(1.0 / np.sum(w / rho_lib[sel.index.to_numpy()]))
            tb_curve = curve(tb_lib[sel.index.to_numpy()], w)
            rec = {"group": grp, "setting": a, "selection": s + 1,
                   "n_compounds": len(sel),
                   "bell_center": float(sel.attrs["design"]["bell_center"]),
                   "bell_sigma": float(sel.attrs["design"]["bell_sigma"]),
                   "bell_class": sel.attrs["design"].get("bell_class", ""),
                   "composition_cell": int(sel.attrs["design"].get("composition_cell", -1)),
                   **{f"mass_{k}": v for k, v in mass.items()},
                   "n_paraffin": int((fam_sel == 0).sum()),
                   "n_naphthene": int((fam_sel == 1).sum()),
                   "n_aromatic": int((fam_sel == 2).sum()),
                   "n_olefinic": int((fam_sel == 3).sum()),
                   "C_min": float(sel["desc_C"].min()),
                   "C_max": float(sel["desc_C"].max()),
                   "RI_min": float(sel["desc_airi_ri"].min()),
                   "RI_max": float(sel["desc_airi_ri"].max()),
                   "C_number": float((x * sel["desc_C"].to_numpy(float)).sum()),
                   "H_number": float((x * sel["desc_H"].to_numpy(float)).sum()),
                   "DBE": float((x * sel["desc_DBE"].to_numpy(float)).sum()),
                   "ring_count": float((x * sel["desc_ring_count"].to_numpy(float)).sum()),
                   "MW": float(1.0 / n_mol.sum()),
                   "density": rho_mix,
                   "T50": float(tb_curve[PERCENTILES.index(50)] - 273.15),
                   "T10": float(tb_curve[PERCENTILES.index(10)] - 273.15),
                   "T90": float(tb_curve[PERCENTILES.index(90)] - 273.15)}
            # slices (Gaussian elution profile), reusing the production code
            rng = np.random.default_rng(7000 + 100 * s + a)
            vm, _w = build_slices(sel, data, spec, rng)
            Xv = np.hstack([vm[spec].to_numpy(float),
                            vm["ri"].to_numpy(float)[:, None]])
            aw = vm["area"].to_numpy(float)
            aw = aw / aw.sum()
            ri_v = vm["ri"].to_numpy(float)
            ri_sel = sel["desc_airi_ri"].to_numpy(float)
            Zsel = sel[zcols].to_numpy(float)
            Ztrue = np.zeros((len(vm), len(zcols)))
            for k, r in enumerate(ri_v):
                m = _w * np.exp(-0.5 * ((r - ri_sel) / 8.0) ** 2)
                Ztrue[k] = (m / m.sum()) @ Zsel
            rec["n_slices"] = int(len(vm))
            rec["slice_C_number"] = float((Ztrue[:, zcols.index("desc_C")]
                                           * aw).sum())
            srows.append(rec)
            Xs.append(Xv.astype(np.float32))
            areas.append(aw.astype(np.float32))
            gids.append(np.full(len(vm), grp, np.int32))
            Ztrues.append(Ztrue.astype(np.float32))
            ris.append(ri_v.astype(np.float32))
    summary = pd.DataFrame(srows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / "selections.csv", index=False)
    np.savez_compressed(OUT / "slices.npz", X=np.vstack(Xs),
                        area=np.concatenate(areas), group=np.concatenate(gids),
                        Ztrue=np.vstack(Ztrues), ri=np.concatenate(ris),
                        percentiles=np.array(PERCENTILES))

    print(f"\n  {len(summary)} virtual complex mixtures, "
          f"{int(summary.n_slices.sum())} slices")
    for k in FAMILY_NAMES:
        v = summary[f"mass_{k}"]
        print(f"    {k:9s} mass fraction {v.min():.3f}-{v.max():.3f} "
              f"(mean {v.mean():.3f})")
    for k, u in (("C_number", ""), ("MW", " g/mol"), ("density", " g/mL"),
                 ("T10", " C"), ("T50", " C"), ("T90", " C")):
        v = summary[k]
        print(f"    {k:9s} {v.min():8.2f}-{v.max():8.2f} (mean {v.mean():8.2f}){u}")
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
