"""Evaluate the 100 virtual complex mixtures with both pipelines.

   individual-only  ``pure_z_model_split.joblib`` (4,180 training compounds)
   augmented        ``augmented_z_model_split.joblib`` (the same compounds plus
                    30,000 co-eluting spectra whose components are also drawn
                    from those 4,180)

The property models are refitted for each pipeline on the same split
(boiling point by the mass-weighted-mean rule, density by the reciprocal rule).

Outputs (models/simulated_mixture_validation/virtual_mixture/)
  eval_by_group.csv   per (group, pipeline): motif composition, the whole
                      distillation curve (T5..T95, K, true and predicted),
                      T10/T50/T90 (degC), density, C/H mass fractions, H/C,
                      number-average MW / C / H / DBE / ring count
  eval_bits.csv       per (bit, pipeline): R2 (continuous) / AUC + MAE (binary)
  eval_motifs.csv     per (group, pipeline, motif): true / predicted mass %
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data                          # noqa: E402
from z_fingerprint_model import sum_normalize                           # noqa: E402
from mixture_augmented_models import fit_mixture_augmented              # noqa: E402

MD = BASE / "models" / "simulated_mixture_validation"
VM = MD / "virtual_mixture"
# the whole distillation range: P0 (first drop) -> P100 (final boiling point),
# so the curve covers 0-100 % distilled like the measured curve of the real
# sample (which is sampled at 0.5-99.5 %)
PERCENTILES = [0] + list(range(5, 96, 5)) + [100]
SPLIT_SEED = 2026


def fit_plain(X, y, n_trees=120):
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=n_trees, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(X, y)
    return m


def curve(vals, weights):
    o = np.argsort(vals)
    return np.interp(np.array(PERCENTILES, float) / 100.0,
                     np.cumsum(weights[o]), vals[o])


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    flags = [c for c in zcols if c.startswith("desc_")
             and set(np.unique(data[c])) <= {0.0, 1.0}]
    bits = [c for c in zcols if not c.startswith("desc_")]
    cont = [c for c in zcols if c.startswith("desc_") and c not in flags]
    jb = [zcols.index(c) for c in bits]
    fb = [zcols.index(c) for c in flags]
    cb = [zcols.index(c) for c in cont]

    tr, te = train_test_split(np.arange(len(data)), test_size=0.2,
                              random_state=SPLIT_SEED)
    allowed = np.zeros(len(data), bool)
    allowed[tr] = True
    Z = data[zcols].to_numpy(float)
    vals_tb = data["chem_Tb_K"].to_numpy(float)
    vals_rho = data["density_label_g_ml"].to_numpy(float)

    print("fitting property models ...")
    # two ways of getting the size of a slice: read it off the translated
    # fingerprint (12.011 C + 1.008 H) or predict it with a property model.
    # Both are reported: the first is a fingerprint consistency check, the second
    # is the quantity the pipeline actually predicts as a property.
    mw_label = 12.011 * data["desc_C"].to_numpy(float) + 1.008 * data["desc_H"].to_numpy(float)

    def fit_plain(X, y):
        m = Pipeline([("i", SimpleImputer()),
                      ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                                min_samples_leaf=2, n_jobs=1))])
        m.fit(X, y)
        return m

    mw_model = fit_plain(Z[allowed], mw_label[allowed])
    ROUTES = {
        "individual compounds only": {
            "translator": joblib.load(MD / "pure_z_model_split.joblib"),
            "Tb": fit_plain(Z[allowed], vals_tb[allowed]),
            "density": fit_plain(Z[allowed], vals_rho[allowed])},
        "+ co-eluting augmentation": {
            "translator": joblib.load(MD / "augmented_z_model_split.joblib"),
            "Tb": fit_mixture_augmented(zcols, vals_tb, rule="mean",
                                        allowed_pool=allowed),
            "density": fit_mixture_augmented(zcols, vals_rho, rule="reciprocal",
                                             allowed_pool=allowed)}}

    d = np.load(VM / "slices.npz")
    X, area, group, Ztrue = d["X"], d["area"], d["group"], d["Ztrue"]
    summ = pd.read_csv(VM / "summary.csv").set_index("group")
    sel = pd.read_csv(VM / "selections.csv")
    true_bits = np.zeros((len(summ), len(bits)))
    lib_bits = data.set_index("identifier")[bits]
    sel = sel.join(lib_bits, on="identifier")
    for gi, g in enumerate(summ.index):
        sub = sel[sel.group == g]
        true_bits[gi] = (sub[bits].to_numpy(float)
                         * sub["mass_fraction"].to_numpy(float)[:, None]).sum(0)
    print(f"{len(X):,} slices over {len(summ)} mixtures")

    # ---- ground truth of every aggregated quantity -------------------------
    # Same definitions as the construction script: mass-weighted elemental
    # composition, number-average C / H / DBE / ring count, and specific-volume
    # mixing for the density -- all from the library labels of the molecules
    # that were selected into the group.
    lib_num = data.set_index("identifier")[
        ["desc_C", "desc_H", "desc_DBE", "desc_ring_count",
         "desc_aromatic_ring_count_estimate", "density_label_g_ml"]]
    selx = sel.join(lib_num, on="identifier")
    # the exact mixture-level value of every one of the 47 fingerprint bits:
    # the mass-weighted mean of the library values of the selected molecules
    lib_z = data.set_index("identifier")[zcols]
    selz = sel.join(lib_z.drop(columns=[c for c in lib_z.columns if c in sel.columns]),
                    on="identifier")
    true_all = {}
    for g in summ.index:
        sub = selz[selz.group == g]
        w = sub["mass_fraction"].to_numpy(float)
        true_all[g] = (sub[zcols].to_numpy(float) * (w / w.sum())[:, None]).sum(0)
    truth = {}
    for g in summ.index:
        sub = selx[selx.group == g]
        w = sub["mass_fraction"].to_numpy(float)
        C = sub["desc_C"].to_numpy(float)
        H = sub["desc_H"].to_numpy(float)
        mw = 12.011 * C + 1.008 * H
        n = w / mw
        x = n / n.sum()
        cw = float((w * 12.011 * C / mw).sum())
        hw = float((w * 1.008 * H / mw).sum())
        truth[g] = {
            "C_wt_true": cw * 100, "H_wt_true": hw * 100,
            "HC_true": (hw / 1.008) / (cw / 12.011),
            "MW_true": float(1.0 / n.sum()),
            "C_number_true": float((x * C).sum()),
            "H_number_true": float((x * H).sum()),
            "DBE_true": float((x * sub["desc_DBE"].to_numpy(float)).sum()),
            "ring_count_true": float((x * sub["desc_ring_count"].to_numpy(float)).sum()),
            # naphthenic = all rings minus the aromatic ones
            "aromatic_ring_count_true": float(
                (x * sub["desc_aromatic_ring_count_estimate"].to_numpy(float)).sum()),
            "naphthene_ring_count_true": float(
                (x * (sub["desc_ring_count"].to_numpy(float)
                      - sub["desc_aromatic_ring_count_estimate"].to_numpy(float))).sum()),
            "density_true": float(1.0 / np.sum(
                w / sub["density_label_g_ml"].to_numpy(float))),
        }

    FAM_COLS = {"paraffin": "mass_paraffin", "naphthene": "mass_naphthene",
                "aromatic": "mass_aromatic", "olefinic": "mass_olefinic"}
    # the predicted family of a slice is read off the **ring-system bits**, i.e.
    # with exactly the definition used for the truth (aromatic ring / aliphatic
    # ring / no ring); using the coarse ``desc_is_aromatic`` flag instead would
    # over-count aromatics, because that flag also fires for highly unsaturated
    # ring systems.
    ARO_B = [zcols.index(c) for c in bits if "ring_aromatic" in c]
    ALI_B = [zcols.index(c) for c in bits if "ring_aliphatic" in c]
    rows, bit_rows, motif_rows, all_rows = [], [], [], []
    tb_by_id = data.set_index("identifier")["chem_Tb_K"]
    truth_curve = {}
    for g in summ.index:
        sub = sel[sel.group == g]
        v = tb_by_id.loc[sub["identifier"].to_numpy()].to_numpy(float)
        truth_curve[g] = curve(v, sub["mass_fraction"].to_numpy(float))
    for tag, mm in ROUTES.items():
        P = mm["translator"].predict(X)
        Pb = np.clip(P[:, jb], 0, 1)
        # structure metrics over all slices
        for k, c in enumerate(cont):
            y, p = Ztrue[:, cb[k]], P[:, cb[k]]
            bit_rows.append({"bit": c, "kind": "continuous", "pipeline": tag,
                             "r2": float(r2_score(y, p)),
                             "auc": np.nan,
                             "mae": float(np.abs(y - p).mean())})
        for k, c in enumerate(flags + bits):
            j = (fb + jb)[k]
            y, p = Ztrue[:, j], np.clip(P[:, j], 0, 1)
            lab = (y > 0.5).astype(int)
            bit_rows.append({"bit": c, "kind": "flag" if k < len(fb) else "motif",
                             "pipeline": tag, "r2": np.nan,
                             "auc": (float(roc_auc_score(lab, p))
                                     if len(np.unique(lab)) == 2 else np.nan),
                             "mae": float(np.abs(y - p).mean())})
        # per-group outputs
        tb = mm["Tb"].predict(P)
        rho = mm["density"].predict(P)
        for gi, g in enumerate(summ.index):
            m = group == g
            aw = area[m]
            aw = aw / aw.sum()
            pred_bits = (Pb[m] * aw[:, None]).sum(0)
            for k, c in enumerate(bits):
                motif_rows.append({"group": g, "pipeline": tag, "motif": c,
                                   "true_mass_pct": float(true_bits[gi][k] * 100),
                                   "pred_mass_pct": float(pred_bits[k] * 100)})
            # per-slice family decision from the ring-system bits + DBE
            Pm = np.clip(P[m], 0, 1)
            pred_all = (P[m] * aw[:, None]).sum(0)
            for k, c in enumerate(zcols):
                all_rows.append({"group": g, "pipeline": tag, "bit": c,
                                 "true": float(true_all[g][k]),
                                 "pred": float(pred_all[k])})
            has_aro = (Pm[:, ARO_B] > 0.5).any(1)
            has_ali = (Pm[:, ALI_B] > 0.5).any(1)
            dbe_p = P[m][:, zcols.index("desc_DBE")]
            is_aro = has_aro
            is_naph = (~has_aro) & has_ali
            is_ole = (~has_aro) & (~has_ali) & (dbe_p > 0.5)
            is_par = (~has_aro) & (~has_ali) & (dbe_p <= 0.5)
            ch = float((is_par * aw).sum())
            ar = float((is_aro * aw).sum())
            naph = float((is_naph * aw).sum())
            ole = float((is_ole * aw).sum())
            tot = ch + ar + naph + ole
            if tot > 0:
                ch, ar, naph, ole = ch / tot, ar / tot, naph / tot, ole / tot
            C = P[m][:, zcols.index("desc_C")]
            H = P[m][:, zcols.index("desc_H")]
            mw = 12.011 * C + 1.008 * H
            n_mol = aw / np.clip(mw, 1e-9, None)
            n_tot = n_mol.sum()
            tc = curve(tb[m], aw)
            row = {
                "group": g, "pipeline": tag,
                "chain_pred": ch, "aromatic_pred": ar, "olefinic_pred": ole,
                "naphthene_pred": naph,
                "motif_MAE_pp": float(np.abs(pred_bits - true_bits[gi]).mean() * 100),
                "motif_R2": float(r2_score(true_bits[gi], pred_bits)),
                "C_wt": float((aw * 12.011 * C / mw).sum() * 100),
                "H_wt": float((aw * 1.008 * H / mw).sum() * 100),
                # atomic H/C of the whole mixture, from the two aggregated mass
                # fractions (never the mean of the per-slice ratios)
                "HC": float(((aw * 1.008 * H / mw).sum() / 1.008)
                            / np.clip((aw * 12.011 * C / mw).sum() / 12.011,
                                      1e-9, None)),
                "MW": float(1.0 / n_mol.sum()),
                "MW_prop": float(1.0 / np.sum(
                    aw / np.clip(mw_model.predict(P[m]), 1e-6, None))),
                "C_number": float((n_mol * C).sum() / n_tot),
                "H_number": float((n_mol * H).sum() / n_tot),
                "DBE": float((n_mol * P[m][:, zcols.index("desc_DBE")]).sum() / n_tot),
                "ring_count": float((n_mol * P[m][:, zcols.index("desc_ring_count")]).sum()
                                    / n_tot),
                # rings split into aromatic and naphthenic (the latter is the
                # remainder, so the two add up to the total ring count)
                "aromatic_ring_count": float(
                    (n_mol * P[m][:, zcols.index("desc_aromatic_ring_count_estimate")]).sum()
                    / n_tot),
                "naphthene_ring_count": float(
                    (n_mol * (P[m][:, zcols.index("desc_ring_count")]
                              - P[m][:, zcols.index("desc_aromatic_ring_count_estimate")])).sum()
                    / n_tot),
                "density": float(1.0 / np.sum(aw / np.clip(rho[m], 1e-6, None))),
                "T10": float(tc[PERCENTILES.index(10)] - 273.15),
                "T50": float(tc[PERCENTILES.index(50)] - 273.15),
                "T90": float(tc[PERCENTILES.index(90)] - 273.15),
                "Tb_curve_MAE_K": float(np.mean(np.abs(tc - truth_curve[g])))}
            # the whole distillation curve, in K, with its ground truth, so the
            # mixture figures no longer need a separate table
            for k, p in enumerate(PERCENTILES):
                row[f"T{p}_K"] = float(tc[k])
                row[f"T{p}_true_K"] = float(truth_curve[g][k])
            # the three reported cut points also in degC, next to their truth,
            # so no figure has to mix the two units by hand
            for p in (10, 50, 90):
                row[f"T{p}_true"] = float(truth_curve[g][PERCENTILES.index(p)] - 273.15)
            rows.append(row)
        print(f"  {tag} done")
    ev = pd.DataFrame(rows)
    for k, v in FAM_COLS.items():
        ev[f"{k}_true"] = ev["group"].map(summ[v])
    ev["chain_true"] = ev["paraffin_true"]
    ev["naphthene_true"] = ev["naphthene_true"]
    ev["aromatic_true"] = ev["aromatic_true"]
    for k in truth[summ.index[0]]:
        ev[k] = ev["group"].map(lambda g, k=k: truth[g][k])
    ev.to_csv(VM / "eval_by_group.csv", index=False)
    pd.DataFrame(bit_rows).to_csv(VM / "eval_bits.csv", index=False)
    pd.DataFrame(motif_rows).to_csv(VM / "eval_motifs.csv", index=False)
    pd.DataFrame(all_rows).to_csv(VM / "eval_allbits.csv", index=False)

    print()
    for tag in ROUTES:
        s = ev[ev.pipeline == tag]
        b = pd.DataFrame(bit_rows)
        b = b[b.pipeline == tag]
        print(f"  {tag}")
        print(f"    motif composition MAE {s.motif_MAE_pp.mean():.2f} pp  "
              f"R2 {s.motif_R2.mean():.3f}")
        print(f"    14 continuous R2 {b[b.kind == 'continuous'].r2.mean():.4f}   "
              f"32 binary AUC {b[b.kind.isin(['motif', 'flag'])].auc.mean():.4f}")
        for col, unit in (("T50", "C"), ("density", "g/mL"), ("MW", "g/mol"),
                          ("C_wt", "wt%")):
            if col == "C_wt":
                continue
            t = summ[col].to_numpy(float)
            p = s.set_index("group").loc[summ.index, col].to_numpy(float)
            print(f"    {col:8s} MAE {np.abs(p - t).mean():9.4g}   "
                  f"bias {(p - t).mean():+9.4g}")
        print(f"    distillation curve MAE {s.Tb_curve_MAE_K.mean():.2f} K")
    print(f"\n  saved {VM / 'eval_by_group.csv'} and eval_bits.csv")


if __name__ == "__main__":
    main()
