#!/usr/bin/env python
"""EI+RI QSPR route comparison on the manuscript 47-D structure space.

Current Z = 20 pure-structure descriptors + 27 interpretable hydrocarbon
substructure bits. Leaky mass/RI columns, constant descriptors, redundant
fingerprint bits, and non-manuscript property targets are removed before model
comparison.  Routes:
  A: true Z + RI -> property (structure upper bound)
  B: EI + RI -> property (direct spectral QSPR)
  C: EI + RI -> Z (joint multi-output regression) -> property
  D: EI + RI -> Z (descriptor regression + per-bit probabilistic FingerID)
      -> property
  E/F/G: bitwise or shared-Z variants used as route controls.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DROP_DESC = {
    "desc_MW",
    "desc_exact_mass",
    "desc_nominal_mass",
    "desc_carbon_mass_fraction",
    "desc_hydrogen_mass_fraction",
    "desc_airi_ri",
    "desc_airi_ri_stddev",
    "desc_airi_ri_confidence_interval",
    "desc_dbe_ge_1",
    "desc_carbon_skeleton_diameter",
    "desc_component_count",
    "desc_c_5_10",
    "desc_c_11_16",
    "desc_c_17_24",
    "desc_c_25_40",
    "desc_is_normal_alkane",
    "desc_triple_bond_count",
    "desc_max_carbon_degree",
    "desc_is_cycloalkane",
    "desc_quaternary_carbon_count",
    "desc_branching_index",
    "desc_is_alkene",
}

DROP_TARGETS = {
    "fuel_density_kg_m3_298K",
    "fuel_API_gravity_298K",
    "fuel_flash_point_C",
    "fuel_LHV_MJ_kg",
    "fuel_log10_dynamic_viscosity_Pa_s_298K",
    "fuel_log10_kinematic_viscosity_cSt_298K",
    "fuel_flash_point_K",
    "chem_Hfl_J_mol",
    "fuel_HHV_MJ_kg",
    "chem_S0g_J_molK",
    "chem_omega",
    "fuel_Hvap_J_kg",
    "fuel_specific_gravity_298K",
    "fuel_cetane_index_D976_Tb_proxy",
    "chem_MW",
    "PubChem_ExactMass",
    "PubChem_MonoisotopicMass",
    "PubChem_HeavyAtomCount",
    "PubChem_TPSA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(BASE / "data/modeling/modeling_table.csv"))
    parser.add_argument("--schema", default=str(BASE / "data/modeling/modeling_schema.json"))
    parser.add_argument("--records", default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--fingerprints", default=str(BASE / "data/fingerprints/hydrocarbon_substructure_fingerprints.csv"))
    parser.add_argument("--out-dir", default=str(BASE / "models/qspr_routes"))
    parser.add_argument("--split", choices=["random", "formula_group"], default="random")
    parser.add_argument("--property-input", choices=["Z", "Z+RI"], default="Z",
                        help="columns fed to the property model (the C route is "
                             "EI+RI -> Z -> property; 'Z+RI' keeps the legacy "
                             "second RI input, 'Z' uses the fingerprint only)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--fingerid-val-size", type=float, default=0.2)
    parser.add_argument("--min-train-labels", type=int, default=20)
    parser.add_argument("--min-test-labels", type=int, default=5)
    return parser.parse_args()


def make_regressor(seed: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "rmse": float(mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5),
        "r2": float(r2_score(y_true[mask], y_pred[mask])) if len(np.unique(y_true[mask])) > 1 else np.nan,
    }


def split_data(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if args.split == "formula_group":
        records = pd.read_csv(args.records, usecols=["identifier", "FORMULA"])
        grouped = data.merge(records, on="identifier", how="left")
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(grouped, groups=grouped["FORMULA"]))
        return (
            grouped.iloc[train_idx].copy(),
            grouped.iloc[test_idx].copy(),
            "GroupShuffleSplit by molecular formula",
        )
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed)
    return train.copy(), test.copy(), "Random molecule split"


def predict_fingerid_z(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    desc_cols: list[str],
    fp_cols: list[str],
    seed: int,
    n_estimators: int,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptor_model = make_regressor(seed, max(40, n_estimators // 2))
    descriptor_model.fit(train[x_cols], train[desc_cols])
    pred_desc = pd.DataFrame(descriptor_model.predict(test[x_cols]), columns=desc_cols, index=test.index)

    bit_train, bit_val = train_test_split(train, test_size=val_size, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(bit_train[x_cols].to_numpy(dtype=float))
    x_val = scaler.transform(bit_val[x_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test[x_cols].to_numpy(dtype=float))

    prob_bits = np.zeros((len(test), len(fp_cols)), dtype=np.float32)
    bit_rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for j, bit in enumerate(fp_cols):
        y_train = bit_train[bit].to_numpy(dtype=int)
        y_val = bit_val[bit].to_numpy(dtype=int)
        prevalence = float(train[bit].mean())
        if len(np.unique(y_train)) < 2:
            prob = np.full(len(test), prevalence, dtype=float)
            auc = np.nan
            status = "constant_train_bit"
        else:
            clf = SGDClassifier(
                loss="log_loss",
                alpha=0.0005,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                random_state=seed,
            )
            clf.fit(x_train, y_train)
            prob = clf.predict_proba(x_test)[:, 1]
            val_prob = clf.predict_proba(x_val)[:, 1]
            auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) == 2 else np.nan
            status = "trained"
        prob_bits[:, j] = prob
        bit_rows.append(
            {
                "bit": bit,
                "prevalence": prevalence,
                "validation_auc": auc,
                "status": status,
            }
        )
    pred_fp = pd.DataFrame(prob_bits, columns=fp_cols, index=test.index)
    z_pred = pd.concat([pred_desc, pred_fp], axis=1)
    return z_pred, pd.DataFrame(bit_rows)


def predict_fingerid_z_bitwise_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    desc_cols: list[str],
    fp_cols: list[str],
    seed: int,
    n_estimators: int,
    val_size: float,
    model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bitwise regression FingerID: one regressor per fingerprint bit.

    model_type is either "svr" or "et".  Descriptor columns are predicted by a
    single multi-output regressor; fingerprint bits are each predicted by an
    independent regressor producing continuous values.
    """
    descriptor_model = make_regressor(seed, max(40, n_estimators // 2))
    descriptor_model.fit(train[x_cols], train[desc_cols])
    pred_desc = pd.DataFrame(descriptor_model.predict(test[x_cols]), columns=desc_cols, index=test.index)

    bit_train, bit_val = train_test_split(train, test_size=val_size, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(bit_train[x_cols].to_numpy(dtype=float))
    x_val = scaler.transform(bit_val[x_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test[x_cols].to_numpy(dtype=float))

    pred_bits = np.zeros((len(test), len(fp_cols)), dtype=np.float64)
    bit_rows = []
    for j, bit in enumerate(fp_cols):
        y_train = bit_train[bit].to_numpy(dtype=float)
        y_val = bit_val[bit].to_numpy(dtype=float)
        prevalence = float(train[bit].mean())
        if len(np.unique(y_train)) < 2:
            pred = np.full(len(test), prevalence, dtype=float)
            auc = np.nan
            status = "constant_bit"
        else:
            if model_type == "svr":
                model = SVR(C=1.0, epsilon=0.1, gamma="scale")
            else:
                model = ExtraTreesRegressor(
                    n_estimators=max(30, n_estimators // 2),
                    random_state=seed,
                    n_jobs=1,
                    min_samples_leaf=2,
                )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            val_pred = model.predict(x_val)
            auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) == 2 else np.nan
            status = "trained"
        pred_bits[:, j] = pred
        bit_rows.append(
            {
                "bit": bit,
                "prevalence": prevalence,
                "validation_auc": auc,
                "status": status,
                "model_type": model_type,
            }
        )
    pred_fp = pd.DataFrame(pred_bits, columns=fp_cols, index=test.index)
    z_pred = pd.concat([pred_desc, pred_fp], axis=1)
    return z_pred, pd.DataFrame(bit_rows)


def compute_bit_aucs(z_pred: pd.DataFrame, test: pd.DataFrame, fp_cols: list[str]) -> list[float]:
    """Compute per-bit ROC-AUC on the test set using predicted scores."""
    aucs = []
    for bit in fp_cols:
        y_true = test[bit].to_numpy(dtype=int)
        y_score = z_pred[bit].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_true, y_score))
    return aucs


def predict_fingerid_z_shared_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_cols: list[str],
    desc_cols: list[str],
    fp_cols: list[str],
    seed: int,
    n_estimators: int,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared-bottom multi-task MLP for fingerprint bits.

    Descriptors use a multi-output ExtraTrees regressor (same as other routes).
    Fingerprint bits share two hidden layers and each bit gets its own output
    unit, so information is shared across bits while keeping bit-level outputs.
    """
    descriptor_model = make_regressor(seed, max(40, n_estimators // 2))
    descriptor_model.fit(train[x_cols], train[desc_cols])
    pred_desc = pd.DataFrame(descriptor_model.predict(test[x_cols]), columns=desc_cols, index=test.index)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[x_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test[x_cols].to_numpy(dtype=float))
    y_train = train[fp_cols].to_numpy(dtype=float)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=400,
        early_stopping=True,
        validation_fraction=val_size,
        random_state=seed,
    )
    mlp.fit(x_train, y_train)
    pred_fp = pd.DataFrame(mlp.predict(x_test), columns=fp_cols, index=test.index)
    z_pred = pd.concat([pred_desc, pred_fp], axis=1)
    return z_pred, pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.table)
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    fp = pd.read_csv(args.fingerprints)
    data = data.merge(fp, on="identifier", how="inner")
    records = pd.read_csv(args.records, usecols=["identifier", "C"])
    data = data.merge(records, on="identifier", how="inner")
    data = data[data["C"] >= 5].copy()  # 排除 C4 及以下（本数据最小 C=5，故无实际剔除）

    joback = pd.read_csv(str(BASE / "data/modeling/joback_properties.csv"))
    cn = pd.read_csv(str(BASE / "data/modeling/cetane_clean.csv"))
    data = data.merge(joback, on="identifier", how="left")
    data = data.merge(cn[["identifier", "cetane_number_clean", "cetane_number_gc"]], on="identifier", how="left")
    for src, dst in [
        ("joback_Tb_K", "chem_Tb_K"),
        ("joback_Tm_K", "chem_Tm_K"),
        ("joback_Tc_K", "chem_Tc_K"),
        ("joback_Pc_Pa", "chem_Pc_Pa"),
        ("joback_Vc_m3_mol", "chem_Vc_m3_mol"),
        ("joback_Hfg_J_mol", "chem_Hfg_J_mol"),
    ]:
        if dst in data.columns and src in data.columns:
            data[dst] = data[dst].fillna(data[src])

    spectrum_cols = [
        c for c in schema["spectrum_columns"]
        if c in data.columns and 35 <= int(c.split("_")[1]) <= 550
    ]
    desc_cols = [c for c in schema["descriptor_columns"] if c in data.columns and c not in DROP_DESC]
    fp_cols = [c for c in fp.columns if c != "identifier" and c not in {"alkyne_any", "alkene_any", "alkene_terminal"}]
    ri_cols = [c for c in ["desc_airi_ri"] if c in data.columns]
    z_cols = [*desc_cols, *fp_cols]
    prop_input_cols = [*z_cols, *ri_cols] if args.property_input == "Z+RI" else list(z_cols)
    a_label = "A_true_Z_plus_RI" if args.property_input == "Z+RI" else "A_true_Z"
    x_cols = [*spectrum_cols, *ri_cols]
    property_cols = [c for c in schema["property_columns"] if c not in DROP_TARGETS]

    valid = data.dropna(subset=property_cols, how="all").copy()
    train, test, split_note = split_data(valid, args)
    usable_props = [
        c
        for c in property_cols
        if train[c].notna().sum() >= args.min_train_labels and test[c].notna().sum() >= args.min_test_labels
    ]

    multiout_model = make_regressor(args.seed, max(40, args.n_estimators // 2))
    multiout_model.fit(train[x_cols], train[z_cols])
    z_multiout = pd.DataFrame(multiout_model.predict(test[x_cols]), columns=z_cols, index=test.index)
    z_multiout_train = pd.DataFrame(multiout_model.predict(train[x_cols]), columns=z_cols, index=train.index)
    z_fingerid, bit_metrics = predict_fingerid_z(
        train=train,
        test=test,
        x_cols=x_cols,
        desc_cols=desc_cols,
        fp_cols=fp_cols,
        seed=args.seed,
        n_estimators=args.n_estimators,
        val_size=args.fingerid_val_size,
    )
    z_fingerid = z_fingerid[z_cols]

    z_svr, bit_metrics_svr = predict_fingerid_z_bitwise_regression(
        train=train,
        test=test,
        x_cols=x_cols,
        desc_cols=desc_cols,
        fp_cols=fp_cols,
        seed=args.seed,
        n_estimators=args.n_estimators,
        val_size=args.fingerid_val_size,
        model_type="svr",
    )
    z_svr = z_svr[z_cols]

    z_et, bit_metrics_et = predict_fingerid_z_bitwise_regression(
        train=train,
        test=test,
        x_cols=x_cols,
        desc_cols=desc_cols,
        fp_cols=fp_cols,
        seed=args.seed,
        n_estimators=args.n_estimators,
        val_size=args.fingerid_val_size,
        model_type="et",
    )
    z_et = z_et[z_cols]

    z_mlp, _ = predict_fingerid_z_shared_mlp(
        train=train,
        test=test,
        x_cols=x_cols,
        desc_cols=desc_cols,
        fp_cols=fp_cols,
        seed=args.seed,
        n_estimators=args.n_estimators,
        val_size=args.fingerid_val_size,
    )
    z_mlp = z_mlp[z_cols]

    metrics = []
    predictions = []
    for prop in usable_props:
        train_mask = train[prop].notna()
        test_mask = test[prop].notna()
        train_prop = train.loc[train_mask]
        test_prop = test.loc[test_mask]
        y = test_prop[prop].to_numpy(dtype=float)

        true_z_model = make_regressor(args.seed, args.n_estimators)
        true_z_model.fit(train_prop[prop_input_cols], train_prop[prop])
        y_true_z = true_z_model.predict(test_prop[prop_input_cols])

        c_input = z_multiout.loc[test_prop.index, z_cols].copy()
        c_input[ri_cols[0]] = test_prop[ri_cols[0]].to_numpy()
        y_multiout = true_z_model.predict(c_input[prop_input_cols])

        d_input = z_fingerid.loc[test_prop.index, z_cols].copy()
        d_input[ri_cols[0]] = test_prop[ri_cols[0]].to_numpy()
        y_fingerid = true_z_model.predict(d_input[prop_input_cols])

        svr_input = z_svr.loc[test_prop.index, z_cols].copy()
        svr_input[ri_cols[0]] = test_prop[ri_cols[0]].to_numpy()
        y_svr = true_z_model.predict(svr_input[prop_input_cols])

        et_input = z_et.loc[test_prop.index, z_cols].copy()
        et_input[ri_cols[0]] = test_prop[ri_cols[0]].to_numpy()
        y_et = true_z_model.predict(et_input[prop_input_cols])

        mlp_input = z_mlp.loc[test_prop.index, z_cols].copy()
        mlp_input[ri_cols[0]] = test_prop[ri_cols[0]].to_numpy()
        y_mlp = true_z_model.predict(mlp_input[prop_input_cols])

        direct_model = make_regressor(args.seed, args.n_estimators)
        direct_model.fit(train_prop[x_cols], train_prop[prop])
        y_direct = direct_model.predict(test_prop[x_cols])

        g_train = pd.concat(
            [train_prop[x_cols].reset_index(drop=True),
             z_multiout_train.loc[train_prop.index, z_cols].reset_index(drop=True)],
            axis=1,
        )
        g_test = pd.concat(
            [test_prop[x_cols].reset_index(drop=True),
             z_multiout.loc[test_prop.index, z_cols].reset_index(drop=True)],
            axis=1,
        )
        g_model = make_regressor(args.seed, args.n_estimators)
        g_model.fit(g_train, train_prop[prop])
        y_g = g_model.predict(g_test)

        for route, pred in [
            (a_label, y_true_z),
            ("B_direct_EI_RI", y_direct),
            ("C_EI_RI_to_Z_multioutput", y_multiout),
            ("D_EI_RI_to_Z_fingerid", y_fingerid),
            ("E_SVR_EI_RI_to_Z_bitwise", y_svr),
            ("E_ET_EI_RI_to_Z_bitwise", y_et),
            ("F_MLP_EI_RI_to_Z_shared", y_mlp),
            ("G_EI_RI_plus_predZ", y_g),
        ]:
            row = {"route": route, "target": prop}
            row.update(score(y, pred))
            metrics.append(row)

        predictions.append(
            pd.DataFrame(
                {
                    "identifier": test_prop["identifier"].to_numpy(),
                    "target": prop,
                    "true": y,
                    "A_true_Z_plus_RI": y_true_z,
                    "B_direct": y_direct,
                    "C_multioutput": y_multiout,
                    "D_fingerid": y_fingerid,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_df.to_csv(out_dir / "route_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    bit_metrics.to_csv(out_dir / "fingerid_bit_metrics.csv", index=False)
    joblib.dump(multiout_model, out_dir / "ei_ri_to_Z_multioutput.joblib")

    auc_routes = {
        "C_multioutput": compute_bit_aucs(z_multiout[fp_cols], test, fp_cols),
        "D_fingerid_prob": compute_bit_aucs(z_fingerid[fp_cols], test, fp_cols),
        "E_SVR_bitwise": compute_bit_aucs(z_svr[fp_cols], test, fp_cols),
        "E_ET_bitwise": compute_bit_aucs(z_et[fp_cols], test, fp_cols),
        "F_MLP_shared": compute_bit_aucs(z_mlp[fp_cols], test, fp_cols),
    }
    auc_summary: dict[str, float] = {}
    for key, values in auc_routes.items():
        auc_summary[f"{key}_mean_auc"] = float(np.nanmean(values))
        auc_summary[f"{key}_median_auc"] = float(np.nanmedian(values))

    summary = {
        "split": args.split,
        "split_note": split_note,
        "n_total": int(len(valid)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_properties": int(len(usable_props)),
        "n_spectrum_features": len(spectrum_cols),
        "ri_columns": ri_cols,
        "n_desc_features": len(desc_cols),
        "n_fingerprint_bits": len(fp_cols),
        "n_Z": len(z_cols),
        "fingerid_mean_auc": float(bit_metrics["validation_auc"].mean()),
        "fingerid_median_auc": float(bit_metrics["validation_auc"].median()),
        "property_columns": usable_props,
        "mean_r2_by_route": metrics_df.groupby("route")["r2"].mean().to_dict() if not metrics_df.empty else {},
        "median_r2_by_route": metrics_df.groupby("route")["r2"].median().to_dict() if not metrics_df.empty else {},
        "bit_auc_by_route": auc_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
