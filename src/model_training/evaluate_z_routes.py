#!/usr/bin/env python
"""Per-bit evaluation of EI+RI -> Z, EI -> Z, and RI -> Z.

For every Z bit, report R2 (continuous descriptors) or ROC-AUC (0/1 bits).
This is the single-molecule structural-fingerprint evaluation, run with TIC
(sum) normalized EI spectra.

Two splits are available and each one writes to its own directory, so the
results never overwrite each other:

  --split random         -> models/z_routes_evaluation/           (default)
  --split formula_group  -> models/z_routes_evaluation_formula_group/

`formula_group` reproduces exactly the GroupShuffleSplit by molecular formula
used by ``train_qspr_routes.py --split formula_group`` (same seed, same test
size, same grouping column), so the two runs can be compared directly.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "src"))
from predict_mixture import load_training_data
from z_fingerprint_model import sum_normalize
from z_labels import BITS, DESCRIPTORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "formula_group"],
                        default="random")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--records",
                        default=str(BASE / "data/hydrocarbon_records.csv"))
    parser.add_argument("--out-dir", default=None,
                        help="defaults to models/z_routes_evaluation[_<split>]")
    return parser.parse_args()


def make_split(data: pd.DataFrame, args: argparse.Namespace) -> tuple:
    """Reproduce the split used by train_qspr_routes.py."""
    if args.split == "formula_group":
        records = pd.read_csv(args.records, usecols=["identifier", "FORMULA"])
        grouped = data.merge(records, on="identifier", how="left")
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                     random_state=args.seed)
        tr_idx, te_idx = next(splitter.split(grouped, groups=grouped["FORMULA"]))
        return tr_idx, te_idx, "GroupShuffleSplit by molecular formula"
    tr_idx, te_idx = train_test_split(np.arange(len(data)), test_size=args.test_size,
                                      random_state=args.seed)
    return tr_idx, te_idx, "Random molecule split"


def main() -> None:
    args = parse_args()
    data, spec, zcols, props = load_training_data()
    data = data.copy()
    spec_X = sum_normalize(data[spec].to_numpy(float))
    ri = data["desc_airi_ri"].to_numpy(float)

    routes = {
        "EI": spec_X,
        "RI": ri[:, None],
        "EI_RI": np.hstack([spec_X, ri[:, None]]),
    }

    tr_idx, te_idx, split_note = make_split(data, args)
    Y = data[zcols].to_numpy(float)
    binary = [c for c in zcols if set(np.unique(data[c].dropna())) <= {0.0, 1.0}]

    rows = []
    for route, X in routes.items():
        Xtr, Xte = X[tr_idx], X[te_idx]
        model = Pipeline([
            ("imputer", SimpleImputer()),
            ("model", ExtraTreesRegressor(n_estimators=80, random_state=2026, min_samples_leaf=2, n_jobs=1)),
        ])
        model.fit(Xtr, Y[tr_idx])
        Ypred = model.predict(Xte)
        for j, c in enumerate(zcols):
            y_true = Y[te_idx, j]
            y_pred = Ypred[:, j]
            if c in binary:
                r2 = np.nan
                auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan
            else:
                r2 = r2_score(y_true, y_pred)
                auc = np.nan
            rows.append({"bit": c, "route": route, "kind": "binary" if c in binary else "continuous",
                         "r2": r2, "auc": auc})

    df = pd.DataFrame(rows)
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        name = "z_routes_evaluation" if args.split == "random" \
            else f"z_routes_evaluation_{args.split}"
        out = BASE / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "z_bit_metrics.csv", index=False)

    summary = {"split": args.split, "split_note": split_note,
               "seed": args.seed, "test_size": args.test_size,
               "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}
    for route in routes:
        sub = df[df.route == route]
        cont = sub[sub.kind == "continuous"]
        binar = sub[sub.kind == "binary"]
        summary[f"{route}_descriptor_mean_r2"] = float(cont["r2"].mean())
        summary[f"{route}_descriptor_median_r2"] = float(cont["r2"].median())
        summary[f"{route}_fingerprint_mean_auc"] = float(binar["auc"].mean())
        summary[f"{route}_fingerprint_median_auc"] = float(binar["auc"].median())

    # Same three blocks that figure 15 reports, so the splits are comparable.
    m = df[df.route == "EI_RI"]
    cont15 = [c for c in DESCRIPTORS if c in zcols and c not in binary]
    flags5 = [c for c in DESCRIPTORS if c in zcols and c in binary]
    bits27 = [c for c in BITS if c in zcols]
    for tag, codes, col in (("15_continuous", cont15, "r2"),
                            ("5_flags", flags5, "auc"),
                            ("27_bits", bits27, "auc")):
        vals = m[m.bit.isin(codes)][col]
        summary[f"EI_RI_{tag}_mean"] = float(vals.mean())
        summary[f"EI_RI_{tag}_median"] = float(vals.median())
        summary[f"EI_RI_{tag}_n"] = int(vals.notna().sum())

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"split={args.split} ({split_note})  n_train={len(tr_idx)} "
          f"n_test={len(te_idx)}\nsaved {out / 'z_bit_metrics.csv'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
