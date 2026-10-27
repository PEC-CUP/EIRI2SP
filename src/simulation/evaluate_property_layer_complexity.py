#!/usr/bin/env python
"""Does the *property-model* layer also gain more when the overlap grows?

The mixture-augmented property models are the **production** ones: 4,180 individual
compounds plus 30,000 co-eluting windows drawn by ``draw_local_components`` (the
sampler the translator is trained with), target = the value given by the mixing
rule (``mixture_augmented_models.fit_mixture_augmented``).  They are evaluated on
molecule-disjoint test beds with n = 1 ... 11 components (anchors rejection-sampled
so that the +-40 RI window is respected), next to the individual-only models.  All
inputs are the TRUE mean fingerprint, so this isolates the property-model layer
from the translator -- i.e. it answers "does the averaged fingerprint suffice for
the mixing-rule value", and it uses exactly the models the pipeline ships.

Output: models/simulated_mixture_validation/property_layer_complexity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data                          # noqa: E402
from z_fingerprint_model import sum_normalize                           # noqa: E402
from split_protocol import split_mask, held_out_mask, MIX_SEED, \
    TEST_SEED_HELDOUT                                                   # noqa: E402
from select_virtual_mixture import draw_local_windows                   # noqa: E402
from mixture_augmented_models import component_mw, rule_value           # noqa: E402

MD = BASE / "models" / "simulated_mixture_validation"
# every integer, matching the co-elution sweep of figure 5
NCOMP = list(range(1, 12))
N_SPEC = 2000


def fit_plain(X, y, n_trees=120):
    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=n_trees,
                                            random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(X, y)
    return m


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    _tr, _te, allowed = split_mask(len(data))
    hold = held_out_mask(allowed)
    ri = data["desc_airi_ri"].to_numpy(float)
    base = (ri >= 860.0) & (ri <= 3100.0)
    mw = 12.011 * data["desc_C"].to_numpy(float) + 1.008 * data["desc_H"].to_numpy(float)
    Z = data[zcols].to_numpy(float)

    # one mixing rule per property, the same three the production pipeline uses
    #   Tb       mass-fraction weighted mean
    #   Tc/Pc/Vc Kay's rule (mole-fraction weighted)
    #   density  specific-volume mixing
    specs = {
        "Tb": ("mean", data["chem_Tb_K"].to_numpy(float)),
        "density": ("reciprocal", data["density_full_g_ml"].to_numpy(float)),
        # The mixture MW of this layer is the MASS-WEIGHTED mean of the component
        # molecular weights, matching the co-elution layer (its mean fingerprint Z-bar
        # carries mass-weighted C and H counts).  The virtual-mixture and real-sample
        # layers use the number average instead; see docs/manuscript_data_and_outline.md.
        "MW": ("mean", data["MW_full_g_mol"].to_numpy(float)),
        "Tc": ("kay", data["chem_Tc_K"].to_numpy(float)),
        "Pc": ("kay", data["chem_Pc_Pa"].to_numpy(float) * 1e-6),
        "Vc": ("kay", data["chem_Vc_m3_mol"].to_numpy(float) * 1e6)}
    mw_comp = component_mw(data)

    # the five property-model pairs are fitted ONCE with the PRODUCTION recipe
    # (4,180 individuals + 30,000 shared co-eluting windows); they do not depend
    # on the number of components in the test bed
    from mixture_augmented_models import fit_mixture_augmented

    fitted = {}
    for name, (rule, vals) in specs.items():
        fitted[name] = (
            fit_plain(Z[allowed], vals[allowed]),
            fit_mixture_augmented(zcols, vals, rule=rule, allowed_pool=allowed))
        print(f"  property model fitted ({name})")
    print("property models fitted")

    rows = []
    for n in NCOMP:
        for seed in (TEST_SEED_HELDOUT, TEST_SEED_HELDOUT + 1):
            X, Y, idx, w = draw_local_windows(
                data, spec, zcols, N_SPEC, np.random.default_rng(seed),
                allowed=base & hold, n_comp=n,
                min_candidates=None if n <= 6 else n)
            Zte = np.einsum("ij,ijk->ik", w, Z[idx])
            for name, (rule, vals) in specs.items():
                y_test = rule_value(rule, idx, w, vals, mw_comp)
                m_ind, m_mix = fitted[name]
                rows.append({
                    "property": name, "n_components": n, "seed": seed,
                    "individual_MAE":
                        float(np.abs(m_ind.predict(Zte) - y_test).mean()),
                    "mixture_aug_MAE":
                        float(np.abs(m_mix.predict(Zte) - y_test).mean())})
        print(f"  n = {n} done")
    t = pd.DataFrame(rows)
    t["reduction_%"] = 100 * (1 - t.mixture_aug_MAE / t.individual_MAE)
    t.to_csv(MD / "property_layer_complexity.csv", index=False)
    print("saved", MD / "property_layer_complexity.csv")
    print(t.pivot_table(index="property", columns="n_components",
                        values="reduction_%").round(1).to_string())


if __name__ == "__main__":
    main()
