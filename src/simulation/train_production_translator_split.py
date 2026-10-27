"""Production translator on the 80/20 molecule split (the fair protocol).

The current production translator is fitted on **all 5,225** individual
compounds plus 30,000 mixtures, so neither the individual-compound side nor any
mixture test has a held-out molecule.  The 2026-10-11 protocol is:

  * individual part of the training set = the **same 4,180 training compounds**
    the individual-only model uses (rule 1);
  * the 30,000 co-eluting spectra draw their components **only from those 4,180**
    compounds;
  * the 1,045 held-out compounds are used for the individual-compound check and
    for the new-molecule mixture tests.

Writes ``models/simulated_mixture_validation/augmented_z_model_split.joblib``
and the matching individual-only model
``models/simulated_mixture_validation/pure_z_model_split.joblib``.  The existing
production files are left untouched so the earlier numbers stay reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "models" / "simulated_mixture_validation"
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "model_training"))
sys.path.insert(0, str(BASE / "src" / "simulation"))

from predict_mixture import load_training_data                        # noqa: E402
from z_fingerprint_model import sum_normalize                         # noqa: E402
from mixture_sampling import N_MIX                                    # noqa: E402
from select_virtual_mixture import draw_local_windows                 # noqa: E402

SPLIT_SEED = 2026
MIX_SEED = 2026


def main() -> None:
    data, spec, zcols, _ = load_training_data()
    data = data.copy()
    data[spec] = sum_normalize(data[spec].to_numpy(float))
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2,
                              random_state=SPLIT_SEED)
    allowed = np.zeros(len(data), bool)
    allowed[tr] = True
    print(f"training compounds {len(tr)}, held-out {len(te)}")

    X = np.hstack([data[spec].to_numpy(float),
                   data["desc_airi_ri"].to_numpy(float)[:, None]])
    Y = data[zcols].to_numpy(float)
    # the co-eluting training spectra are local windows of the *same* object the
    # virtual complex mixtures describe: same 20-cell family design, same smooth
    # bell over the retention index, same synthesis
    Xm, Ym, _idx, _w = draw_local_windows(
        data, spec, zcols, N_MIX, np.random.default_rng(MIX_SEED),
        allowed=allowed)
    print(f"co-eluting training spectra {len(Xm)} (components from the training "
          f"compounds only)")
    imp = SimpleImputer().fit(X[tr])

    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(imp.transform(X[tr]), Y[tr])
    joblib.dump(m, OUT / "pure_z_model_split.joblib")
    print("saved", OUT / "pure_z_model_split.joblib")

    m = Pipeline([("i", SimpleImputer()),
                  ("m", ExtraTreesRegressor(n_estimators=120, random_state=2026,
                                            min_samples_leaf=2, n_jobs=1))])
    m.fit(np.vstack([imp.transform(X[tr]), imp.transform(Xm)]),
          np.vstack([Y[tr], Ym]))
    joblib.dump(m, OUT / "augmented_z_model_split.joblib")
    print("saved", OUT / "augmented_z_model_split.joblib")


if __name__ == "__main__":
    main()
