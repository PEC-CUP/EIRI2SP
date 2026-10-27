# Data dictionary

## `data/library/compound_library.csv`

- `identifier`: InChIKey used as the stable compound identifier.
- `SMILES`: molecular structure.
- `split`: frozen `train` or `test` assignment.
- `class_strict`: RDKit-derived structural class.

## `data/fingerprints/fingerprints_47bit.csv`

- `identifier`: compound identifier.
- `split`: frozen split assignment.
- 47 fingerprint columns whose order is frozen in `src/z_labels.py`.

## `data/properties/property_labels_8.csv`

- `MW_g_mol`: molecular weight.
- `Tb_K`: boiling point.
- `XLogP`: XLogP.
- `Vc_cm3_mol`: critical volume.
- `Pc_MPa`: critical pressure.
- `density_g_mL`: density at 298 K.
- `Tc_K`: critical temperature.
- `Tm_K`: melting point.
- Source fields ending in `_source` record whether the value is experimental, from a public database, or completed by a documented calculation.

## `data/mixtures/`

- `virtual_mixture_selections.csv`: 200 group definitions, 124 compounds per group, and design mass fractions.
- `virtual_mixture_summary.csv`: group-level composition, structural quantities, and properties.

## `data/real_sample/`

- `sample_slices.csv`: 460 processed GC-EI-MS slices of the the three real samples, including the TIC area, retention index, and 451 normalized mass channels.
- `sample_summary.csv`: measured and predicted sample-level values.
- `sample_structural_composition.csv`: measured and predicted structural composition.
- `sample_predicted_curves.csv`: predicted boiling-point curves.
- `sample_fingerprint_neighbours.csv`: fingerprint-distance ranking against the virtual mixtures.
