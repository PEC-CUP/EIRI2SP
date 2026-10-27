# Reproducing EIRI2SP

## Inputs supplied by the user

The licensed NIST-derived inputs are not part of this repository. Place the following files locally:

```text
data/modeling/modeling_table.csv
data/modeling/modeling_schema.json
```

`modeling_table.csv` must contain the identifier, SMILES, molecular descriptors, and EI channels used by the frozen 451-channel representation. `modeling_schema.json` must define the spectrum and descriptor columns. The published `data/library/modeling_schema.json` can be used as the schema reference.

## 1. Check the public package

```bash
python scripts/validate_public_data.py
```

## 2. Rebuild the 46-position fingerprint

```bash
python src/dataset_construction/build_hydrocarbon_substructure_fingerprints.py
```

The frozen fingerprint order is defined in `src/z_labels.py`. The published `data/fingerprints/fingerprints_47bit.csv` is the version used in the manuscript.

## 3. Train the individual-only and augmented translators

```bash
python src/simulation/train_production_translator_split.py
```

The script uses the 4,180 training molecules and draws 30,000 synthetic co-eluting spectra from that pool. It writes the individual-only and augmented translators to the configured model directory. The 1,045 held-out molecules are not used in training.

## 4. Train and evaluate the property models

```bash
python src/model_training/train_qspr_routes.py
python src/model_training/evaluate_z_routes.py
```

The individual property model takes the 46-position fingerprint as input. The mixture-augmented property model takes the mass-weighted mean fingerprint and uses the mixing rules in `src/model_training/mixing_rules.py`.

## 5. Evaluate co-elution and virtual mixtures

```bash
python src/simulation/evaluate_coelution_complexity.py
python src/simulation/evaluate_property_layer_complexity.py
python src/simulation/evaluate_virtual_mixture.py
```

## 6. Process the real sample

```bash
python src/dataset_construction/parse_mzml.py
python src/simulation/real_samples.py
python src/simulation/evaluate_real_samples.py
python src/simulation/real_structural_composition.py
```

## Scope note

The repository does not contain figure-generation scripts or manuscript-production files. The numerical tables needed to rebuild the figures are in `data/evaluation/`.
