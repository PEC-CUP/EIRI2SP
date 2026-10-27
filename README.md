# EIRI2SP

EIRI2SP is an interpretable route from electron ionization mass spectra (EI) and retention indices (RI) to a 46-position structural fingerprint and molecular properties. The fingerprint is defined from molecular structure, predicted from the EI spectrum and RI by EIRI2Struct, and used by EIRI2Prop to predict eight molecular properties. The same representation is extended to co-eluting spectra, virtual complex mixtures, and three real hydrocarbon mixtures through chromatographic slicing and mixing rules.

This repository contains the public derived data, the processed real-sample data, and the core scripts needed to rebuild and evaluate the model. It does not redistribute NIST spectra or synthetic spectra derived from NIST spectra.

## Repository layout

```text
data/
  library/          compound metadata, split assignment and modelling schemas
  fingerprints/     frozen 46-position fingerprints and fingerprint definitions
  properties/       eight property labels and label-source fields
  mixtures/         200-group virtual complex-mixture design and summary tables
  evaluation/       metric tables for individual compounds, co-elution and mixtures
  real_sample/      processed slice fingerprints and reference values for the three real samples
src/
  dataset_construction/
  model_training/
  simulation/
docs/
scripts/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Public data

The following files are safe to redistribute and are included in this repository:

- `data/library/compound_library.csv`: InChIKey identifier, SMILES, structural class, and the frozen training/test assignment.
- `data/library/hydrocarbon_records.csv`: compound names, CAS numbers, molecular formula, carbon and hydrogen counts, and structural class. No mass spectra are present.
- `data/fingerprints/fingerprints_46bit.csv`: the 46-position fingerprint Z for all 5,225 compounds.
- `data/properties/property_labels_8.csv`: the eight property labels and their source fields.
- `data/mixtures/`: the 200-group virtual mixture design and the mixture-level summary.
- `data/evaluation/`: numerical evaluation tables used in the paper.
- `data/real_sample/`: processed slice fingerprints and reference values for the three real samples measured by the authors.

The repository intentionally does **not** contain:

- NIST EI mass spectra or any extracted EI peak table;
- synthetic co-eluting spectra generated from NIST spectra;
- synthetic virtual-mixture slice spectra generated from NIST spectra;
- trained model binaries derived from the licensed spectra.

## Reproducing the model

The scripts expect a modelling table containing the licensed EI channels and the frozen modelling schema. Place the user-supplied file at:

```text
data/modeling/modeling_table.csv
data/modeling/modeling_schema.json
```

These paths are excluded by `.gitignore`. `docs/REPRODUCING.md` describes the required columns and the sequence of commands.

A quick structural check of the public package can be run without any licensed spectra:

```bash
python scripts/validate_public_data.py
```

## Key definitions

- The fingerprint has 19 descriptor positions and 27 substructure positions.
- The frozen order is defined in `src/z_labels.py`.
- The 80/20 split is defined in `src/simulation/split_protocol.py`: 4,180 training molecules and 1,045 held-out molecules, seed 2026.
- Ind denotes training on individual compounds only. Aug denotes training with individual compounds plus synthetic co-eluting spectra.
- NIST-derived retention indices are used for training and virtual validation. Real-sample retention indices are calibrated from the n-alkane ladder of the same chromatographic run.

## Citation

Please cite the EIRI2SP manuscript and this repository using the metadata in `CITATION.cff`.

## Licenses

Code is released under the MIT License. Public derived data are released under CC BY 4.0, subject to the source-specific notes in `DATA_LICENSE.md`.
