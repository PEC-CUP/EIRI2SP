# Compound library

This directory contains structure-level metadata and the frozen split only. It contains no EI mass spectra.

- `compound_library.csv`: identifier, SMILES, structural class and `train`/`test` assignment.
- `hydrocarbon_records.csv`: names, CAS numbers, formulas, C/H counts and structural classes.
- `hydrocarbon_class_strict.csv`: RDKit-derived four-class assignment used to check class consistency.
- `modeling_schema.json` and `final_schema.json`: column definitions for a locally supplied modelling table.

To retrain the spectral models, place the licensed EI modelling table under the git-ignored `data/modeling/` directory.
