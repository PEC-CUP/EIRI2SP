# Release manifest

## Included

| Path | Content |
|---|---|
| `data/library/compound_library.csv` | 5,225 compound identifiers, SMILES, formula, structural class and frozen split |
| `data/library/hydrocarbon_records.csv` | names, CAS numbers, formulas, C/H counts and structural classes |
| `data/library/modeling_schema.json` | modelling schema metadata |
| `data/fingerprints/fingerprints_46bit.csv` | the 46-position structural fingerprint used in the paper |
| `data/fingerprints/descriptor_definitions.csv` | definitions of the 19 descriptor positions |
| `data/fingerprints/hydrocarbon_substructure_fingerprints.csv` | substructure positions of the frozen fingerprint |
| `data/properties/property_labels_8.csv` | eight property targets for all 5,225 compounds |
| `data/mixtures/virtual_mixture_selections.csv` | 200 virtual mixture designs and compound mass fractions |
| `data/mixtures/virtual_mixture_summary.csv` | virtual mixture composition and property summary |
| `data/evaluation/*.csv` | numerical evaluation tables for individual, co-eluting and mixture layers |
| `data/real_sample/sample_summary.csv` | bulk property reference and prediction for the three real samples |
| `data/real_sample/sample_structure.csv` | structural-family composition for the three real samples |
| `data/real_sample/sample_distillation.csv` | distillation reference and prediction for the three real samples |
| `data/real_sample/sample_1_slices.csv` | processed slice fingerprints for sample 1# |
| `data/real_sample/sample_2_slices.csv` | processed slice fingerprints for sample 2# |
| `data/real_sample/sample_3_slices.csv` | processed slice fingerprints for sample 3# |
| `src/dataset_construction/` | fingerprint and modelling-table construction scripts |
| `src/model_training/` | translator and property-model training/evaluation scripts |
| `src/simulation/` | co-elution, virtual-mixture and real-sample scripts |
| `scripts/validate_public_data.py` | public package integrity check |
| `scripts/export_public_tables.py` | export of the public tables from local licensed inputs |

## Excluded

| Content | Reason |
|---|---|
| NIST EI mass spectra | NIST database redistribution terms |
| Extracted EI peak matrices | direct representation of NIST spectra |
| Synthetic co-eluting spectra | derived from NIST EI spectra |
| Synthetic virtual-mixture slice spectra | derived from NIST EI spectra |
| Trained model binaries | trained on licensed spectra and not required for data reuse |
| Manuscript figures and raw experimental data | not part of the public data/code release |
| Figure-generation scripts and manuscript files | not part of the public data/code release |
