# Data licensing and redistribution scope

## Released here

The following derived or author-generated tables are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0):

- compound identifiers, SMILES, structural classes and split assignments;
- the 46-position fingerprint computed from molecular structure;
- property labels and their source fields;
- virtual complex-mixture design and summary tables;
- numerical evaluation tables;
- processed GC-EI-MS slices and reference values for the the three real samples measured by the authors.

## Not released here

No NIST EI mass spectra are redistributed. Synthetic co-eluting spectra and virtual-mixture slice spectra generated from those spectra are also excluded. Users who wish to rerun training must obtain the source spectra under the applicable NIST license and place them in the local, git-ignored `data/modeling/` directory.

Property labels collected from public databases remain subject to the terms of their original sources. The source fields in `data/properties/property_labels_8.csv` identify whether a value is experimental, from a public database, or completed by a documented calculation.
