# Data

This directory contains the source and processed data used for the anti-leishmanial QSAR workflow.

## raw

- `chembl_leishmania_donovani_activities.csv`: ChEMBL-derived bioactivity records.
- `chembl_leishmania_compounds.csv`: ChEMBL-derived compound metadata and molecular descriptors.

## processed

- `activities_clean.csv`: curated activity table after filtering and pIC50 calculation.
- `leishmania_ml_dataset.csv`: modeling dataset with SMILES, pIC50, and selected descriptors.
- `X_descriptors.csv`: descriptor-only feature matrix.
- `X_fingerprints.csv`: Morgan fingerprint feature matrix.
- `X_combined.csv`: combined descriptor and fingerprint feature matrix.
- `y_pIC50.csv`: modeling target vector.

The processed matrices are retained so readers can reproduce model development without rerunning every preprocessing step.
