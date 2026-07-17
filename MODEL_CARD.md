# Model card

## Intended use

This project provides QSAR models for prioritizing compounds with predicted anti-leishmanial activity. Predictions are intended for research triage, hypothesis generation, and compound-list ranking. They are not a substitute for experimental validation.

## Target

- Regression target: `pIC50`
- Lower IC50 corresponds to higher pIC50 and stronger predicted activity.
- Estimated IC50 in nM is calculated as `10 ** (9 - pIC50)`.

## Feature representation

- 11 molecular descriptors:
  `Molecular_Weight`, `AlogP`, `TPSA`, `HBA`, `HBD`, `RO5_Violations`, `Rotatable_Bonds`, `QED`, `Aromatic_Rings`, `Heavy_Atoms`, `NP_Likeness`
- 1024-bit Morgan fingerprint features
- Total feature count: 1035

## Registered models

The model registry contains:

- Random forest
- Extra trees
- Histogram gradient boosting
- Ridge regression baseline
- Consensus ensemble

The consensus ensemble is the default app model because it exposes model-agreement behavior and reduces dependence on a single learner.

## Current validation metrics

| Model | Holdout R2 | MAE | RMSE | Sampled CV R2 |
|---|---:|---:|---:|---:|
| Consensus ensemble | 0.629 | 0.374 | 0.525 | 0.628 |
| Extra trees | 0.623 | 0.363 | 0.529 | 0.621 |
| Random forest | 0.609 | 0.385 | 0.539 | 0.605 |
| Histogram gradient boosting | 0.598 | 0.403 | 0.547 | 0.598 |
| Ridge regression | 0.499 | 0.463 | 0.611 | 0.486 |

Metrics are stored in `results/model_comparison_metrics.csv`.

## Reproducibility environment

The shipped joblib artifacts were trained and validated with the pinned dependencies in `requirements.txt`, including scikit-learn 1.8.0. Keep these pins for deployment unless the model registry is regenerated with `scripts/train_model_registry.py`.

## Applicability domain

The app reports two domain checks:

- Maximum Morgan fingerprint Tanimoto similarity to the training set.
- Nearest-neighbor descriptor distance after standardization.

The current thresholds are stored in `results/model_card.json` and `results/applicability_domain.joblib`. A prediction is flagged as lower confidence when structural similarity is low, descriptor distance is high, or descriptor values fall outside the central training distribution.

## Researcher-facing outputs

For each molecule, the app reports:

- predicted pIC50 and estimated IC50
- activity class
- model confidence
- priority score
- approximate prediction interval/model spread
- applicability-domain status
- nearest known training compounds
- drug-likeness flags
- SHAP feature ranking

## Limitations

- The models are trained on curated ChEMBL-derived records and inherit the limitations of the source assays.
- Applicability-domain flags should be treated seriously for structurally novel compounds.
- Predictions are best used for prioritization before wet-lab testing, not as final evidence of activity.
