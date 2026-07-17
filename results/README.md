# Results

This directory stores reusable model and analysis outputs.

- `model.joblib`: trained model used by the Streamlit prediction app.
- `model_registry.joblib`: multi-model registry used by the app, including the consensus model.
- `applicability_domain.joblib`: training-set data and thresholds for domain and nearest-neighbor reporting.
- `model_comparison_metrics.csv`: holdout and sampled cross-validation metrics for registered models.
- `model_card.json`: machine-readable model metadata.
- `shap_feature_importance.csv`: SHAP-ranked feature importance table.
- `activity_cliffs_full.csv`, `top_20_activity_cliffs.csv`, `top_activity_cliff_compounds.csv`: activity-cliff analysis outputs.
- `scaffold_activity_summary.csv`: scaffold-level activity summary.
- `y_randomization_scores.csv`: Y-randomization validation scores.
- PNG files: result plots used for review, reporting, or app documentation.
