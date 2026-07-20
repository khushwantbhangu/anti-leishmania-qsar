# Anti-leishmania QSAR

This repository contains a QSAR workflow and Streamlit app for estimating anti-leishmanial activity from molecular structure. The workflow combines ChEMBL-derived bioactivity curation, molecular descriptors, Morgan fingerprints, regression models, applicability-domain checks, activity-cliff analysis, scaffold analysis, and SHAP interpretation.
https://doi.org/10.5281/zenodo.21418535

## Web app

Use the app here:

https://anti-leishmania-qsar-f26lzfytm944qspa6db8ma.streamlit.app/

The app supports single-compound prediction and batch screening from a CSV file with a `smiles` column. See `USAGE.md` for input format, example SMILES strings, and output interpretation.

## Aim

Leishmaniasis is a neglected tropical disease with limited therapeutic options. The goal of this project is to support early-stage compound prioritization by providing transparent model outputs rather than a single unqualified prediction.

## Repository structure

```text
app/           Streamlit app and prediction utilities
data/          Raw ChEMBL-derived inputs and processed modeling matrices
figures/       Figures from exploratory analysis and validation
notebooks/     Reproducible analysis notebooks, ordered by workflow stage
results/       Model registry, trained models, validation outputs, and analysis tables
manuscript/    Manuscript notes and supplementary-material files
```

## App outputs

For each molecule, the app reports:

- predicted pIC50 and estimated IC50
- activity class
- confidence label
- applicability-domain status
- nearest known training analogs
- model agreement and model spread
- drug-likeness and descriptor flags
- SHAP-based feature interpretation
- screening priority score

## Reproducible workflow

Run notebooks in numerical order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_merge_datasets.ipynb`
4. `notebooks/04_exploratory_data_analysis.ipynb`
5. `notebooks/05_feature_engineering_and_modeling.ipynb`
6. `notebooks/06_model_development_and_validation.ipynb`
7. `notebooks/07_shap_interpretability.ipynb`
8. `notebooks/08_activity_cliff_analysis.ipynb`
9. `notebooks/09_scaffold_analysis.ipynb`
10. `notebooks/10_y_randomization_validation.ipynb`
11. `notebooks/11_predict_new_compounds.ipynb`

## Data and model artifacts

The processed feature matrices are included to support model reproduction. The app uses `results/model_registry.joblib` and `results/applicability_domain.joblib` when available. `results/model.joblib` stores the best single model for compatibility with older workflows.

See `MODEL_CARD.md` for model details, validation metrics, intended use, and limitations.

## Citation

If you use this project in your research, please cite this repository using the information available in the `CITATION.cff` file.

**DOI:** https://doi.org/10.5281/zenodo.21418535

## License

This project is released under the MIT License. See `LICENSE` for details.
