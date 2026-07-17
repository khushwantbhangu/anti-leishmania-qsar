# Interpretable QSAR framework for anti-leishmanial drug discovery

This repository contains a reproducible computational workflow and Streamlit application for predicting anti-leishmanial activity from molecular structure. The study combines ChEMBL-derived bioactivity curation, descriptor and Morgan fingerprint feature engineering, machine-learning model validation, activity-cliff analysis, scaffold analysis, and SHAP-based interpretation.

## Scientific aim

Leishmaniasis remains a neglected tropical disease with limited therapeutic options. This project provides an interpretable QSAR workflow to help prioritize compounds with predicted anti-leishmanial activity and identify molecular features associated with potency.

## Repository structure

```text
app/           Streamlit prediction app and reusable prediction utilities
data/          Raw ChEMBL-derived inputs and processed modeling matrices
figures/       Publication-ready figures
notebooks/     Reproducible analysis notebooks, ordered by workflow stage
results/       Model registry, trained models, validation outputs, and analysis tables
manuscript/    Manuscript-facing notes and supplementary-material placeholders
```

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

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the prediction app

```bash
streamlit run app/streamlit_app.py
```

On Windows, `run_app.bat` creates `.venv` if needed, installs dependencies, and starts the Streamlit app.

## Research app features

The Streamlit app is designed for practical screening work:

- single-compound prediction with pIC50 and estimated IC50
- batch CSV screening and ranked candidate export
- consensus and individual model predictions
- applicability-domain checks
- nearest known training analogs
- model spread / approximate prediction interval
- drug-likeness and descriptor flags
- SHAP-based feature interpretation
- publication-facing model evidence view

## Data and model artifacts

The processed feature matrices are retained to support direct model reproduction. The app uses `results/model_registry.joblib` and `results/applicability_domain.joblib` when available. `results/model.joblib` stores the best single model for legacy compatibility.

See `MODEL_CARD.md` for model details, validation metrics, intended use, and limitations.

## Web availability

See `DEPLOYMENT.md` for public deployment options. The recommended route is Streamlit Community Cloud using this GitHub repository and `app/streamlit_app.py` as the entry point. Hugging Face Spaces is a good alternative for broad researcher access.

## Citation

Please cite the accompanying manuscript and this repository using `CITATION.cff`. Before public release, update the repository URL and archive a tagged release with Zenodo to obtain a DOI.

## License

This project is released under the MIT License. See `LICENSE` for details.
