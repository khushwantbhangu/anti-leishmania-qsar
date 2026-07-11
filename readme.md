# Interpretable Machine Learning-Based QSAR Framework for Anti-Leishmanial Drug Discovery

## Overview
This repository accompanies a computational drug discovery study focused on anti-leishmanial activity prediction using interpretable machine learning. The workflow combines curated bioactivity data, molecular descriptors, Morgan fingerprints, model validation, and SHAP-based interpretation.

## Scientific Background
Leishmaniasis remains a neglected tropical disease with limited therapeutic options. Computational QSAR models can guide medicinal chemistry by prioritizing compounds likely to show improved biological activity and by identifying molecular features associated with potency.

## Research Gap
Many cheminformatics workflows stop at predictive performance and do not provide interpretable mechanistic insight. This project emphasizes reproducibility, transparency, and interpretability for anti-leishmanial QSAR modeling.

## Objectives
- Build a reproducible QSAR workflow for anti-leishmanial activity prediction.
- Compare descriptor-based and fingerprint-based representations.
- Evaluate predictive performance with robust validation.
- Use SHAP analysis to interpret model decisions.
- Organize the analysis for publication and reuse.

## Workflow
1. Data acquisition and cleaning.
2. Feature generation from molecular descriptors and Morgan fingerprints.
3. Model training and validation.
4. Model interpretation using SHAP.
5. Figure generation and documentation.

## Repository Structure
- data/raw: original input datasets.
- data/processed: cleaned datasets and feature matrices.
- notebooks: reproducible analysis notebooks.
- figures: publication-quality figures.
- results: model summaries and selected output tables.
- manuscript: manuscript-related materials.

## Installation
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage
Run the notebooks in the following order:
1. 01_data_exploration.ipynb
2. 02_data_cleaning.ipynb
3. 03_merge_datasets.ipynb
4. 04_exploratory_data_analysis.ipynb
5. 05_feature_engineering_and_modeling.ipynb
6. 06_model_development_and_validation.ipynb
7. 07_shap_interpretability.ipynb
8. 08_activity_cliff_analysis.ipynb
9. 09_scaffold_analysis.ipynb
10. 10_y_randomization_validation.ipynb
11. 11_predict_new_compounds.ipynb

## Dataset Description
The repository uses ChEMBL-derived datasets for anti-leishmanial activity and associated molecular descriptors. The processed dataset includes pIC50 values and engineered features suitable for machine learning.

## Results
The workflow generates predictive models, validation metrics, SHAP-based interpretations, activity cliff summaries, scaffold analyses, and publication figures.

## Figures
Representative figures are stored in the figures directory, including descriptor distributions, model comparison summaries, and interpretability outputs.

## Citation
Please cite the accompanying manuscript and this repository using the CITATION.cff file.

## License
This project is licensed under the MIT License. See LICENSE for details.

## Contact
For questions or collaboration, please contact Khushwant Singh.
