# Using the web app

Public app:

https://anti-leishmania-qsar-f26lzfytm944qspa6db8ma.streamlit.app/

The app estimates anti-leishmanial activity from molecular SMILES strings. Results are intended for compound prioritization and should be interpreted together with the reported confidence and applicability-domain information.

## Single-compound prediction

1. Open the web app.
2. Select **Single compound**.
3. Paste a valid SMILES string.
4. Click **Analyze compound**.
5. Review the prediction, confidence label, applicability-domain status, nearest known analogs, model agreement, descriptor flags, and SHAP feature ranking.

Example SMILES:

```text
CC(=O)OC1=CC=CC=C1C(=O)O
```

## Batch screening

Use **Batch screening** to score multiple compounds at once.

The CSV file must contain a column named `smiles`.

Example:

```csv
smiles,compound_id
CC(=O)OC1=CC=CC=C1C(=O)O,CMPD_001
CCOC(=O)C1=CC=CC=C1,CMPD_002
```

The exported results include predicted pIC50, estimated IC50, activity class, confidence, applicability-domain information, nearest known analog, model spread, drug-likeness flags, and a priority score.

## Interpreting outputs

- **pIC50**: predicted potency on a logarithmic scale. Higher values indicate stronger predicted activity.
- **Estimated IC50**: back-calculated nanomolar estimate from predicted pIC50. Lower values indicate stronger predicted activity.
- **Confidence**: summary of applicability-domain status and model-spread behavior.
- **Applicability domain**: indicates whether the molecule is similar enough to the training data for the prediction to be more reliable.
- **Nearest known analogs**: training-set molecules most similar to the query structure.
- **Priority score**: screening-oriented score combining predicted potency, domain status, uncertainty, and simple drug-likeness checks.
- **SHAP features**: model features contributing most strongly to the prediction.

## Recommended use

Use the tool for early-stage screening, analog comparison, and candidate prioritization. Treat predictions as computational hypotheses that require experimental validation.
