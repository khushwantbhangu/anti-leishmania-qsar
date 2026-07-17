# Streamlit app

Public app:

https://anti-leishmania-qsar-f26lzfytm944qspa6db8ma.streamlit.app/

## Notes

- The app works out of the box by loading the model registry in `results/model_registry.joblib`.
- Feature generation is aligned to the loaded model schema. The bundled registry uses 11 molecular descriptors plus 1024 Morgan fingerprint bits.
- The default model is a consensus ensemble; individual registered models can also be selected.
- Reliability output includes applicability-domain status, nearest known analogs, descriptor warnings, model spread, and drug-likeness flags.
- Single-compound predictions include descriptor checks, a molecule rendering, SHAP interpretation, nearest analogs, model agreement, and an HTML report export.
- Batch predictions accept a CSV upload or public CSV URL with a `smiles` column and export ranked screening results as CSV.

## Inputs

- Batch screening accepts a CSV file with a `smiles` column, or a direct public CSV URL.
- Single-compound prediction accepts one valid SMILES string.

## Security note

- When using public URLs, ensure the link points to a trusted CSV; the app will download and parse it locally.
