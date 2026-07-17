# Streamlit app

Run locally (from the repository root):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

On Windows, `run_app.bat` creates `.venv` if needed, installs dependencies, and starts the app.

## Notes

- The app works out of the box by loading the model registry in `results/model_registry.joblib`.
- Feature generation is aligned to the loaded model schema. The bundled registry uses 11 molecular descriptors plus 1024 Morgan fingerprint bits.
- The default model is a consensus ensemble; individual registered models can also be selected.
- Reliability output includes applicability-domain status, nearest known analogs, descriptor warnings, model spread, and drug-likeness flags.
- You can provide your own trained model by placing it at `results/model.joblib` or uploading a `.joblib` / `.pkl` file in the sidebar.
- Single-compound predictions include descriptor checks, a molecule rendering, SHAP interpretation, nearest analogs, model agreement, and an HTML report export.
- Batch predictions accept a CSV upload or public CSV URL with a `smiles` column and export ranked screening results as CSV.

## Inputs

- To run batch predictions you can upload a CSV with a `smiles` column, or paste a public CSV URL into the app (for example a raw GitHub link or an S3/HTTP URL).
- You can supply a trained model either by placing `model.joblib` in `results/` or by uploading the `.joblib` file directly in the UI.

## Security note

- When using public URLs, ensure the link points to a trusted CSV; the app will download and parse it locally.
