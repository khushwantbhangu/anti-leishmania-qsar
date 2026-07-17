# Deployment

The Streamlit entry point is:

```text
app/streamlit_app.py
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repository.
3. Use the following settings:

```text
Repository: khushwantbhangu/anti-leishmania-qsar
Branch: main
Main file path: app/streamlit_app.py
Python version: 3.13
```

4. Confirm that `requirements.txt` and `packages.txt` are present in the repository root.
5. Confirm that these model artifacts are available:
   - `results/model_registry.joblib`
   - `results/applicability_domain.joblib`
   - `results/model.joblib`
6. Deploy and test a few valid SMILES strings.

The repository uses Git LFS for `.joblib` model files. If model files do not load during deployment, check that Git LFS objects were uploaded with the repository.

## Hugging Face Spaces

Hugging Face Spaces can also host the app. Use the Streamlit SDK or a Docker-based Space, keep the same entry point, and include the files listed above.

## Release archive

For citation, create a tagged GitHub release and archive it with Zenodo. Add the final DOI and repository URL to `CITATION.cff`.
