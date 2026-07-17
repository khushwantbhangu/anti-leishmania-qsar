# Public web deployment

The Streamlit app entry point is:

```bash
app/streamlit_app.py
```

## Initial GitHub upload

This repository contains trained `.joblib` model artifacts. Track these with Git LFS before pushing:

```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add .
git commit -m "Prepare anti-leishmania QSAR app for publication"
```

If you already created an empty GitHub repository, connect it and push:

```bash
git remote add origin https://github.com/<username>/<repository>.git
git branch -M main
git push -u origin main
```

If you use GitHub CLI, authenticate first:

```bash
gh auth login
gh repo create <repository> --public --source . --remote origin --push
```

## Recommended path: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Confirm `requirements.txt` is present at the repository root.
3. In Streamlit Community Cloud, create a new app from the GitHub repository.
4. Set the main file path to `app/streamlit_app.py`.
5. Ensure the repository includes `results/model_registry.joblib`, `results/applicability_domain.joblib`, and `results/model.joblib`.
6. Deploy and test the public URL with a few known SMILES strings.

Keep only one dependency-management file in the repository root for cloud deployment. This project uses `requirements.txt`.

## Alternative path: Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select Streamlit as the Space SDK.
3. Upload or connect the repository files.
4. Keep `app/streamlit_app.py`, `requirements.txt`, `data/`, and `results/model.joblib`.
5. Keep `results/model_registry.joblib` and `results/applicability_domain.joblib` for reliability reporting.
6. Test file upload and batch prediction after the Space finishes building.

## Citation and archival release

For publication, make a tagged GitHub release and archive it with Zenodo to obtain a DOI. Update `CITATION.cff` with the final repository URL and DOI before manuscript submission.
