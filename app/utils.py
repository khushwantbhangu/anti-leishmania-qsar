from __future__ import annotations

import html
import io
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Contrib.NP_Score import npscorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


DESCRIPTOR_COLUMNS = [
    "Molecular_Weight",
    "AlogP",
    "TPSA",
    "HBA",
    "HBD",
    "RO5_Violations",
    "Rotatable_Bonds",
    "QED",
    "Aromatic_Rings",
    "Heavy_Atoms",
    "NP_Likeness",
]
DEFAULT_FINGERPRINT_BITS = 1024
DEFAULT_RADIUS = 2


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _np_model():
    return npscorer.readNPModel()


def parse_smiles(smiles: str):
    clean = str(smiles).strip()
    if not clean:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError("Invalid SMILES. Check the molecule syntax and try again.")
    return clean, mol


def compute_descriptors(mol) -> pd.DataFrame:
    if mol is None:
        raise ValueError("Unable to parse SMILES.")

    desc = {
        "Molecular_Weight": round(float(Descriptors.MolWt(mol)), 2),
        "AlogP": round(float(Descriptors.MolLogP(mol)), 2),
        "TPSA": round(float(Descriptors.TPSA(mol)), 2),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "RO5_Violations": float(
            sum(
                [
                    Lipinski.NumHDonors(mol) > 5,
                    Lipinski.NumHAcceptors(mol) > 10,
                    Descriptors.MolLogP(mol) > 5,
                    Descriptors.MolWt(mol) > 500,
                ]
            )
        ),
        "Rotatable_Bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "QED": round(float(QED.qed(mol)), 2),
        "Aromatic_Rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Heavy_Atoms": float(mol.GetNumHeavyAtoms()),
        "NP_Likeness": round(float(npscorer.scoreMol(mol, _np_model())), 2),
    }
    return pd.DataFrame([desc], columns=DESCRIPTOR_COLUMNS)


def compute_morgan_fingerprints(
    mol,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_BITS,
) -> np.ndarray:
    if mol is None:
        raise ValueError("Unable to parse SMILES.")
    if int(n_bits) <= 0:
        raise ValueError("Fingerprint bits must be a positive integer.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )
    fp = generator.GetFingerprint(mol)
    return np.asarray(fp, dtype=np.int8)


def assemble_features_from_smiles(
    smiles: str,
    n_bits: int = DEFAULT_FINGERPRINT_BITS,
    radius: int = DEFAULT_RADIUS,
) -> pd.DataFrame:
    _, mol = parse_smiles(smiles)
    desc_df = compute_descriptors(mol)
    fp = compute_morgan_fingerprints(mol, radius=radius, n_bits=n_bits)
    fp_df = pd.DataFrame([fp], columns=[str(i) for i in range(int(n_bits))])
    return pd.concat([desc_df.reset_index(drop=True), fp_df], axis=1).astype(float)


def get_model_feature_names(model) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return [str(name) for name in names]


def fingerprint_bits_for_model(model, default: int = DEFAULT_FINGERPRINT_BITS) -> int:
    feature_names = get_model_feature_names(model)
    if feature_names:
        fingerprint_columns = [int(name) for name in feature_names if str(name).isdigit()]
        if fingerprint_columns:
            return max(fingerprint_columns) + 1

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features > len(DESCRIPTOR_COLUMNS):
        return int(n_features) - len(DESCRIPTOR_COLUMNS)

    return int(default)


def align_features_to_model(features: pd.DataFrame, model) -> pd.DataFrame:
    aligned = features.copy()
    aligned.columns = aligned.columns.astype(str)

    expected_names = get_model_feature_names(model)
    if expected_names:
        missing = [column for column in expected_names if column not in aligned.columns]
        if missing:
            raise ValueError(
                "The feature builder is missing model inputs: "
                + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
            )
        return aligned.loc[:, expected_names].astype(float)

    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and aligned.shape[1] != int(expected_count):
        raise ValueError(
            f"Model expects {expected_count} features, but the app generated "
            f"{aligned.shape[1]}. Use a model trained with the repository feature schema."
        )

    return aligned.astype(float)


def assemble_features_for_model(smiles: str, model) -> pd.DataFrame:
    n_bits = fingerprint_bits_for_model(model)
    features = assemble_features_from_smiles(smiles, n_bits=n_bits)
    return align_features_to_model(features, model)


def load_model(
    model_path: str | Path | None = None,
    uploaded_model: bytes | BinaryIO | None = None,
):
    if uploaded_model is not None:
        if isinstance(uploaded_model, bytes):
            return joblib.load(io.BytesIO(uploaded_model))
        return joblib.load(uploaded_model)

    root = _project_root()
    fallback = root / "results" / "model.joblib"

    if model_path is None:
        path = fallback
    else:
        raw = str(model_path).strip()
        if not raw or raw in {".", ".."}:
            path = fallback
        else:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (root / path).resolve()
            if not path.exists() or path.is_dir():
                path = fallback

    if not path.exists():
        X = pd.read_csv(root / "data" / "processed" / "X_combined.csv")
        y = pd.read_csv(root / "data" / "processed" / "y_pIC50.csv")["pIC50"]
        X.columns = X.columns.astype(str)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)

    return joblib.load(path)


def predict(model, features: pd.DataFrame) -> float:
    ready = align_features_to_model(features, model)
    pred = model.predict(ready)
    return float(pred[0])


def explain_shap(model, features: pd.DataFrame, top_k: int = 20) -> dict:
    try:
        ready = align_features_to_model(features, model)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(ready, check_additivity=False)
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, 0]
        elif values.ndim == 2:
            values = values[0]

        importance = (
            pd.DataFrame(
                {
                    "feature": ready.columns.astype(str),
                    "mean_abs_shap": np.abs(values).reshape(-1),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .head(int(top_k))
            .sort_values("mean_abs_shap", ascending=True)
        )
        return {"importance": importance, "raw_shap": shap_values}
    except Exception as exc:
        return {"error": str(exc)}


def pIC50_to_ic50_nm(prediction: float) -> float:
    return float(10 ** (9 - prediction))


def molecule_image_bytes(smiles: str, size: tuple[int, int] = (560, 380)) -> io.BytesIO:
    _, mol = parse_smiles(smiles)
    image = Draw.MolToImage(mol, size=size)
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output


def generate_report_html(
    smiles: str,
    prediction: float,
    features: pd.DataFrame,
    shap_info: dict,
) -> str:
    descriptor_table = features.loc[:, [col for col in DESCRIPTOR_COLUMNS if col in features.columns]]
    title = "QSAR prediction report"
    escaped_smiles = html.escape(smiles)
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;}"
        "table{border-collapse:collapse;margin-top:8px;}th,td{border:1px solid #ddd;"
        "padding:6px 8px;text-align:right;}th{background:#f3f6f8;}"
        "h1,h2{color:#1f2937;}code{background:#f3f6f8;padding:2px 4px;}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<h2>Input SMILES</h2><p><code>{escaped_smiles}</code></p>",
        f"<h2>Predicted pIC50</h2><p>{prediction:.3f}</p>",
        f"<p>Estimated IC50: {pIC50_to_ic50_nm(prediction):,.2f} nM</p>",
        "<h2>Molecular descriptors</h2>",
        descriptor_table.to_html(index=False),
    ]

    importance = shap_info.get("importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        html_parts.append("<h2>Top SHAP feature importances</h2>")
        html_parts.append(importance.sort_values("mean_abs_shap", ascending=False).to_html(index=False))
    if "error" in shap_info:
        html_parts.append("<h2>SHAP note</h2>")
        html_parts.append(f"<pre>{html.escape(str(shap_info['error']))}</pre>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def html_bytesio(html_report: str) -> io.BytesIO:
    return io.BytesIO(html_report.encode("utf-8"))
