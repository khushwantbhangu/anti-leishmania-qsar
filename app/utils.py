from __future__ import annotations

import html
import io
import json
import contextlib
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
MODEL_REGISTRY_PATH = "results/model_registry.joblib"
APPLICABILITY_DOMAIN_PATH = "results/applicability_domain.joblib"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _np_model():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return npscorer.readNPModel()


def parse_smiles(smiles: str):
    clean = str(smiles).strip()
    if not clean:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError("Invalid SMILES. Check the molecule syntax and try again.")
    return clean, mol


def canonicalize_smiles(smiles: str) -> str:
    _, mol = parse_smiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


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


def load_model_registry(path: str | Path | None = None) -> dict | None:
    registry_path = _project_root() / MODEL_REGISTRY_PATH if path is None else Path(path)
    if not registry_path.is_absolute():
        registry_path = (_project_root() / registry_path).resolve()
    if not registry_path.exists():
        return None
    return joblib.load(registry_path)


def load_applicability_domain(path: str | Path | None = None) -> dict | None:
    ad_path = _project_root() / APPLICABILITY_DOMAIN_PATH if path is None else Path(path)
    if not ad_path.is_absolute():
        ad_path = (_project_root() / ad_path).resolve()
    if not ad_path.exists():
        return None
    return joblib.load(ad_path)


def load_model_comparison() -> pd.DataFrame:
    path = _project_root() / "results" / "model_comparison_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_model_card() -> dict:
    path = _project_root() / "results" / "model_card.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_options(registry: dict | None) -> dict[str, str]:
    if not registry:
        return {}
    return {
        key: entry.get("label", key)
        for key, entry in registry.get("models", {}).items()
    }


def estimator_from_registry(registry: dict, model_key: str | None = None):
    if not registry:
        raise ValueError("Model registry is not available.")
    key = model_key or registry.get("default_model_key") or registry.get("best_model_key")
    models = registry.get("models", {})
    if key not in models:
        raise ValueError(f"Model '{key}' was not found in the registry.")
    return models[key]["estimator"]


def registry_model_metadata(registry: dict | None, model_key: str | None) -> dict:
    if not registry or not model_key:
        return {}
    return registry.get("models", {}).get(model_key, {})


def predict(model, features: pd.DataFrame) -> float:
    ready = align_features_to_model(features, model)
    pred = model.predict(ready)
    return float(pred[0])


def model_prediction_panel(registry: dict | None, features: pd.DataFrame) -> pd.DataFrame:
    if not registry:
        return pd.DataFrame()

    rows = []
    for key, entry in registry.get("models", {}).items():
        try:
            estimator = entry["estimator"]
            ready = align_features_to_model(features, estimator)
            rows.append(
                {
                    "model_key": key,
                    "model": entry.get("label", key),
                    "pIC50": float(estimator.predict(ready)[0]),
                    "r2": entry.get("metrics", {}).get("r2"),
                    "mae": entry.get("metrics", {}).get("mae"),
                    "rmse": entry.get("metrics", {}).get("rmse"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_key": key,
                    "model": entry.get("label", key),
                    "pIC50": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def prediction_interval(model, features: pd.DataFrame) -> dict:
    ready = align_features_to_model(features, model)

    if hasattr(model, "estimators_"):
        tree_predictions = []
        estimators = getattr(model, "estimators_", [])
        for estimator in estimators:
            if isinstance(estimator, tuple):
                estimator = estimator[1]
            try:
                tree_predictions.append(float(estimator.predict(ready)[0]))
            except Exception:
                pass
        if len(tree_predictions) >= 2:
            values = np.asarray(tree_predictions, dtype=float)
            return {
                "lower": float(np.quantile(values, 0.05)),
                "upper": float(np.quantile(values, 0.95)),
                "std": float(values.std()),
                "source": "ensemble_members",
            }

    return {"lower": np.nan, "upper": np.nan, "std": np.nan, "source": "unavailable"}


def _candidate_shap_estimators(model):
    yield model
    named_estimators = getattr(model, "named_estimators_", None)
    if isinstance(named_estimators, dict):
        for estimator in named_estimators.values():
            yield estimator
    for estimator in getattr(model, "estimators_", []):
        if isinstance(estimator, tuple):
            estimator = estimator[1]
        yield estimator


def explain_shap(model, features: pd.DataFrame, top_k: int = 20) -> dict:
    last_error = None
    for candidate in _candidate_shap_estimators(model):
        try:
            ready = align_features_to_model(features, candidate)
            explainer = shap.TreeExplainer(candidate)
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
            return {
                "importance": importance,
                "raw_shap": shap_values,
                "model_used": candidate.__class__.__name__,
            }
        except Exception as exc:
            last_error = exc

    try:
        message = str(last_error) if last_error else "No compatible estimator was available for SHAP."
    except Exception:
        message = "No compatible estimator was available for SHAP."
    return {"error": message}


def pIC50_to_ic50_nm(prediction: float) -> float:
    return float(10 ** (9 - prediction))


def activity_class(prediction: float) -> str:
    if prediction >= 7:
        return "high"
    if prediction >= 6:
        return "moderate"
    if prediction >= 5:
        return "weak"
    return "low"


def property_flags(features: pd.DataFrame) -> dict:
    descriptors = features.loc[:, [col for col in DESCRIPTOR_COLUMNS if col in features.columns]].iloc[0]
    ro5_violations = int(descriptors["RO5_Violations"])
    flags = {
        "lipinski_pass": ro5_violations == 0,
        "ro5_violations": ro5_violations,
        "high_lipophilicity": bool(descriptors["AlogP"] > 5),
        "high_molecular_weight": bool(descriptors["Molecular_Weight"] > 500),
        "low_qed": bool(descriptors["QED"] < 0.35),
        "high_tpsa": bool(descriptors["TPSA"] > 140),
    }
    warnings = []
    if not flags["lipinski_pass"]:
        warnings.append(f"{ro5_violations} Lipinski rule-of-five violation(s)")
    if flags["low_qed"]:
        warnings.append("Low QED drug-likeness")
    if flags["high_tpsa"]:
        warnings.append("High polar surface area")
    flags["warnings"] = warnings
    return flags


def tanimoto_similarity_to_training(features: pd.DataFrame, ad: dict) -> tuple[np.ndarray, np.ndarray]:
    fingerprint_columns = ad["fingerprint_columns"]
    query = features.loc[:, fingerprint_columns].to_numpy(dtype=bool)[0]
    training = np.asarray(ad["fingerprints"], dtype=bool)
    intersection = np.logical_and(query, training).sum(axis=1)
    union = np.logical_or(query, training).sum(axis=1)
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )
    order = np.argsort(-similarity)
    return similarity, order


def descriptor_distance_to_training(features: pd.DataFrame, ad: dict) -> tuple[np.ndarray, np.ndarray]:
    descriptor_columns = ad["descriptor_columns"]
    scaler = ad["descriptor_scaler"]
    training_frame = pd.DataFrame(ad["descriptors"], columns=descriptor_columns)
    training = scaler.transform(training_frame)
    query = scaler.transform(features.loc[:, descriptor_columns].astype(float))
    distances = np.linalg.norm(training - query[0], axis=1)
    order = np.argsort(distances)
    return distances, order


def applicability_report(features: pd.DataFrame, ad: dict | None, top_n: int = 5) -> dict:
    if not ad:
        return {
            "available": False,
            "in_domain": None,
            "confidence": "unknown",
            "warnings": ["Applicability-domain artifact is not available."],
            "neighbors": pd.DataFrame(),
        }

    similarity, similarity_order = tanimoto_similarity_to_training(features, ad)
    distances, distance_order = descriptor_distance_to_training(features, ad)
    descriptor_ranges = ad.get("descriptor_ranges", {})

    descriptor_outliers = []
    for column, ranges in descriptor_ranges.items():
        value = float(features[column].iloc[0])
        if value < ranges["q01"] or value > ranges["q99"]:
            descriptor_outliers.append(
                {
                    "descriptor": column,
                    "value": value,
                    "reference_q01": ranges["q01"],
                    "reference_q99": ranges["q99"],
                }
            )

    max_similarity = float(similarity[similarity_order[0]])
    nearest_distance = float(distances[distance_order[0]])
    tanimoto_threshold = ad["thresholds"]["tanimoto_similarity_05"]
    descriptor_threshold = ad["thresholds"]["descriptor_nn_distance_95"]

    in_domain = (
        max_similarity >= tanimoto_threshold
        and nearest_distance <= descriptor_threshold
        and len(descriptor_outliers) <= 2
    )

    warnings = []
    if max_similarity < tanimoto_threshold:
        warnings.append("Low structural similarity to the training set")
    if nearest_distance > descriptor_threshold:
        warnings.append("Descriptor profile is distant from the training distribution")
    if descriptor_outliers:
        warnings.append(f"{len(descriptor_outliers)} descriptor(s) outside the central training range")

    neighbor_rows = []
    for idx in similarity_order[:top_n]:
        neighbor_rows.append(
            {
                "Molecule ChEMBL ID": ad["molecule_ids"][idx],
                "SMILES": ad["smiles"][idx],
                "known_pIC50": float(ad["pIC50"][idx]),
                "tanimoto_similarity": float(similarity[idx]),
                "descriptor_distance": float(distances[idx]),
            }
        )

    return {
        "available": True,
        "in_domain": bool(in_domain),
        "confidence": "high" if in_domain and max_similarity >= 0.55 else ("medium" if in_domain else "low"),
        "max_tanimoto_similarity": max_similarity,
        "nearest_descriptor_distance": nearest_distance,
        "tanimoto_threshold": float(tanimoto_threshold),
        "descriptor_distance_threshold": float(descriptor_threshold),
        "descriptor_outliers": descriptor_outliers,
        "warnings": warnings,
        "neighbors": pd.DataFrame(neighbor_rows),
    }


def confidence_label(applicability: dict, model_std: float | None = None) -> str:
    base = applicability.get("confidence", "unknown")
    if model_std is not None and np.isfinite(model_std):
        if model_std > 0.45:
            return "low"
        if model_std > 0.25 and base == "high":
            return "medium"
    return base


def priority_score(prediction: float, applicability: dict, flags: dict, model_std: float | None = None) -> float:
    potency = np.clip((prediction - 4.5) / 2.5, 0, 1) * 55
    domain = 20 if applicability.get("in_domain") else 0
    qed = 10 if not flags.get("low_qed") else 3
    lipinski = 10 if flags.get("lipinski_pass") else 2
    uncertainty = 5
    if model_std is not None and np.isfinite(model_std):
        uncertainty = max(0, 5 - (model_std * 8))
    return float(np.clip(potency + domain + qed + lipinski + uncertainty, 0, 100))


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
    applicability: dict | None = None,
    model_panel: pd.DataFrame | None = None,
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
        f"<p>Predicted activity class: {activity_class(prediction)}</p>",
        "<h2>Molecular descriptors</h2>",
        descriptor_table.to_html(index=False),
    ]

    if applicability and applicability.get("available"):
        html_parts.append("<h2>Applicability domain</h2>")
        html_parts.append(
            "<p>"
            f"In domain: {applicability.get('in_domain')}<br>"
            f"Confidence: {html.escape(str(applicability.get('confidence')))}<br>"
            f"Nearest Tanimoto similarity: {applicability.get('max_tanimoto_similarity', float('nan')):.3f}<br>"
            f"Nearest descriptor distance: {applicability.get('nearest_descriptor_distance', float('nan')):.3f}"
            "</p>"
        )
        neighbors = applicability.get("neighbors")
        if isinstance(neighbors, pd.DataFrame) and not neighbors.empty:
            html_parts.append("<h3>Nearest known training compounds</h3>")
            html_parts.append(neighbors.to_html(index=False))

    if isinstance(model_panel, pd.DataFrame) and not model_panel.empty:
        html_parts.append("<h2>Model agreement</h2>")
        html_parts.append(model_panel.to_html(index=False))

    importance = shap_info.get("importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        html_parts.append("<h2>Top SHAP feature importances</h2>")
        if shap_info.get("model_used"):
            html_parts.append(f"<p>Explanation model: {html.escape(str(shap_info['model_used']))}</p>")
        html_parts.append(importance.sort_values("mean_abs_shap", ascending=False).to_html(index=False))
    if "error" in shap_info:
        html_parts.append("<h2>SHAP note</h2>")
        html_parts.append(f"<pre>{html.escape(str(shap_info['error']))}</pre>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def html_bytesio(html_report: str) -> io.BytesIO:
    return io.BytesIO(html_report.encode("utf-8"))
