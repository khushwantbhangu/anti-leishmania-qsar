from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
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
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SAMPLE_SIZE = 1800


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def optional_xgboost_model():
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    return XGBRegressor(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def build_model_specs() -> dict:
    specs = {
        "random_forest": {
            "label": "Random forest",
            "description": "Bagged decision-tree ensemble robust to nonlinear descriptor-fingerprint interactions.",
            "estimator": RandomForestRegressor(
                n_estimators=90,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "extra_trees": {
            "label": "Extra trees",
            "description": "Extremely randomized tree ensemble; useful as a complementary high-variance nonlinear model.",
            "estimator": ExtraTreesRegressor(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "hist_gradient_boosting": {
            "label": "Histogram gradient boosting",
            "description": "Boosted-tree model for nonlinear relationships with a different bias profile from bagging.",
            "estimator": HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.05,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        },
        "ridge": {
            "label": "Ridge regression",
            "description": "Regularized linear baseline for comparison and failure-mode detection.",
            "estimator": make_pipeline(
                StandardScaler(with_mean=False),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            ),
        },
    }

    xgb = optional_xgboost_model()
    if xgb is not None:
        specs["xgboost"] = {
            "label": "XGBoost",
            "description": "Gradient-boosted trees when xgboost is available in the environment.",
            "estimator": xgb,
        }

    return specs


def nearest_nonself_thresholds(fingerprints: np.ndarray, descriptors: pd.DataFrame) -> dict:
    descriptor_scaler = StandardScaler()
    descriptor_scaled = descriptor_scaler.fit_transform(descriptors)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(
        np.arange(len(descriptor_scaled)),
        size=min(CV_SAMPLE_SIZE, len(descriptor_scaled)),
        replace=False,
    )

    descriptor_nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    descriptor_nn.fit(descriptor_scaled[sample_idx])
    descriptor_distances, _ = descriptor_nn.kneighbors(descriptor_scaled[sample_idx])
    descriptor_threshold = float(np.quantile(descriptor_distances[:, 1], 0.95))

    fp = fingerprints.astype(bool)
    nearest_tanimoto = []
    fp_sample = fp[sample_idx]
    for i, row in zip(sample_idx, fp_sample):
        intersection = np.logical_and(row, fp).sum(axis=1)
        union = np.logical_or(row, fp).sum(axis=1)
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )
        similarity[i] = -1.0
        nearest_tanimoto.append(float(similarity.max()))

    return {
        "descriptor_nn_distance_95": descriptor_threshold,
        "tanimoto_similarity_05": float(np.quantile(nearest_tanimoto, 0.05)),
        "descriptor_scaler": descriptor_scaler,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    X = pd.read_csv(DATA / "X_combined.csv")
    X.columns = X.columns.astype(str)
    y = pd.read_csv(DATA / "y_pIC50.csv")["pIC50"]
    dataset = pd.read_csv(DATA / "leishmania_ml_dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    specs = build_model_specs()
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_idx = X.sample(n=min(CV_SAMPLE_SIZE, len(X)), random_state=RANDOM_STATE).index
    X_cv = X.loc[cv_idx].reset_index(drop=True)
    y_cv = y.loc[cv_idx].reset_index(drop=True)
    registry_models = {}
    rows = []

    for key, spec in specs.items():
        estimator = clone(spec["estimator"])
        estimator.fit(X_train, y_train)
        test_pred = estimator.predict(X_test)
        metrics = regression_metrics(y_test, test_pred)
        cv_scores = cross_val_score(
            clone(spec["estimator"]),
            X_cv,
            y_cv,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_sd"] = float(cv_scores.std())

        registry_models[key] = {
            "label": spec["label"],
            "description": spec["description"],
            "estimator": estimator,
            "metrics": metrics,
        }
        rows.append({"model_key": key, "model": spec["label"], **metrics})

    model_comparison = (
        pd.DataFrame(rows)
        .sort_values(["r2", "mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_model_key = str(model_comparison.loc[0, "model_key"])

    voting_estimators = [
        (key, registry_models[key]["estimator"])
        for key in model_comparison["model_key"].head(min(3, len(model_comparison)))
    ]
    consensus = VotingRegressor(voting_estimators, n_jobs=-1)
    consensus.fit(X_train, y_train)
    consensus_pred = consensus.predict(X_test)
    consensus_metrics = regression_metrics(y_test, consensus_pred)
    consensus_cv = cross_val_score(consensus, X_cv, y_cv, cv=cv, scoring="r2", n_jobs=-1)
    consensus_metrics["cv_r2_mean"] = float(consensus_cv.mean())
    consensus_metrics["cv_r2_sd"] = float(consensus_cv.std())

    registry_models["consensus"] = {
        "label": "Consensus ensemble",
        "description": "Voting ensemble of the top-performing validated models.",
        "estimator": consensus,
        "metrics": consensus_metrics,
    }
    model_comparison = pd.concat(
        [
            model_comparison,
            pd.DataFrame(
                [
                    {
                        "model_key": "consensus",
                        "model": "Consensus ensemble",
                        **consensus_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values(["r2", "mae"], ascending=[False, True])

    model_comparison.to_csv(RESULTS / "model_comparison_metrics.csv", index=False)

    fingerprints = X[[column for column in X.columns if column.isdigit()]].to_numpy(dtype=np.int8)
    descriptors = X[DESCRIPTOR_COLUMNS].astype(float)
    thresholds = nearest_nonself_thresholds(fingerprints, descriptors)
    descriptor_scaler = thresholds.pop("descriptor_scaler")

    applicability_domain = {
        "feature_columns": list(X.columns),
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "fingerprint_columns": [column for column in X.columns if column.isdigit()],
        "molecule_ids": dataset["Molecule ChEMBL ID"].astype(str).tolist(),
        "smiles": dataset["SMILES"].astype(str).tolist(),
        "pIC50": dataset["pIC50"].astype(float).to_numpy(),
        "descriptors": descriptors.to_numpy(dtype=np.float32),
        "fingerprints": fingerprints,
        "descriptor_scaler": descriptor_scaler,
        "thresholds": thresholds,
        "descriptor_ranges": {
            column: {
                "min": float(descriptors[column].min()),
                "max": float(descriptors[column].max()),
                "q01": float(descriptors[column].quantile(0.01)),
                "q99": float(descriptors[column].quantile(0.99)),
            }
            for column in DESCRIPTOR_COLUMNS
        },
    }

    registry = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "pIC50",
        "feature_columns": list(X.columns),
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "fingerprint_bits": len(applicability_domain["fingerprint_columns"]),
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_sample_size": int(len(X_cv)),
        },
        "best_model_key": best_model_key,
        "default_model_key": "consensus",
        "models": registry_models,
    }

    joblib.dump(registry, RESULTS / "model_registry.joblib", compress=3)
    joblib.dump(applicability_domain, RESULTS / "applicability_domain.joblib", compress=3)
    joblib.dump(registry_models[best_model_key]["estimator"], RESULTS / "model.joblib", compress=3)

    model_card = {
        "created_at": registry["created_at"],
        "target": "pIC50",
        "samples": registry["dataset"]["n_samples"],
        "features": registry["dataset"]["n_features"],
        "default_model": registry_models["consensus"]["label"],
        "best_single_model": registry_models[best_model_key]["label"],
        "applicability_domain": thresholds,
        "metrics": model_comparison.to_dict(orient="records"),
    }
    (RESULTS / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    print(model_comparison.to_string(index=False))
    print(f"Saved registry to {RESULTS / 'model_registry.joblib'}")
    print(f"Saved applicability-domain artifact to {RESULTS / 'applicability_domain.joblib'}")


if __name__ == "__main__":
    main()
