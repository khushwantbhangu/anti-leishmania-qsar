from __future__ import annotations

import io
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import (  # noqa: E402
    DESCRIPTOR_COLUMNS,
    DEFAULT_FINGERPRINT_BITS,
    activity_class,
    align_features_to_model,
    applicability_report,
    canonicalize_smiles,
    confidence_label,
    estimator_from_registry,
    explain_shap,
    generate_report_html,
    html_bytesio,
    load_applicability_domain,
    load_model_card,
    load_model_comparison,
    load_model_registry,
    model_options,
    model_prediction_panel,
    molecule_image_bytes,
    pIC50_to_ic50_nm,
    predict,
    prediction_interval,
    priority_score,
    property_flags,
    registry_model_metadata,
    assemble_features_for_model,
)


EXAMPLE_SMILES = "CCOC(=O)C1=CC=CC=C1"


st.set_page_config(
    page_title="Anti-leishmania QSAR research tool",
    page_icon=":material/science:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_registry():
    return load_model_registry()


@st.cache_resource(show_spinner=False)
def get_applicability_domain():
    return load_applicability_domain()


@st.cache_data(ttl="30m", max_entries=8, show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


@st.cache_data(show_spinner=False)
def get_model_comparison():
    return load_model_comparison()


@st.cache_data(show_spinner=False)
def get_model_card():
    return load_model_card()


def initialize_state() -> None:
    st.session_state.setdefault("single_result", None)
    st.session_state.setdefault("batch_result", None)


def active_model(registry: dict | None, model_key: str | None):
    if not registry or not model_key:
        raise ValueError("Model registry is not available.")
    return estimator_from_registry(registry, model_key), model_options(registry)[model_key], model_key


def render_shap_chart(shap_info: dict) -> None:
    if "error" in shap_info:
        st.warning(f"SHAP explanation failed: {shap_info['error']}", icon=":material/warning:")
        return

    importance = shap_info.get("importance")
    if importance is None or importance.empty:
        st.info("No SHAP importances were returned.", icon=":material/info:")
        return

    if shap_info.get("model_used"):
        st.caption(f"Explanation model: {shap_info['model_used']}")

    chart = (
        alt.Chart(importance)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean absolute SHAP value"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".5f"),
            ],
        )
        .properties(height=max(260, min(680, 24 * len(importance))))
    )
    st.altair_chart(chart)


def render_descriptor_grid(features: pd.DataFrame) -> None:
    descriptors = features.loc[:, [col for col in DESCRIPTOR_COLUMNS if col in features.columns]].iloc[0]
    with st.container(horizontal=True):
        st.metric("Molecular weight", descriptors["Molecular_Weight"], border=True, format="%.2f")
        st.metric("AlogP", descriptors["AlogP"], border=True, format="%.2f")
        st.metric("TPSA", descriptors["TPSA"], border=True, format="%.2f")
        st.metric("QED", descriptors["QED"], border=True, format="%.2f")

    descriptor_df = descriptors.rename_axis("descriptor").reset_index(name="value")
    st.dataframe(
        descriptor_df,
        hide_index=True,
        column_config={
            "descriptor": st.column_config.TextColumn("Descriptor", pinned=True),
            "value": st.column_config.NumberColumn("Value", format="%.3f"),
        },
    )


def render_applicability(applicability: dict, flags: dict) -> None:
    with st.container(horizontal=True):
        st.metric(
            "Domain",
            "Inside" if applicability.get("in_domain") else "Outside",
            border=True,
        )
        st.metric(
            "Nearest similarity",
            applicability.get("max_tanimoto_similarity"),
            border=True,
            format="%.3f",
        )
        st.metric(
            "Descriptor distance",
            applicability.get("nearest_descriptor_distance"),
            border=True,
            format="%.3f",
        )
        st.metric("RO5 violations", flags.get("ro5_violations", 0), border=True, format="%d")

    warnings = list(applicability.get("warnings", [])) + list(flags.get("warnings", []))
    if warnings:
        st.warning("; ".join(warnings), icon=":material/warning:")
    else:
        st.success("No major applicability-domain or drug-likeness warnings.", icon=":material/check_circle:")


def render_neighbors(applicability: dict) -> None:
    neighbors = applicability.get("neighbors")
    if not isinstance(neighbors, pd.DataFrame) or neighbors.empty:
        st.info("Nearest-neighbor information is not available.", icon=":material/info:")
        return

    st.dataframe(
        neighbors,
        hide_index=True,
        column_config={
            "Molecule ChEMBL ID": st.column_config.TextColumn("Molecule ChEMBL ID", pinned=True),
            "SMILES": st.column_config.TextColumn("SMILES"),
            "known_pIC50": st.column_config.NumberColumn("Known pIC50", format="%.3f"),
            "tanimoto_similarity": st.column_config.NumberColumn("Tanimoto similarity", format="%.3f"),
            "descriptor_distance": st.column_config.NumberColumn("Descriptor distance", format="%.3f"),
        },
    )


def render_model_panel(model_panel: pd.DataFrame) -> None:
    if model_panel is None or model_panel.empty:
        st.info("Model agreement data is not available.", icon=":material/info:")
        return

    st.dataframe(
        model_panel.sort_values("pIC50", ascending=False),
        hide_index=True,
        column_config={
            "model_key": None,
            "model": st.column_config.TextColumn("Model", pinned=True),
            "pIC50": st.column_config.NumberColumn("Predicted pIC50", format="%.3f"),
            "r2": st.column_config.NumberColumn("Holdout R2", format="%.3f"),
            "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
        },
    )


def render_single_result(result: dict, show_feature_preview: bool) -> None:
    prediction = result["prediction"]
    features = result["features"]
    smiles = result["smiles"]
    shap_info = result["shap_info"]
    applicability = result["applicability"]
    interval = result["interval"]
    flags = result["flags"]
    confidence = result["confidence"]
    model_panel = result["model_panel"]

    st.subheader("Prediction result")
    with st.container(horizontal=True):
        st.metric("Predicted pIC50", prediction, border=True, format="%.3f")
        st.metric("Estimated IC50", pIC50_to_ic50_nm(prediction), border=True, format="%.2f nM")
        st.metric("Activity class", activity_class(prediction), border=True)
        st.metric("Confidence", confidence, border=True)
        st.metric("Priority score", result["priority"], border=True, format="%.1f")

    if pd.notna(interval.get("lower")):
        st.caption(
            f"Approximate prediction interval: {interval['lower']:.3f} to "
            f"{interval['upper']:.3f} pIC50; model spread SD {interval['std']:.3f}."
        )

    left, right = st.columns([0.85, 1.55], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.markdown("**Molecule**")
            try:
                st.image(molecule_image_bytes(smiles), caption=smiles)
            except Exception as exc:
                st.warning(f"Could not draw molecule: {exc}", icon=":material/warning:")

    with right:
        view = st.segmented_control(
            "Result view",
            ["Reliability", "Similar known compounds", "Model agreement", "Interpretability", "Descriptors", "Feature matrix"],
            default="Reliability",
            required=True,
            key="single_result_view",
            width="stretch",
        )
        with st.container(border=True):
            if view == "Reliability":
                render_applicability(applicability, flags)
            elif view == "Similar known compounds":
                render_neighbors(applicability)
            elif view == "Model agreement":
                render_model_panel(model_panel)
            elif view == "Interpretability":
                render_shap_chart(shap_info)
            elif view == "Descriptors":
                render_descriptor_grid(features)
            elif show_feature_preview:
                st.dataframe(features.iloc[:, :60], hide_index=True)
            else:
                st.info("Enable feature preview in the sidebar to inspect the model matrix.", icon=":material/visibility:")

    report_html = generate_report_html(
        smiles,
        prediction,
        features,
        shap_info,
        applicability=applicability,
        model_panel=model_panel,
    )
    st.download_button(
        "Download HTML report",
        data=html_bytesio(report_html),
        file_name="qsar_research_report.html",
        mime="text/html",
        icon=":material/download:",
    )


def read_batch_input(uploaded, csv_url: str) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if csv_url and csv_url.strip():
        return load_csv_from_url(csv_url.strip())
    raise ValueError("Upload a CSV file or paste a public CSV URL.")


def choose_shap_model(registry: dict | None, model_key: str | None, model):
    if registry and model_key == "consensus":
        return estimator_from_registry(registry, registry.get("best_model_key"))
    return model


def run_single_prediction(smiles: str, model, registry: dict | None, model_key: str | None, ad: dict | None, top_k: int, neighbors: int) -> dict:
    canonical = canonicalize_smiles(smiles)
    features = assemble_features_for_model(canonical, model)
    features = align_features_to_model(features, model)
    prediction = predict(model, features)
    applicability = applicability_report(features, ad, top_n=neighbors)
    interval = prediction_interval(model, features)
    flags = property_flags(features)
    model_panel = model_prediction_panel(registry, features)
    confidence = confidence_label(applicability, interval.get("std"))
    shap_info = explain_shap(choose_shap_model(registry, model_key, model), features, top_k=top_k)
    return {
        "smiles": canonical,
        "features": features,
        "prediction": prediction,
        "applicability": applicability,
        "interval": interval,
        "flags": flags,
        "model_panel": model_panel,
        "confidence": confidence,
        "priority": priority_score(prediction, applicability, flags, interval.get("std")),
        "shap_info": shap_info,
    }


def run_batch_predictions(df: pd.DataFrame, model, registry: dict | None, model_key: str | None, ad: dict | None) -> pd.DataFrame:
    if "smiles" not in df.columns:
        raise ValueError("Batch CSV must contain a `smiles` column.")

    rows = []
    progress = st.progress(0, text="Preparing batch predictions")
    smiles_values = df["smiles"].astype(str).tolist()
    total = max(1, len(smiles_values))

    for index, smiles in enumerate(smiles_values, start=1):
        try:
            canonical = canonicalize_smiles(smiles)
            features = assemble_features_for_model(canonical, model)
            prediction = predict(model, features)
            applicability = applicability_report(features, ad, top_n=1)
            interval = prediction_interval(model, features)
            flags = property_flags(features)
            confidence = confidence_label(applicability, interval.get("std"))
            neighbors = applicability.get("neighbors")
            nearest = neighbors.iloc[0].to_dict() if isinstance(neighbors, pd.DataFrame) and not neighbors.empty else {}
            descriptors = features.loc[:, DESCRIPTOR_COLUMNS].iloc[0].to_dict()
            rows.append(
                {
                    "input_smiles": smiles,
                    "canonical_smiles": canonical,
                    "pIC50": prediction,
                    "IC50_nM": pIC50_to_ic50_nm(prediction),
                    "activity_class": activity_class(prediction),
                    "confidence": confidence,
                    "priority_score": priority_score(prediction, applicability, flags, interval.get("std")),
                    "in_domain": applicability.get("in_domain"),
                    "max_tanimoto_similarity": applicability.get("max_tanimoto_similarity"),
                    "nearest_known_pIC50": nearest.get("known_pIC50"),
                    "nearest_chembl_id": nearest.get("Molecule ChEMBL ID"),
                    "prediction_interval_low": interval.get("lower"),
                    "prediction_interval_high": interval.get("upper"),
                    "model_spread_sd": interval.get("std"),
                    "RO5_violations": flags.get("ro5_violations"),
                    "QED": descriptors["QED"],
                    "AlogP": descriptors["AlogP"],
                    "Molecular_Weight": descriptors["Molecular_Weight"],
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "input_smiles": smiles,
                    "canonical_smiles": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
        progress.progress(index / total, text=f"Processed {index} of {total} compounds")

    progress.empty()
    result = pd.DataFrame(rows)
    if "priority_score" in result.columns:
        result = result.sort_values("priority_score", ascending=False, na_position="last")
    return result


def render_batch_result(result: pd.DataFrame) -> None:
    successful = int((result["status"] == "ok").sum())
    failed = int((result["status"] != "ok").sum())
    high_priority = int((result.get("priority_score", pd.Series(dtype=float)) >= 70).sum())

    st.subheader("Screening results")
    with st.container(horizontal=True):
        st.metric("Compounds", len(result), border=True, format="%d")
        st.metric("Predicted", successful, border=True, format="%d")
        st.metric("High priority", high_priority, border=True, format="%d")
        st.metric("Errors", failed, border=True, format="%d", delta_color="inverse")

    st.dataframe(
        result,
        hide_index=True,
        column_config={
            "input_smiles": st.column_config.TextColumn("Input SMILES", pinned=True),
            "canonical_smiles": st.column_config.TextColumn("Canonical SMILES"),
            "pIC50": st.column_config.NumberColumn("pIC50", format="%.3f"),
            "IC50_nM": st.column_config.NumberColumn("IC50 (nM)", format="%.2f"),
            "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.1f"),
            "max_tanimoto_similarity": st.column_config.NumberColumn("Nearest similarity", format="%.3f"),
            "model_spread_sd": st.column_config.NumberColumn("Model spread SD", format="%.3f"),
            "QED": st.column_config.NumberColumn("QED", format="%.2f"),
            "AlogP": st.column_config.NumberColumn("AlogP", format="%.2f"),
            "Molecular_Weight": st.column_config.NumberColumn("Molecular weight", format="%.2f"),
        },
    )

    st.download_button(
        "Download screening CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="qsar_screening_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_model_evidence(registry: dict | None) -> None:
    comparison = get_model_comparison()
    model_card = get_model_card()

    st.subheader("Model evidence")
    if model_card:
        with st.container(horizontal=True):
            st.metric("Training samples", model_card.get("samples"), border=True, format="%d")
            st.metric("Features", model_card.get("features"), border=True, format="%d")
            st.metric("Default model", model_card.get("default_model", "Unavailable"), border=True)
            st.metric("Best single model", model_card.get("best_single_model", "Unavailable"), border=True)

    if not comparison.empty:
        st.dataframe(
            comparison,
            hide_index=True,
            column_config={
                "model_key": None,
                "model": st.column_config.TextColumn("Model", pinned=True),
                "r2": st.column_config.NumberColumn("Holdout R2", format="%.3f"),
                "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "cv_r2_mean": st.column_config.NumberColumn("Sampled CV R2", format="%.3f"),
                "cv_r2_sd": st.column_config.NumberColumn("CV SD", format="%.3f"),
            },
        )

    if registry:
        with st.container(border=True):
            st.markdown("**Registered models**")
            for key, entry in registry.get("models", {}).items():
                st.markdown(f"- **{entry.get('label', key)}**: {entry.get('description', '')}")


initialize_state()
registry = get_registry()
applicability_domain = get_applicability_domain()
options = model_options(registry)

with st.sidebar:
    st.header("Model setup")
    if options:
        default_key = registry.get("default_model_key", next(iter(options)))
        model_key = st.selectbox(
            "Prediction model",
            options=list(options.keys()),
            index=list(options.keys()).index(default_key) if default_key in options else 0,
            format_func=lambda key: options[key],
        )
    else:
        model_key = None
        st.error("Model registry was not found.", icon=":material/error:")

    st.header("Research settings")
    top_k = st.slider("Top SHAP features", min_value=5, max_value=40, value=18, step=1)
    neighbor_count = st.slider("Nearest known compounds", min_value=3, max_value=10, value=5, step=1)
    show_feature_preview = st.toggle("Show feature preview", value=False)

    st.caption(
        "The bundled registry uses "
        f"{DEFAULT_FINGERPRINT_BITS}-bit Morgan fingerprints plus molecular descriptors."
    )


st.title("Anti-leishmania QSAR research tool")
st.caption(
    "Multi-model activity prediction with applicability-domain checks, nearest-neighbor context, "
    "model agreement, and screening exports."
)

mode = st.segmented_control(
    "Workspace",
    ["Single compound", "Batch screening", "Model evidence"],
    default="Single compound",
    required=True,
    key="workspace_mode",
    width="stretch",
)

if mode == "Single compound":
    with st.form("single_prediction_form", border=True):
        smiles_input = st.text_area(
            "SMILES",
            value=EXAMPLE_SMILES,
            height=110,
            placeholder="Paste one canonical or isomeric SMILES string",
        )
        submitted = st.form_submit_button("Analyze compound", type="primary", icon=":material/play_arrow:")

    if submitted:
        try:
            with st.spinner("Generating model, domain, and analog evidence..."):
                model, model_label, active_key = active_model(registry, model_key)
                result = run_single_prediction(
                    smiles_input,
                    model,
                    registry,
                    active_key,
                    applicability_domain,
                    top_k,
                    neighbor_count,
                )
                result["model_label"] = model_label
            st.session_state.single_result = result
            st.toast("Compound analysis complete", icon=":material/check_circle:")
        except Exception as exc:
            st.session_state.single_result = None
            st.error(f"Analysis failed: {exc}", icon=":material/error:")

    if st.session_state.single_result:
        render_single_result(st.session_state.single_result, show_feature_preview)
    else:
        with st.container(border=True):
            render_model_evidence(registry)

elif mode == "Batch screening":
    with st.form("batch_prediction_form", border=True):
        uploaded = st.file_uploader("CSV with a `smiles` column", type=["csv"])
        csv_url = st.text_input("Public CSV URL", placeholder="https://...")
        submitted = st.form_submit_button("Screen compounds", type="primary", icon=":material/play_arrow:")

    if submitted:
        try:
            with st.spinner("Screening compound library..."):
                model, _, active_key = active_model(registry, model_key)
                batch_df = read_batch_input(uploaded, csv_url)
                st.session_state.batch_result = run_batch_predictions(
                    batch_df,
                    model,
                    registry,
                    active_key,
                    applicability_domain,
                )
            st.toast("Batch screening complete", icon=":material/check_circle:")
        except Exception as exc:
            st.session_state.batch_result = None
            st.error(f"Batch screening failed: {exc}", icon=":material/error:")

    if st.session_state.batch_result is not None:
        render_batch_result(st.session_state.batch_result)
    else:
        with st.container(border=True):
            st.markdown("**Screening output includes** pIC50, IC50, activity class, applicability-domain status, nearest training analog, model spread, drug-likeness flags, and a priority score.")

else:
    render_model_evidence(registry)
