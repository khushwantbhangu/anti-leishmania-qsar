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
    align_features_to_model,
    assemble_features_for_model,
    explain_shap,
    generate_report_html,
    html_bytesio,
    load_model,
    molecule_image_bytes,
    pIC50_to_ic50_nm,
    predict,
)


DEFAULT_MODEL_PATH = ROOT / "results" / "model.joblib"
EXAMPLE_SMILES = "CCOC(=O)C1=CC=CC=C1"


st.set_page_config(
    page_title="Anti-leishmania QSAR predictor",
    page_icon=":material/science:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_model_from_path(model_path: str):
    return load_model(model_path)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_model_from_bytes(model_bytes: bytes):
    return load_model(uploaded_model=io.BytesIO(model_bytes))


@st.cache_data(ttl="15m", max_entries=8, show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def initialize_state() -> None:
    st.session_state.setdefault("single_result", None)
    st.session_state.setdefault("batch_result", None)


def active_model(model_path: str, model_file):
    if model_file is not None:
        return get_model_from_bytes(model_file.getvalue()), model_file.name
    return get_model_from_path(model_path), model_path


def render_descriptor_grid(features: pd.DataFrame) -> None:
    descriptors = features.loc[:, [col for col in DESCRIPTOR_COLUMNS if col in features.columns]].iloc[0]
    metric_row = st.container(horizontal=True)
    with metric_row:
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


def render_shap_chart(shap_info: dict) -> None:
    if "error" in shap_info:
        st.warning(f"SHAP explanation failed: {shap_info['error']}", icon=":material/warning:")
        return

    importance = shap_info.get("importance")
    if importance is None or importance.empty:
        st.info("No SHAP importances were returned for this prediction.", icon=":material/info:")
        return

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


def render_single_result(result: dict, show_feature_preview: bool) -> None:
    prediction = result["prediction"]
    features = result["features"]
    smiles = result["smiles"]
    shap_info = result["shap_info"]

    st.subheader("Prediction result")
    with st.container(horizontal=True):
        st.metric("Predicted pIC50", prediction, border=True, format="%.3f")
        st.metric("Estimated IC50", pIC50_to_ic50_nm(prediction), border=True, format="%.2f nM")
        st.metric("Model inputs", features.shape[1], border=True, format="%d")

    left, right = st.columns([0.9, 1.4], vertical_alignment="top")
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
            ["Interpretability", "Descriptors", "Feature matrix"],
            default="Interpretability",
            required=True,
            key="single_result_view",
            width="stretch",
        )
        with st.container(border=True):
            if view == "Interpretability":
                st.markdown("**Top molecular drivers**")
                render_shap_chart(shap_info)
            elif view == "Descriptors":
                st.markdown("**Computed descriptors**")
                render_descriptor_grid(features)
            elif show_feature_preview:
                st.markdown("**Model feature preview**")
                st.dataframe(features.iloc[:, :40], hide_index=True)
            else:
                st.info(
                    "Turn on feature preview in the sidebar to inspect the model matrix.",
                    icon=":material/visibility:",
                )

    report_html = generate_report_html(smiles, prediction, features, shap_info)
    st.download_button(
        "Download HTML report",
        data=html_bytesio(report_html),
        file_name="qsar_prediction_report.html",
        mime="text/html",
        icon=":material/download:",
    )


def read_batch_input(uploaded, csv_url: str) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if csv_url and csv_url.strip():
        return load_csv_from_url(csv_url.strip())
    raise ValueError("Upload a CSV file or paste a public CSV URL.")


def run_batch_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    if "smiles" not in df.columns:
        raise ValueError("Batch CSV must contain a `smiles` column.")

    rows = []
    progress = st.progress(0, text="Preparing batch predictions")
    smiles_values = df["smiles"].astype(str).tolist()
    total = max(1, len(smiles_values))

    for index, smiles in enumerate(smiles_values, start=1):
        clean = smiles.strip()
        try:
            features = assemble_features_for_model(clean, model)
            prediction = predict(model, features)
            descriptors = features.loc[:, DESCRIPTOR_COLUMNS].iloc[0].to_dict()
            rows.append(
                {
                    "smiles": clean,
                    "pIC50": prediction,
                    "IC50_nM": pIC50_to_ic50_nm(prediction),
                    "status": "ok",
                    "error": "",
                    "Molecular_Weight": descriptors["Molecular_Weight"],
                    "AlogP": descriptors["AlogP"],
                    "QED": descriptors["QED"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "smiles": clean,
                    "pIC50": None,
                    "IC50_nM": None,
                    "status": "error",
                    "error": str(exc),
                    "Molecular_Weight": None,
                    "AlogP": None,
                    "QED": None,
                }
            )
        progress.progress(index / total, text=f"Processed {index} of {total} compounds")

    progress.empty()
    return pd.DataFrame(rows)


def render_batch_result(result: pd.DataFrame) -> None:
    successful = int((result["status"] == "ok").sum())
    failed = int((result["status"] != "ok").sum())

    st.subheader("Batch results")
    with st.container(horizontal=True):
        st.metric("Compounds", len(result), border=True, format="%d")
        st.metric("Predicted", successful, border=True, format="%d")
        st.metric("Errors", failed, border=True, format="%d", delta_color="inverse")

    st.dataframe(
        result,
        hide_index=True,
        column_config={
            "smiles": st.column_config.TextColumn("SMILES", pinned=True),
            "pIC50": st.column_config.NumberColumn("pIC50", format="%.3f"),
            "IC50_nM": st.column_config.NumberColumn("IC50 (nM)", format="%.2f"),
            "Molecular_Weight": st.column_config.NumberColumn("Molecular weight", format="%.2f"),
            "AlogP": st.column_config.NumberColumn("AlogP", format="%.2f"),
            "QED": st.column_config.NumberColumn("QED", format="%.2f"),
            "status": st.column_config.TextColumn("Status"),
            "error": st.column_config.TextColumn("Error"),
        },
    )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction CSV",
        data=csv_bytes,
        file_name="qsar_batch_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )


initialize_state()

with st.sidebar:
    st.header("Model setup")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    model_file = st.file_uploader("Upload trained model", type=["joblib", "pkl"])

    st.header("Interpretability")
    top_k = st.slider("Top SHAP features", min_value=5, max_value=40, value=18, step=1)
    show_feature_preview = st.toggle("Show feature preview", value=False)

    st.caption(
        "The app infers fingerprint size from the model schema. The bundled model "
        f"uses {DEFAULT_FINGERPRINT_BITS}-bit Morgan fingerprints plus molecular descriptors."
    )


st.title("Anti-leishmania QSAR predictor")
st.caption(
    "Predict anti-leishmanial activity from SMILES with descriptor checks, model-schema "
    "alignment, SHAP interpretation, and exportable reports."
)

mode = st.segmented_control(
    "Prediction mode",
    ["Single compound", "Batch CSV"],
    default="Single compound",
    required=True,
    key="prediction_mode",
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
        submitted = st.form_submit_button(
            "Predict activity",
            type="primary",
            icon=":material/play_arrow:",
        )

    if submitted:
        try:
            with st.spinner("Loading model and preparing molecular features..."):
                model, model_label = active_model(model_path, model_file)
                features = assemble_features_for_model(smiles_input, model)
                features = align_features_to_model(features, model)
                prediction = predict(model, features)
                shap_info = explain_shap(model, features, top_k=top_k)
            st.session_state.single_result = {
                "smiles": smiles_input.strip(),
                "prediction": prediction,
                "features": features,
                "shap_info": shap_info,
                "model_label": model_label,
            }
            st.toast("Prediction complete", icon=":material/check_circle:")
        except Exception as exc:
            st.session_state.single_result = None
            st.error(f"Prediction failed: {exc}", icon=":material/error:")

    if st.session_state.single_result:
        render_single_result(st.session_state.single_result, show_feature_preview)
    else:
        with st.container(border=True):
            st.markdown("**Ready when you are**")
            st.write(
                "Enter a molecule and submit the form to compute descriptors, align features "
                "to the trained model, and predict pIC50."
            )

else:
    with st.form("batch_prediction_form", border=True):
        uploaded = st.file_uploader("CSV with a `smiles` column", type=["csv"])
        csv_url = st.text_input("Public CSV URL", placeholder="https://...")
        submitted = st.form_submit_button(
            "Run batch prediction",
            type="primary",
            icon=":material/play_arrow:",
        )

    if submitted:
        try:
            with st.spinner("Loading model and reading batch input..."):
                model, _ = active_model(model_path, model_file)
                batch_df = read_batch_input(uploaded, csv_url)
            st.session_state.batch_result = run_batch_predictions(batch_df, model)
            st.toast("Batch prediction complete", icon=":material/check_circle:")
        except Exception as exc:
            st.session_state.batch_result = None
            st.error(f"Batch prediction failed: {exc}", icon=":material/error:")

    if st.session_state.batch_result is not None:
        render_batch_result(st.session_state.batch_result)
    else:
        with st.container(border=True):
            st.markdown("**Batch input**")
            st.write(
                "Upload a CSV or provide a public CSV URL. The file must include a `smiles` "
                "column; additional columns are ignored during prediction."
            )
