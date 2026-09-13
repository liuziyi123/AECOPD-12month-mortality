from pathlib import Path
import hashlib
import warnings

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# App configuration
# ============================================================
st.set_page_config(
    page_title="AECOPD 12-Month Mortality Risk Prediction",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "AECOPD_SVM_FINAL.joblib"
EXPECTED_MODEL_NAME = "SVM"
EXPECTED_FEATURES = [
    "无创通气",
    "雾化激素",
    "静脉激素",
    "肺结核",
    "年龄",
    "住院天数",
    "入院时_mMRC分级",
]
EXPECTED_MODEL_SHA256 = "fa1788a7a528ff65df1ef277cdaa0ba1cbb919e18e08d87fc762e4013690baad"

FEATURE_LABELS = {
    "无创通气": "Non-invasive ventilation",
    "雾化激素": "Nebulized corticosteroid use",
    "静脉激素": "Intravenous corticosteroid use",
    "肺结核": "PTB",
    "年龄": "Age (years)",
    "住院天数": "Length of hospital stay (days)",
    "入院时_mMRC分级": "mMRC grade on admission",
}


# ============================================================
# Model loading and integrity checks
# ============================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_resource(show_spinner="Loading the locked SVM model...")
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep AECOPD_SVM_FINAL.joblib in the same folder as streamlit_app.py."
        )

    actual_hash = sha256_file(MODEL_PATH)
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Model integrity check failed. The deployed model file is not the "
            "same locked SVM file used to build this app."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError("Unexpected model file structure: expected a dictionary bundle.")

    required_keys = {"name", "final_model", "training_threshold"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise KeyError(f"Model file is missing required keys: {sorted(missing)}")

    if str(bundle["name"]) != EXPECTED_MODEL_NAME:
        raise RuntimeError(
            f"Unexpected selected model: {bundle['name']!r}; expected {EXPECTED_MODEL_NAME!r}."
        )

    model = bundle["final_model"]
    feature_names = list(getattr(model, "feature_names_in_", []))
    if feature_names != EXPECTED_FEATURES:
        raise RuntimeError(
            "Feature-name/order check failed. "
            f"Expected {EXPECTED_FEATURES}, got {feature_names}."
        )

    threshold = float(bundle["training_threshold"])
    return model, threshold, actual_hash


try:
    MODEL, THRESHOLD, MODEL_HASH = load_model_bundle()
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.exception(exc)
    st.stop()


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .main .block-container {max-width: 860px; padding-top: 2rem; padding-bottom: 3rem;}
    .app-subtitle {font-size: 1.02rem; color: #555; margin-top: -0.55rem; margin-bottom: 1.25rem;}
    .risk-card {border: 1px solid #e7e7e7; border-radius: 14px; padding: 1.1rem 1.2rem; margin: .6rem 0 1rem 0;}
    .risk-number {font-size: 2.35rem; font-weight: 750; line-height: 1.05;}
    .small-note {font-size: .88rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================
st.title("AECOPD 12-Month Mortality Risk Prediction")
st.markdown(
    '<div class="app-subtitle">An explainable machine-learning clinical decision-support calculator for patients hospitalized with acute exacerbation of COPD.</div>',
    unsafe_allow_html=True,
)

with st.expander("About this calculator", expanded=False):
    st.write(
        "This web application deploys the locked, training-fitted and probability-calibrated "
        "support vector machine (SVM) model. It uses seven routinely available clinical "
        "predictors to estimate the probability of all-cause mortality within 12 months "
        "after discharge. The model is not re-fitted, re-tuned, or recalibrated in this app."
    )
    st.caption(
        "The classification threshold shown below was derived from cross-fitted training "
        "predictions using the Youden index. It should not be interpreted as a treatment threshold."
    )


# ============================================================
# Input form
# ============================================================
st.subheader("Patient information")
st.caption("Enter information from the index AECOPD hospitalization.")

with st.form("prediction_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age (years)", min_value=18, max_value=120, value=70, step=1
        )
        los = st.number_input(
            "Length of hospital stay (days)", min_value=1, max_value=365, value=10, step=1
        )
        mmrc = st.selectbox(
            "mMRC grade on admission", options=[0, 1, 2, 3, 4], index=2
        )
        tuberculosis = st.selectbox(
            "PTB", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c2:
        niv = st.selectbox(
            "Non-invasive ventilation", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        nebulized_cs = st.selectbox(
            "Nebulized corticosteroid use", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        iv_cs = st.selectbox(
            "Intravenous corticosteroid use", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button("Calculate 12-month mortality risk", type="primary", use_container_width=True)


# ============================================================
# Prediction
# ============================================================
if submitted:
    patient = pd.DataFrame(
        [{
            "无创通气": float(niv),
            "雾化激素": float(nebulized_cs),
            "静脉激素": float(iv_cs),
            "肺结核": float(tuberculosis),
            "年龄": float(age),
            "住院天数": float(los),
            "入院时_mMRC分级": float(mmrc),
        }],
        columns=EXPECTED_FEATURES,
    )

    try:
        probability = float(MODEL.predict_proba(patient)[:, 1][0])
    except Exception as exc:
        st.error("Prediction failed. Please check the deployment environment and model dependencies.")
        st.exception(exc)
        st.stop()

    probability = max(0.0, min(1.0, probability))
    above_threshold = probability >= THRESHOLD

    st.divider()
    st.subheader("Prediction result")

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="small-note">Estimated probability of 12-month all-cause mortality after discharge</div>
            <div class="risk-number">{probability * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(probability)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Training-derived threshold", f"{THRESHOLD * 100:.1f}%")
    with m2:
        st.metric(
            "Model classification",
            "At/above threshold" if above_threshold else "Below threshold",
        )

    if above_threshold:
        st.info(
            "The predicted probability is at or above the model's training-derived classification threshold. "
            "This threshold is a model operating point, not an indication for any specific treatment."
        )
    else:
        st.info(
            "The predicted probability is below the model's training-derived classification threshold. "
            "Clinical follow-up should still be based on the patient's complete clinical assessment."
        )

    st.subheader("Input summary")
    summary_rows = [
        (FEATURE_LABELS["年龄"], f"{int(age)}"),
        (FEATURE_LABELS["住院天数"], f"{int(los)}"),
        (FEATURE_LABELS["入院时_mMRC分级"], f"{int(mmrc)}"),
        (FEATURE_LABELS["无创通气"], "Yes" if niv else "No"),
        (FEATURE_LABELS["雾化激素"], "Yes" if nebulized_cs else "No"),
        (FEATURE_LABELS["静脉激素"], "Yes" if iv_cs else "No"),
        (FEATURE_LABELS["肺结核"], "Yes" if tuberculosis else "No"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Predictor", "Value"])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ============================================================
# Disclaimer / technical information
# ============================================================
st.divider()
st.subheader("Important information")
st.warning(
    "For research and clinical decision-support use only. This calculator does not establish a diagnosis, "
    "does not prescribe treatment, and should not replace professional clinical judgment. The model was "
    "developed for the study population and endpoint described in the accompanying manuscript; prospective "
    "clinical-impact validation is required before routine clinical use."
)
st.caption(
    "This app does not intentionally write submitted patient values to local files or a database. "
    "Do not enter directly identifying information."
)

with st.expander("Technical model information", expanded=False):
    st.write(f"Selected model: **{EXPECTED_MODEL_NAME}**")
    st.write(f"Training-derived classification threshold: **{THRESHOLD:.6f}**")
    st.write("Expected predictors (model order):")
    for i, feature in enumerate(EXPECTED_FEATURES, 1):
        st.write(f"{i}. {FEATURE_LABELS[feature]}")
    st.caption(f"Locked model SHA-256: {MODEL_HASH}")

