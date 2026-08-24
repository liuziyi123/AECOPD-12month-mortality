# AECOPD 12-Month Mortality Risk Prediction — Streamlit deployment

This repository deploys the locked, probability-calibrated SVM model used for the AECOPD 12-month post-discharge mortality study.

## Files

- `streamlit_app.py` — Streamlit web application.
- `AECOPD_SVM_FINAL.joblib` — locked SVM model bundle; do not retrain or replace unless the manuscript analysis is updated.
- `requirements.txt` — deployment dependencies.
- `.streamlit/config.toml` — minimal Streamlit configuration.

## Critical deployment setting

Use **Python 3.12** in Streamlit Community Cloud Advanced settings. The model was serialized with **scikit-learn 1.9.0**, which is therefore pinned in `requirements.txt`.

## Model identity

- Model: SVM
- Training-derived Youden threshold: `0.09638321770979928`
- Expected predictors: 7
- Model SHA-256: `fa1788a7a528ff65df1ef277cdaa0ba1cbb919e18e08d87fc762e4013690baad`

## Local run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Manuscript note

The web application performs prediction only. It loads the locked, training-fitted calibrated model and does not re-fit, re-tune, recalibrate, or reselect the classification threshold.
