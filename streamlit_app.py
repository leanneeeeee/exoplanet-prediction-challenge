import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from model_api import load_model, predict_row
from preprocess import load_and_clean, standardize_dataset
import io

# ---------------------------
# App Configuration + Theme
# ---------------------------
st.set_page_config(page_title="🪐 ExoScan — Exoplanet Classifier", page_icon="🪐", layout="wide")

# Global aesthetic styling (dark cosmic theme)
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
      html, body, [class^="css"]  { font-family: 'Inter', sans-serif; }
      .main { background: radial-gradient(1000px 600px at 10% 10%, #0b132b 0%, #0b0c10 40%, #0b0c10 100%); color:#C5C6C7; }
      h1, h2, h3 { color:#66FCF1 !important; letter-spacing:0.3px; }
      .stTabs [data-baseweb="tab"] { font-weight:600; }
      .stButton>button { background:#45A29E; color:white; border-radius:12px; padding:0.6rem 1rem; border:0; box-shadow:0 4px 20px rgba(69,162,158,0.25); }
      .stButton>button:hover { transform: translateY(-1px); }
      .stDownloadButton>button { background:#1F2833; color:#66FCF1; border-radius:12px; border:1px solid #23313f; }
      .metric-card { padding:14px 16px; border-radius:14px; background:#11161f; border:1px solid #1f2a36; }
      .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#0e1b25; color:#9bd7d1; border:1px solid #1f3b46; }
      .small-muted { color:#9aa4ad; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Robust CSV loader function
# ---------------------------
def robust_read_csv(uploaded_file):
    try:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        lines = text.splitlines()

        header_line = None
        mission = "Unknown"

        for i, line in enumerate(lines):
            lower = line.lower()
            if "koi_period" in lower or "kepid" in lower:
                header_line = i; mission = "Kepler"; break
            elif "epic" in lower:              # ✅ detect EPIC-style K2 rows
                header_line = i
                mission = "K2"
                break
            elif "tic_id" in lower or "tic" in lower:
                header_line = i
                mission = "TESS"
                break
            elif "pl_orbper" in lower:
                header_line = i
                mission = "TESS"  # fallback only if nothing else matched
                break
            elif "column" in lower and "period" in lower:
                header_line = i + 1; break

        if header_line is None:
            raise ValueError("Could not find recognizable header for Kepler, K2, or TESS")

        df = pd.read_csv(io.StringIO("\n".join(lines[header_line:])), sep=",", engine="python", on_bad_lines="skip", comment="#")
        st.success(f"✅ CSV loaded successfully! Mission: {mission}")
        return df, mission

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), "Unknown"

# ---------------------------
# App Title & Model Setup
# ---------------------------
st.title("🪐 ExoScan — Exoplanet Classifier")
st.caption("Explore exoplanets across Kepler, K2, and TESS missions — or upload your own dataset to retrain and predict.")

# ---------------------------
# Caching and Lazy Loading
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_cached_model(model_name: str = "xgb"):
    """Load ML model once and cache it."""
    with st.spinner("🚀 Loading model..."):
        model = api_load_model(model_name)
    return model

@st.cache_data(show_spinner=False)
def load_cached_dataset(path: str = "data/merged.csv"):
    """Load and preprocess dataset once and cache it."""
    with st.spinner("🧠 Loading and preprocessing dataset..."):
        df, FEATURES = load_and_clean(path)
    return df, FEATURES

st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.selectbox("Choose model:", ["XGBoost", "Random Forest", "Stacking"], help="Switch between trained model variants")
model = load_model("xgb" if model_type == "XGBoost" else "rf" if model_type == "Random Forest" else "stack")

_, FEATURES = load_and_clean("data/merged.csv")

# 2️⃣ Initialize session storage
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# 3️⃣ Load built-in NASA datasets BEFORE the UI
BUILT_IN_DATASETS = {
    "Kepler (NASA)": "data/kepler.csv",
    "K2 (NASA)": "data/k2.csv",
    "TESS (NASA)": "data/tess.csv"
}

for name, path in BUILT_IN_DATASETS.items():
    if name not in st.session_state.datasets:
        try:
            df, mission = load_and_clean(path)
            st.session_state.datasets[name] = {"df": df, "mission": mission}
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {e}")

# Sidebar quick guide
with st.sidebar.container(border=True):
    st.markdown("#### 🛰️ Quick Guide")
    st.markdown("- Pick a dataset (built-in or upload)\n- Click **Predict**\n- (Optional) enable **Explainability** to view SHAP")
    st.markdown("<span class='small-muted'>Tip: SHAP is compute-heavy; toggle it only when needed.</span>", unsafe_allow_html=True)

# ---------------------------
# Load built-in NASA datasets
# ---------------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

BUILT_IN_DATASETS = {
    "Kepler (NASA)": "data/kepler.csv",
    "K2 (NASA)": "data/k2.csv",
    "TESS (NASA)": "data/tess.csv"
}

for name, path in BUILT_IN_DATASETS.items():
    if name not in st.session_state.datasets:
        try:
            df, mission = load_and_clean(path)
            st.session_state.datasets[name] = {"df": df, "mission": mission}
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {e}")

# ---------------------------
# Tabs for main workflows
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "🔁 Retrain", "🧮 Manual Input"])

# ---------------------------
# PREDICT TAB
# ---------------------------
with tab1:
    st.subheader("📤 Select or Upload Dataset")

    dataset_choice = st.selectbox(
        "Select a dataset:",
        list(st.session_state.datasets.keys()) + ["Upload new dataset"],
    )

    upload = None
    df_new, mission = pd.DataFrame(), "Unknown"

    if dataset_choice == "Upload new dataset":
        upload = st.file_uploader("Upload CSV file", type=["csv"], help="NASA CSVs or your own")

    if dataset_choice != "Upload new dataset":
        df_new = st.session_state.datasets[dataset_choice]["df"]
        mission = st.session_state.datasets[dataset_choice]["mission"]
        st.success(f"📦 Loaded dataset: {dataset_choice}")
    elif upload:
        df_new, mission = robust_read_csv(upload)
        if not df_new.empty:
            dataset_name = upload.name
            st.session_state.datasets[dataset_name] = {"df": df_new, "mission": mission}
            st.success(f"📦 Stored dataset: {dataset_name}")

    if not df_new.empty:
        # Top metrics row
        c1, c2, c3, c4 = st.columns([1,1,1,2])
        with c1: st.markdown(f"<div class='metric-card'><div class='small-muted'>Mission</div><h3>{mission}</h3></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><div class='small-muted'>Rows</div><h3>{len(df_new):,}</h3></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><div class='small-muted'>Columns</div><h3>{len(df_new.columns)}</h3></div>", unsafe_allow_html=True)
        with c4: st.markdown("<div class='metric-card'><div class='small-muted'>Status</div><span class='pill'>Ready for prediction</span></div>", unsafe_allow_html=True)

        with st.expander("👀 Preview dataset (first 8 rows)"):
            st.dataframe(df_new.head(8), use_container_width=True)

        df_new, mission = standardize_dataset(df_new, FEATURES)
        missing = [f for f in FEATURES if f not in df_new.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # Optional explainability toggle
            enable_shap = st.toggle("🧠 Enable Explainability (SHAP)", value=False, help="Compute global & local explanations")
            if st.button("🔮 Predict Dataset", use_container_width=True):
                try:
                    '''FEATURES = model.get_booster().feature_names
                    preds, probs = predict_row(model, df_new, FEATURES)'''
                    # ✅ Automatically handle feature names for any model type
                    try:
                        if hasattr(model, "get_booster"):
                            FEATURES = model.get_booster().feature_names
                        elif hasattr(model, "feature_names_in_"):
                            FEATURES = list(model.feature_names_in_)
                        else:
                            # Fallback: use the pre-defined feature list
                            from preprocess import KEEP_FEATURES
                            FEATURES = KEEP_FEATURES
                    except Exception:
                        from preprocess import KEEP_FEATURES
                        FEATURES = KEEP_FEATURES

                    preds, probs = predict_row(model, df_new, FEATURES)

                    df_new["pred_label"] = preds

                    # Label mapping for accuracy
                    if "label" not in df_new.columns:
                        if "koi_disposition" in df_new.columns:
                            mapping = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
                            df_new["label"] = df_new["koi_disposition"].map(lambda x: mapping.get(str(x).upper().strip(), np.nan))
                        elif "disposition" in df_new.columns:
                            mapping = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
                            df_new["label"] = df_new["disposition"].map(lambda x: mapping.get(str(x).upper().strip(), np.nan))
                        elif "tfopwg_disp" in df_new.columns:
                            mapping = {"FP": 0, "PC": 1, "CP": 2, "KP": 2}
                            df_new["label"] = df_new["tfopwg_disp"].map(lambda x: mapping.get(str(x).upper().strip(), np.nan))

                    if "label" in df_new.columns:
                        df_new = df_new.dropna(subset=["label"])
                        df_new["label"] = df_new["label"].astype(int)

                    from sklearn.metrics import accuracy_score, classification_report
                    st.subheader("📊 Prediction Results")
                    st.dataframe(df_new[["pred_label"] + FEATURES], use_container_width=True)

                    # KPI row
                    if "label" in df_new.columns and not df_new["label"].isna().all():
                        y_true = df_new["label"].astype(int)
                        y_pred = df_new["pred_label"].astype(int)
                        acc = accuracy_score(y_true, y_pred)
                        k1, k2 = st.columns(2)
                        with k1: st.metric("Accuracy", f"{acc*100:.2f}%")
                        with k2: st.bar_chart(df_new['pred_label'].value_counts(), use_container_width=True)
                        report = classification_report(y_true, y_pred, output_dict=False)
                        with st.expander("Detailed classification report"):
                            st.text(report)
                    else:
                        st.info("No ground-truth labels found; accuracy not computed.")

                    # SHAP Explainability (optional)
                    if enable_shap:
                        with st.spinner("Computing SHAP values..."):
                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(df_new[FEATURES])

                                with st.expander("🌍 Global Feature Importance", expanded=True):
                                    fig1, ax1 = plt.subplots()
                                    shap.summary_plot(shap_values, df_new[FEATURES], show=False)
                                    st.pyplot(fig1, use_container_width=True)

                                with st.expander("🔬 Explain a Single Prediction"):
                                    idx = st.number_input("Sample index", 0, len(df_new)-1, 0)
                                    fig2, ax2 = plt.subplots()
                                    shap.waterfall_plot(shap.Explanation(values=shap_values[idx], base_values=getattr(explainer, 'expected_value', 0), data=df_new.iloc[idx]))
                                    st.pyplot(fig2, use_container_width=True)
                            except Exception as e:
                                st.warning(f"SHAP visualization unavailable: {e}")

                    # Download
                    csv = df_new.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predicted Results", csv, file_name=f"predicted_{mission.lower()}.csv", use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------------------------
# RETRAIN TAB
# ---------------------------
with tab2:
    st.subheader("🔁 Retrain Model")

    if len(st.session_state.datasets) == 0:
        st.info("Upload or use built-in datasets from the Predict tab to retrain the model.")
    else:
        retrain_datasets = st.multiselect(
            "Select one or more datasets to retrain the model:",
            list(st.session_state.datasets.keys())
        )

        # Training controls
        colA, colB = st.columns([1,1])
        with colA:
            do_shuffle = st.checkbox("Shuffle data", value=True)
        with colB:
            st.caption("Retraining uses standardized features only.")

        if st.button("⚙️ Retrain Model", use_container_width=True):
            if len(retrain_datasets) == 0:
                st.warning("Please select at least one dataset.")
            else:
                try:
                    X_total, y_total = [], []
                    progress = st.progress(0)
                    for i, name in enumerate(retrain_datasets):
                        df_data = st.session_state.datasets[name]["df"]
                        df_std, _ = standardize_dataset(df_data, FEATURES)
                        if "label" in df_std.columns:
                            X_total.append(df_std[FEATURES])
                            y_total.append(df_std["label"])
                            st.success(f"✅ Included labeled dataset: {name}")
                        else:
                            X_total.append(df_std[FEATURES])
                            st.info(f"ℹ️ Included unlabeled dataset: {name}")
                        progress.progress(int((i+1)/max(len(retrain_datasets),1)*100))

                    if not X_total:
                        st.error("No valid datasets found.")
                    else:
                        X_all = pd.concat(X_total)
                        if do_shuffle:
                            X_all = X_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
                        y_all = pd.concat(y_total) if y_total else np.zeros(len(X_all))

                        try:
                            with st.spinner("⏳ Retraining model..."):
                                # incremental_update may come from your training utilities
                                model = incremental_update(model, X_all, y_all)  # type: ignore
                                joblib.dump(model, f"models/{model_type.lower()}_model.joblib")
                            st.toast("Model retrained and saved.")
                            st.success("🎉 Model retrained and saved successfully!")
                        except NameError:
                            st.error("`incremental_update` is not defined. Please import or implement it in your training pipeline.")

                except Exception as e:
                    st.error(f"Retraining failed: {e}")

# ---------------------------
# MANUAL INPUT TAB
# ---------------------------
with tab3:
    st.subheader("🧮 Manual Input Prediction")
    cols = st.columns(2)
    vals = {}

    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            safe_key = f"num_input_{i}_{f.replace(' ', '_').replace('.', '_')}"
            vals[f] = st.number_input(
                f,
                value=0.0,
                help=f"Enter value for {f}",
                key=safe_key   # ✅ guaranteed unique and sanitized
            )

    if st.button("🔮 Predict Manual Input", use_container_width=True):
        try:
            row = pd.DataFrame([vals])
            pred, proba = predict_row(model, row, FEATURES)
            st.success(f"Predicted Class: {pred[0]}")
            st.bar_chart(
                pd.DataFrame(proba[0],
                index=["Class 0", "Class 1", "Class 2"],
                columns=["Probability"]),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Manual prediction failed: {e}")


# ---------------------------
# SIDEBAR ACTIONS (Dataset Summary + Clear)
# ---------------------------
with st.sidebar.container(border=True):
    st.markdown("#### 📂 Dataset Summary")
    try:
        sel = dataset_choice if 'dataset_choice' in locals() else None
        if sel and sel in st.session_state.datasets:
            dsel = st.session_state.datasets[sel]["df"]
            mis = st.session_state.datasets[sel]["mission"]
            st.markdown(f"**Selected:** {sel}")
            st.caption(f"Mission: {mis} | Rows: {len(dsel):,} | Cols: {len(dsel.columns)}")
        else:
            st.caption("No dataset selected yet")
    except Exception:
        st.caption("No dataset selected yet")

    if st.button("🗑️ Clear all stored datasets", use_container_width=True):
        st.session_state.datasets.clear()
        st.success("All stored datasets cleared.")