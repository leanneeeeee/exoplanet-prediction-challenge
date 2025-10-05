import streamlit as st
import pandas as pd
import numpy as np
import joblib

from model_api import load_model, predict_row
from preprocess import load_and_clean, standardize_dataset
import io
import os

# ---------------------------
# App Configuration + Theme
# ---------------------------
st.set_page_config(page_title="🪐 Exoplanet Prediction Challenge", page_icon="🪐", layout="wide")

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
st.title("🪐 Exoplanet Prediction Challenge")
st.caption("Explore, predict, and play with exoplanet data from NASA missions — Kepler, K2, and TESS.")

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
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "🎮 Guess Planet Type", "🧮 Manual Input"])

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
        
        #@TEST
        if "row_id" not in df_new.columns:
            df_new["row_id"] = df_new.index
        #@ENDTEST

        df_new, mission = standardize_dataset(df_new, FEATURES)
        missing = [f for f in FEATURES if f not in df_new.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # Optional explainability toggle
            if st.button("🔮 Predict Dataset", use_container_width=True):
                try:
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

                    

                    # Download
                    csv = df_new.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predicted Results", csv, file_name=f"predicted_{mission.lower()}.csv", use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")




# ---------------------------
# 🎮 Exoplanet Type Guessing Game (NEW TAB)
# ---------------------------
with tab2:
    st.subheader("🪐 Exoplanet Type Guessing Game\n")


    # --- Use session state to persist predictions across reruns ---
    if "predicted_df" not in st.session_state:
        st.session_state.predicted_df = pd.DataFrame()

    # Save predictions to session state after running Predict Dataset
    if not df_new.empty and "pred_label" in df_new.columns:
        st.session_state.predicted_df = df_new.copy()

    df_game = st.session_state.predicted_df  # restore predictions

    if df_game.empty:
        st.info("No predictions available yet. Please run 🔮 Predict Dataset first.")
    else:
        # --- Add planet type based on radius rules ---
        def classify_planet_type(radius):
            if radius > 6:
                return "Gas Giant"
            elif 3 < radius <= 6:
                return "Neptunian"
            elif 1.5 < radius <= 3:
                return "Super-Earth"
            else:
                return "Terrestrial"

        df_game["planet_type"] = df_game["koi_prad"].apply(classify_planet_type)

        
        with st.expander("🛈 What do these columns mean?", expanded=True):
            st.markdown("""
            - **koi_prad** → Planet radius in Earth radii  
            - **koi_period** → Orbital period in days
            - **koi_teq** → Planet equilibrium temperature in Kelvin  
            """)

        # --- Select top 5 confirmed / high-probability candidates ---
        df_game["prob_confirmed"] = df_game.get("pred_prob_confirmed", 0.0)
        top5 = df_game[df_game["pred_label"].isin([1,2])].nlargest(5, "prob_confirmed")

        if top5.empty:
            st.warning("No confirmed or high-probability candidates available to play.")
        else:
            # 💡 Hover-only Tips button (place above "Top 5 Most Likely Planets")
            st.markdown("""
            <div class="tips-wrap">
            <span class="tips-btn">💡 Tips</span>
            <div class="tips-pop">
                <div class="tips-title">Radius → Type</div>
                <table class="tips-table">
                <thead><tr><th>Radius (⨁)</th><th>Planet Type</th></tr></thead>
                <tbody>
                    <tr><td>≤ 1.5</td><td>Terrestrial</td></tr>
                    <tr><td>1.5 – 3</td><td>Super-Earth</td></tr>
                    <tr><td>3 – 6</td><td>Neptunian</td></tr>
                    <tr><td>&gt; 6</td><td>Gas Giant</td></tr>
                </tbody>
                </table>
            </div>
            </div>
            <style>
            .tips-wrap{position:relative;display:flex;justify-content:flex-end;margin:6px 0 4px;}
            .tips-btn{background:#0e1722;border:1px solid #1d2a36;color:#9bd7d1;
                        padding:4px 10px;border-radius:999px;font-size:0.9rem;cursor:default;}
            .tips-pop{position:absolute;top:36px;right:0;background:#0f1623;border:1px solid #1f2a36;
                        border-radius:12px;padding:10px 12px;width:280px;box-shadow:0 8px 30px rgba(0,0,0,.4);
                        opacity:0;visibility:hidden;pointer-events:none;transform:translateY(-4px);
                        transition:opacity .15s ease, transform .15s ease, visibility .15s ease; z-index:9999;}
            .tips-wrap:hover .tips-pop{opacity:1;visibility:visible;pointer-events:auto;transform:translateY(0);}
            .tips-title{color:#9bd7d1;font-weight:600;margin-bottom:6px;}
            .tips-table{width:100%;border-collapse:collapse;font-size:.9rem;}
            .tips-table th,.tips-table td{border-bottom:1px solid #223041;padding:6px 8px;text-align:center;}
            </style>
            """, unsafe_allow_html=True)

            st.markdown("**Top 5 Most Likely Planets:** (use Row_ID to pick)")
            display_cols = ["koi_prad", "koi_period", "koi_teq"]
            st.dataframe(top5[display_cols], use_container_width=True)

            # --- User selects planet by original row_id ---
            chosen_rowid = st.selectbox("Pick a planet to guess (Row ID)", top5["row_id"].tolist())
            guessed_type = st.selectbox(
                "Your guess for planet type",
                ["Terrestrial", "Super-Earth", "Neptunian", "Gas Giant"]
            )

            # Reveal (local image + italic fun fact)
            if st.button("✅ Check Guess", use_container_width=True):
                # --- safely fetch the selected planet row ---
                src = playable if 'playable' in locals() else st.session_state.get("predicted_df", pd.DataFrame())
                if src is None or src.empty:
                    st.error("No planet data available. Run Predict first.")
                else:
                    if "row_id" not in src.columns:
                        src = src.reset_index(drop=False).rename(columns={"index": "row_id"})
                    rid = st.session_state.get("game_selected_rowid")
                    sel = src[src["row_id"] == rid]
                    if sel.empty:
                        sel = src.iloc[[0]]
                    planet = sel.iloc[0]

                    # --- tiny local classifier (avoids relying on outer scope) ---
                    def _classify(radius):
                        try:
                            r = float(radius)
                        except Exception:
                            return "Unknown"
                        return ("Gas Giant" if r > 6 else
                                "Neptunian" if r > 3 else
                                "Super-Earth" if r > 1.5 else
                                "Terrestrial")

                    actual_type = _classify(planet.get("koi_prad", np.nan))
                    st.markdown(f"**Your Guess:** {guessed_type or '—'}")
                    st.markdown(f"**Actual Type:** {actual_type}")

                    if guessed_type and actual_type in guessed_type:
                        st.success("🎉 Correct! Nice instincts, planet hunter.")
                    else:
                        st.warning("❌ Not quite. Check the hints and try another!")

                    IMG = {
                        "Terrestrial": ("images/terrestrial.jpg", "*Fun Fact: Some terrestrial exoplanets might have molten surfaces like lava worlds.*"),
                        "Super-Earth": ("images/superearth.jpg", "*Fun Fact: Super-Earths could have crushing gravity and thick atmospheres.*"),
                        "Neptunian":   ("images/neptunian.jpg", "*Fun Fact: Neptunian worlds have hazy skies and fierce supersonic winds.*"),
                        "Gas Giant":   ("images/gasgiant.jpg", "*Fun Fact: Gas giants can have storms that last for centuries, like Jupiter’s Great Red Spot.*"),
                        "Unknown":     (None, "*Not enough info to classify this one.*")
                    }
                    path, funfact = IMG.get(actual_type, (None, "*No image available.*"))
                    
                    # --- centered image + centered italic caption ---
                    IMG_WIDTH = 360  # tweak to your taste

                    if path and os.path.exists(path):
                        col_l, col_c, col_r = st.columns([1, 2, 1])
                        with col_c:
                            st.image(path, width=IMG_WIDTH)
                            st.markdown(
                                f"<em style='display:block; text-align:center; margin-top:6px;'>{funfact.strip('*')}</em>",
                                unsafe_allow_html=True
                            )
                    else:
                        col_l, col_c, col_r = st.columns([1, 2, 1])
                        with col_c:
                            st.info("Image not found. Place images here: /images/terrestrial.jpg, superearth.jpg, neptunian.jpg, gasgiant.jpg")
                            st.markdown(
                                f"<em style='display:block; text-align:center; margin-top:6px;'>{funfact.strip('*')}</em>",
                                unsafe_allow_html=True
                            )


# ---------------------------
# MANUAL INPUT TAB
# ---------------------------
with tab3:
    st.subheader("🧮 Manual Input Prediction")
    cols = st.columns(2)
    vals = {}
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