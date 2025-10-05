import pandas as pd
import numpy as np

KEEP_FEATURES = [
  "koi_period","koi_duration","koi_depth","koi_prad",
  "koi_sma","koi_incl","koi_teq","koi_srad","koi_smass","koi_kepmag"
]

# --- Universal feature mapping across Kepler, K2, and TESS ---
FEATURE_MAP = {
    "koi_period": ["koi_period", "kep_period", "pl_orbper", "orbital_period"],
    "koi_duration": ["koi_duration", "pl_trandurh", "transit_duration"],
    "koi_depth": ["koi_depth", "pl_trandep", "transit_depth"],
    "koi_prad": ["koi_prad", "pl_rade", "planet_radius"],
    "koi_teq": ["koi_teq", "pl_eqt", "equilibrium_temp", "temp_teq"],
    "koi_srad": ["koi_srad", "st_rad", "stellar_radius"],
    "koi_kepmag": ["koi_kepmag", "st_tmag", "sy_vmag"]
}

def detect_mission(df):
    cols = set(df.columns.str.lower())
    if any("koi_" in c for c in cols):
        return "Kepler"
    elif any("pl_" in c for c in cols):
        return "TESS"
    elif any("kep_" in c for c in cols):
        return "K2"
    else:
        return "Unknown"

def auto_map_features(df, feature_map):
    """Rename columns based on known aliases (Kepler, K2, TESS)."""
    df_cols = [c.lower().strip() for c in df.columns]
    rename_dict = {}

    for target_col, aliases in feature_map.items():
        for alias in aliases:
            if alias.lower() in df_cols:
                rename_dict[alias] = target_col
                break

    df.rename(columns=rename_dict, inplace=True)
    return df


def fuzzy_match_unmapped(df, expected_features, threshold=80):
    """Fallback fuzzy matching for columns not caught by alias mapping (optimized)."""
    if len(df.columns) > 200:
        print("⚠️ Large dataset detected — skipping fuzzy matching for speed.")
        return df

    # Import only when function is actually called
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        print("RapidFuzz not installed — skipping fuzzy matching.")
        return df

    df_cols = [c.lower().strip() for c in df.columns]
    rename_dict = {}
    for feature in expected_features:
        if feature in df_cols:
            continue
        match = process.extractOne(feature, df_cols, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            rename_dict[match[0]] = feature

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    return df


def standardize_dataset(df, features):
    """
    Standardizes column names across Kepler, K2, and TESS datasets.
    Uses FEATURE_MAP and fuzzy matching to auto-align to Kepler-style koi_* names.
    Ensures no valid numeric data is lost or replaced with NaN.
    """
    mission = "Unknown"
    lower_cols = [c.lower() for c in df.columns]

    # --- Mission Detection ---
    if any("kepid" in c for c in lower_cols):
        mission = "Kepler"
    elif any("epic" in c for c in lower_cols):
        mission = "K2"
    elif any("tic" in c for c in lower_cols) or "st_tmag" in lower_cols or "tfopwg_disp" in lower_cols:
        mission = "TESS"


    # --- Step 1: Auto-map using FEATURE_MAP ---
    df = auto_map_features(df, FEATURE_MAP)

    # --- Step 2: Add mission-specific mappings ---
    if mission == "K2":
        rename_map = {
            "pl_orbper": "koi_period",
            "pl_trandur": "koi_duration",   # convert later
            "pl_trandep": "koi_depth",
            "pl_rade": "koi_prad",
            "pl_orbsmax": "koi_sma",
            "pl_orbincl": "koi_incl",
            "pl_eqt": "koi_teq",
            "st_rad": "koi_srad",
            "st_mass": "koi_smass"
        }
        df.rename(columns=rename_map, inplace=True)

        # Convert transit duration from days → hours
        if "pl_trandur" in df.columns:
            df["koi_duration"] = df["pl_trandur"].astype(float) * 24

        # --- Brightness handling ---
        if "sy_vmag" in df.columns:
            df["koi_kepmag"] = df["sy_vmag"] - 0.1
        elif "sy_kmag" in df.columns:
            df["koi_kepmag"] = df["sy_kmag"]
        else:
            df["koi_kepmag"] = np.nan


    elif mission == "TESS":
        rename_map = {
            "pl_orbper": "koi_period",
            "pl_trandurh": "koi_duration",     # ✅ already in hours
            "pl_trandep": "koi_depth",
            "pl_rade": "koi_prad",
            "pl_orbsmax": "koi_sma",
            "pl_orbincl": "koi_incl",
            "pl_eqt": "koi_teq",
            "st_rad": "koi_srad",
            "st_mass": "koi_smass",
            "st_tmag": "koi_kepmag",
        }
        df.rename(columns=rename_map, inplace=True)

    # --- Step 3: Fuzzy match unmapped columns ---
    df = fuzzy_match_unmapped(df, expected_features=features, threshold=85)

    # --- Step 4: Ensure all required columns exist ---
    for f in features:
        if f not in df.columns:
            df[f] = np.nan

    # --- Step 5: Clean up ---
    df = df.replace({pd.NA: np.nan})

    return df, mission


def load_and_clean(path):
    """
    Loads and cleans Kepler, K2, or TESS dataset automatically.
    Works even if label or feature columns differ in name.
    """
    df = pd.read_csv(path, comment="#", low_memory=False)
    lower_cols = [c.lower() for c in df.columns]

    # --- Mission detection ---
    if any("koi_" in c for c in lower_cols) or "kepid" in lower_cols:
        mission = "Kepler"
    elif any("epic" in c for c in lower_cols):
        mission = "K2"
    elif any("tic" in c for c in lower_cols) or "st_tmag" in lower_cols or "tfopwg_disp" in lower_cols:
        mission = "TESS"
    else:
        mission = "Unknown"

    # --- Determine label column ---
    label_col = None
    for candidate in ["koi_disposition", "disposition", "tfopwg_disp"]:
        if candidate in df.columns:
            label_col = candidate
            break

    # --- Standardize and map features ---
    df_std, _ = standardize_dataset(df.copy(), KEEP_FEATURES)

    # --- Label mapping ---
    if label_col:
        if label_col == "tfopwg_disp":
            mapping = {"FP": 0, "PC": 1, "CP": 2, "KP": 2}
        else:
            mapping = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}

        df_std["label"] = df[label_col].map(lambda x: mapping.get(str(x).upper().strip(), np.nan))
        df_std = df_std.dropna(subset=["label"])
        df_std["label"] = df_std["label"].astype(int)
    else:
        df_std["label"] = np.nan

    # --- Keep numeric features ---
    for f in KEEP_FEATURES:
        df_std[f] = pd.to_numeric(df_std[f], errors="coerce")
    df_std[KEEP_FEATURES] = df_std[KEEP_FEATURES].fillna(df_std[KEEP_FEATURES].median())

    return df_std, mission

