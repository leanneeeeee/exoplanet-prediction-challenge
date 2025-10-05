import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = "models/xgb_model.joblib"

'''def load_model():
    return joblib.load(MODEL_PATH)'''
def load_model(model_type="xgb"):
    if model_type == "rf":
        return joblib.load("models/random_forest_model.joblib")
    elif model_type == "stack":
        return joblib.load("models/stacking_model.joblib")
    else:
        return joblib.load("models/xgb_model.joblib")


def predict_row(model, row_df, features):
    X = row_df[features].astype(float)
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    return pred, proba

def incremental_update(model, X_new, y_new):
    # XGBoost continued training approach: retrain using previous model as init_model
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_new, label=y_new)
    params = model.get_xgb_params()
    # Use small num_boost_round to update
    model_booster = model.get_booster()
    updated = xgb.train(params, dtrain, num_boost_round=10, xgb_model=model_booster)
    # Save updated model
    updated_clf = xgb.XGBClassifier()
    updated_clf._Booster = updated
    return updated_clf
