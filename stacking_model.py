import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from preprocess import load_and_clean, KEEP_FEATURES

# --- Load and clean merged dataset ---
df, mission = load_and_clean("data/merged.csv")

# --- Define features and target ---
X = df[KEEP_FEATURES]
y = df["label"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --- Base models ---
estimators = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", num_class=3)),
    ('rf', RandomForestClassifier(n_estimators=150, random_state=42))
]

# --- Meta model ---
meta = LogisticRegression(max_iter=1000)

# --- Stacking Classifier ---
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    stack_method='predict_proba',
    n_jobs=-1
)
stack_model.fit(X_train, y_train)

# --- Evaluate ---
y_train_pred = stack_model.predict(X_train)
y_test_pred = stack_model.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

# --- Save model ---
joblib.dump(stack_model, "models/stacking_model.joblib")
print("✅ Model saved to models/stacking_model.joblib")
