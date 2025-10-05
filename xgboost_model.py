import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_clean, KEEP_FEATURES

# Load dataset
df, mission = load_and_clean("data/merged.csv")

X = df[KEEP_FEATURES]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", num_class=3)
model.fit(X_train, y_train)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

# Save model
joblib.dump(model, "models/xgb_model.joblib")
print("✅ Model saved to models/xgb_model.joblib")
