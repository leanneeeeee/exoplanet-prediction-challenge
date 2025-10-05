import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# --- Train Random Forest ---
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

# --- Save model ---
joblib.dump(model, "models/random_forest_model.joblib")
print("✅ Model saved to models/random_forest_model.joblib")
