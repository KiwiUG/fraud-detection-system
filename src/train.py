import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

from preprocess import load_and_preprocess_data

# ---- Load Data ----
print("📂 Loading dataset...")
X, y, preprocessor = load_and_preprocess_data("data/AIML_Dataset.csv")

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✅ Training samples: {len(X_train)}, Fraud cases: {sum(y_train)}")

# ---- Step 1: Apply preprocessing first ----
print("🔄 Transforming features...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ---- Step 2: Handle Class Imbalance ----
print("📊 Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_transformed, y_train)
print(f"✅ After SMOTE: {X_res.shape[0]} samples ({sum(y_res)} frauds)")

# ---- Step 3: Train Random Forest ----
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    # class_weight="balanced",  <-- FIX: Removed this redundancy
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_res, y_res)
print("✅ Model training complete.")

joblib.dump(model, "chosen_model/rf_fraud_model.joblib")
joblib.dump(preprocessor, "chosen_model/preprocessor_rf.joblib")

# ---- Step 4: Evaluate ----
y_pred = model.predict(X_test_transformed)
print("\n--- Model Evaluation ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))