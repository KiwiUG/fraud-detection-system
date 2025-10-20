# evaluate.py
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from preprocess import preprocess_data

# --- CONFIG ---
DATA_FILEPATH = 'data/upi_transactions.csv'
MODEL_PATH = 'model/xgb_fraud_model.joblib'  # or any .joblib model

# 1️⃣ Load & preprocess
X, y, cat_cols = preprocess_data(DATA_FILEPATH)

# 2️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3️⃣ Handle categorical encoding (same as train.py)
print("🎯 Applying one-hot encoding for categorical columns...")
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align test columns with training
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# 4️⃣ Sanity check
print("\n--- DATA CHECK ---")
if X_test_encoded.isnull().values.any():
    sys.exit("❌ Stop: Found NaN values in X_test. Clean data before evaluation.")
print("✅ No NaN values found.")

# 5️⃣ Load model (handle dict or direct)
print(f"\n📦 Loading model from: {MODEL_PATH}")
obj = joblib.load(MODEL_PATH)
model = obj.get("model") if isinstance(obj, dict) and "model" in obj else obj
print(f"✅ Model loaded ({type(model).__name__})")

# 6️⃣ Make predictions (handle both binary classifiers & probabilistic)
print("\n🔮 Making predictions...")
try:
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
        # Default threshold = 0.5
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_encoded)
except Exception as e:
    sys.exit(f"❌ Prediction failed: {e}")

# 7️⃣ Evaluate
print("\n--- MODEL EVALUATION ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n🔢 Summary Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Evaluation complete.")
