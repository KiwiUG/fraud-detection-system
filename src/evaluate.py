# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

# --- Configuration ---
DATA_FILEPATH = 'data/raw/upi_transactions.csv'  
MODEL_PATH = 'model/isolation_forest_model.joblib'

# 1. Load and preprocess data
X, y = preprocess_data(DATA_FILEPATH)

# 2. Split data (ensure it's the same split as in train.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Load the pre-trained model
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# 4. Make predictions on the test set
predictions = model.predict(X_test)

# 5. Map predictions to match fraud_flag format
# Isolation Forest: -1 for anomalies (fraud), 1 for inliers (normal)
# We map these to 1 (fraud) and 0 (normal) respectively.
mapped_predictions = pd.Series(predictions).map({1: 0, -1: 1})

# 6. Evaluate and print results
print("\n--- Model Evaluation ---")
print("✅ Classification Report:")
print(classification_report(y_test, mapped_predictions))

print("✅ Confusion Matrix:")
cm = confusion_matrix(y_test, mapped_predictions)
print(cm)

# Optional: Display the confusion matrix visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()