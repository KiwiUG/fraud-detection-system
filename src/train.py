# train.py
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

# --- Configuration ---
DATA_FILEPATH = 'data/raw/upi_transactions.csv'  
MODEL_SAVE_PATH = 'model/isolation_forest_model.joblib'

# 1. Load and preprocess data
X, y = preprocess_data(DATA_FILEPATH)

# 2. Split data into training and testing sets
# We only need X_train for the unsupervised Isolation Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Initialize and Train the Isolation Forest Model
# 'contamination' is the expected proportion of anomalies (fraud) in the data.
# You should adjust this based on your dataset's characteristics.
# A common starting point is a small value like 0.01 (1%).
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

print("Training Isolation Forest model...")
# The model is fit only on the features (X_train), as it's unsupervised.
iso_forest.fit(X_train)
print("Training complete.")

# 4. Save the trained model
joblib.dump(iso_forest, MODEL_SAVE_PATH)
print(f"✅ Model saved successfully to {MODEL_SAVE_PATH}")
