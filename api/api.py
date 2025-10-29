import joblib
import pandas as pd
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import numpy as np # Added for math operations

# --- CONFIGURATION ---
TRANSACTION_FILE = "data/user_data.csv"
MODEL_FILE = os.path.join("chosen_model", "rf_fraud_model.joblib")
PREPROCESSOR_FILE = os.path.join("chosen_model", "preprocessor_rf.joblib")

# --- DATA LOADING (No changes) ---
def load_and_index_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Transaction file '{file_path}' not found.")
        return None
    
    indexed_data = {}
    for user_id, group in df.groupby('user_id'):
        indexed_data[user_id] = group.to_dict("records")

    print(f"✅ Loaded {df.shape[0]} transactions for {len(indexed_data)} unique users.")
    return indexed_data

# --- ML COMPONENT LOADING (No changes) ---
def load_ml_components(model_filepath: str, preprocessor_filepath: str):
    try:
        model = joblib.load(model_filepath)
        preprocessor = joblib.load(preprocessor_filepath)
        print(f"✅ Loaded model and preprocessor.")
        return model, preprocessor
    except FileNotFoundError:
        print("CRITICAL ERROR: Model or Preprocessor files not found.")
        return None, None

# --- FEATURE ENGINEERING (No changes) ---
def single_instance_feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df['sender_balance_delta'] = df['sender_new_bal'] - df['sender_old_bal']
    df['receiver_balance_delta'] = df['receiver_new_bal'] - df['receiver_old_bal']
    df['sender_expected_new'] = df['sender_old_bal'] - df['amount']
    df['sender_diff_expected'] = df['sender_new_bal'] - df['sender_expected_new']
    
    num_features = [
        'amount', 'sender_old_bal', 'sender_new_bal',
        'receiver_old_bal', 'receiver_new_bal',
        'sender_balance_delta', 'receiver_balance_delta',
        'sender_diff_expected'
    ]
    cat_features = ['type']

    return df[num_features + cat_features]

# --- CORE PREDICTION LOGIC (CHANGED) ---
def check_user_reputation(user_id: str, model, preprocessor, indexed_data):
    """
    Runs the ML model on a user's entire history and finds the
    MAXIMUM fraud probability.
    
    Returns a tuple: (max_fraud_probability, transactions_analyzed)
    """
    transaction_history = indexed_data.get(user_id)
    if not transaction_history:
        return 0.0, 0 # Return 0% risk and 0 transactions

    # We will find the highest fraud score in the user's history
    max_fraud_probability = 0.0
    
    for i, transaction_dict in enumerate(transaction_history):
        try:
            df_raw = pd.DataFrame([transaction_dict])
            X_engineered = single_instance_feature_engineering(df_raw)
            X_transformed = preprocessor.transform(X_engineered)
            
            # --- CHANGE 1: Use predict_proba() ---
            # This returns probabilities for [class_0, class_1]
            # We want the probability of class 1 (fraud)
            current_fraud_prob = model.predict_proba(X_transformed)[0][1]

            # --- CHANGE 2: Find the maximum probability ---
            if current_fraud_prob > max_fraud_probability:
                max_fraud_probability = current_fraud_prob
            
            # --- CHANGE 3: Removed the 'break' statement ---
            # We must check ALL transactions to find the maximum.
        
        except Exception as e:
            print(f"    - ERROR processing Txn #{i+1} for user {user_id}: {e}")
            continue

    # Return the highest probability found
    return max_fraud_probability, len(transaction_history)

# --- API SETUP ---
app = FastAPI(
    title="Fraud Detection API",
    description="An API to check user reputation based on transaction history.",
    version="1.0.0"
)

@app.on_event("startup")
def load_assets():
    print("Server starting up...")
    model, preprocessor = load_ml_components(MODEL_FILE, PREPROCESSOR_FILE)
    if model is None or preprocessor is None:
        raise RuntimeError("Could not load ML components. Server cannot start.")
    
    indexed_data = load_and_index_data(TRANSACTION_FILE)
    if indexed_data is None:
        raise RuntimeError("Could not load transaction data. Server cannot start.")
    
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.indexed_data = indexed_data
    print("✅ Server startup complete. Ready to accept requests.")


@app.get("/reputation/{user_id}")
def get_user_reputation(user_id: str, request: Request):
    """
    Get the reputation for a single user ID.
    Returns a JSON object with a risk_percentage (0-100).
    """
    print(f"Received request for user_id: {user_id}")

    model = request.app.state.model
    preprocessor = request.app.state.preprocessor
    indexed_data = request.app.state.indexed_data
    
    if user_id not in indexed_data:
        print(f"User ID not found: {user_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"User ID '{user_id}' not found in transaction history."
        )

    # --- API LOGIC CHANGED ---
    
    # 1. Get the max fraud score (e.g., 0.925)
    max_prob, count = check_user_reputation(user_id, model, preprocessor, indexed_data)
    
    # 2. Convert to a percentage (e.g., 92.5)
    risk_percentage = round(max_prob * 100, 2)
    
    # 3. Define a risk level and message for the frontend
    if risk_percentage > 75:
        risk_level = "HIGH"
        message = "User has past transactions with a very high probability of fraud."
    elif risk_percentage > 30:
        risk_level = "MEDIUM"
        message = "User has some past transactions with suspicious characteristics."
    else:
        risk_level = "LOW"
        message = "User history appears clean. No high-risk transactions found."
        
    print(f"Returning risk_percentage for {user_id}: {risk_percentage}%")
    
    # 4. Return the new JSON response
    return {
        "user_id": user_id,
        "risk_percentage": risk_percentage,
        "risk_level": risk_level,
        "message": message,
        "transactions_analyzed": count
    }

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running. Go to /docs for documentation."}

# --- RUN THE API ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

