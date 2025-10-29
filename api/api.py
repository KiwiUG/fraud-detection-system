import joblib
import pandas as pd
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import requests # <-- 1. IMPORT REQUESTS

# --- PATH CONFIGURATION ---
# This line finds the directory where this script (api.py) is located
API_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION (UPDATED) ---
# Read the secret URL from the environment variable set in Render
MODEL_URL = os.environ.get("MODEL_URL")

# --- ADD THIS CHECK ---
if not MODEL_URL:
    print("CRITICAL ERROR: 'MODEL_URL' environment variable not set.")
    # This will stop the server from starting if the URL is missing
    raise ValueError("MODEL_URL environment variable not set. Server cannot start.")
# --- END OF CHECK ---


# Local paths where files will be stored on the server
# UPDATED: Removed "chosen_model" from the path
LOCAL_MODEL_PATH = os.path.join(API_DIR, "rf_fraud_model.joblib")
LOCAL_PREPROCESSOR_PATH = os.path.join(API_DIR, "preprocessor_rf.joblib")

TRANSACTION_FILE = os.path.join(API_DIR, "user_data.csv")
# --- END OF CHANGES ---

# --- DATA LOADING ---
def load_and_index_data(file_path: str):
    """
    Loads transactions and indexes them by 'user_id'.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Transaction file '{file_path}' not found.")
        print("Make sure 'user_data.csv' is in your 'api/data' folder and committed to Git.")
        return None
    
    indexed_data = {}
    for user_id, group in df.groupby('user_id'):
        indexed_data[user_id] = group.to_dict("records")

    print(f"✅ Loaded {df.shape[0]} transactions for {len(indexed_data)} unique users.")
    return indexed_data

# --- ML COMPONENT LOADING (MODIFIED) ---
def load_ml_components(model_url: str, local_model_path: str, local_preprocessor_path: str):
    """
    Downloads the model from Firebase if it doesn't exist,
    then loads both components.
    """
    try:
        # --- 3. ADD DOWNLOAD LOGIC ---
        # Check if the model file *already exists* on the server's disk
        if not os.path.exists(local_model_path):
            print(f"Model not found locally. Downloading from Firebase (this may take a moment)...")
            
            # Download the file
            response = requests.get(model_url)
            response.raise_for_status() # This will raise an error if the download fails
            
            # Write the file to disk
            with open(local_model_path, 'wb') as f:
                f.write(response.content)
            print("✅ Model downloaded successfully.")
        else:
            print("✅ Model file already exists locally.")
        # --- END OF DOWNLOAD LOGIC ---

        # Now, load the files from the local disk
        model = joblib.load(local_model_path)
        
        # Preprocessor is small and should be in Git, so we just load it
        if not os.path.exists(local_preprocessor_path):
            print(f"CRITICAL ERROR: Preprocessor file not found at {local_preprocessor_path}")
            # UPDATED: Changed error message
            print("Make sure 'preprocessor_rf.joblib' is in your 'api/' folder and committed to Git.")
            return None, None
            
        preprocessor = joblib.load(local_preprocessor_path)
        
        print("✅ Loaded model and preprocessor.")
        return model, preprocessor
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load components. Error: {e}")
        return None, None

# --- FEATURE ENGINEERING ---
def single_instance_feature_engineering(df_raw: pd.DataFrame):
    """
    Applies the feature engineering logic used during training.
    """
    df = df_raw.copy()

    # Create the new features
    df['sender_balance_delta'] = df['sender_new_bal'] - df['sender_old_bal']
    df['receiver_balance_delta'] = df['receiver_new_bal'] - df['receiver_old_bal']
    df['sender_expected_new'] = df['sender_old_bal'] - df['amount']
    df['sender_diff_expected'] = df['sender_new_bal'] - df['sender_expected_new']
    
    # Define the exact columns the preprocessor expects
    num_features = [
        'amount', 'sender_old_bal', 'sender_new_bal',
        'receiver_old_bal', 'receiver_new_bal',
        'sender_balance_delta', 'receiver_balance_delta',
        'sender_diff_expected'
    ]
    cat_features = ['type']

    # Return *only* those columns
    return df[num_features + cat_features]

# --- CORE PREDICTION LOGIC ---
def check_user_reputation(user_id: str, model, preprocessor, indexed_data):
    """
    Runs the ML model on a user's entire history and finds the
    MAXIMUM fraud probability.
    """
    transaction_history = indexed_data.get(user_id)
    if not transaction_history: 
        return 0.0, 0 # Return 0% risk and 0 transactions

    max_fraud_probability = 0.0
    
    for i, transaction_dict in enumerate(transaction_history):
        try:
            df_raw = pd.DataFrame([transaction_dict])
            X_engineered = single_instance_feature_engineering(df_raw)
            X_transformed = preprocessor.transform(X_engineered)
            
            # Get probability of class 1 (fraud)
            current_fraud_prob = model.predict_proba(X_transformed)[0][1]

            if current_fraud_prob > max_fraud_probability:
                max_fraud_probability = current_fraud_prob
        
        except Exception as e:
            # Log error but continue checking other transactions
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
    """
    Load all ML assets and data into memory on server startup.
    """
    print("Server starting up...")
    
    # --- 4. UPDATE THE FUNCTION CALL ---
    model, preprocessor = load_ml_components(MODEL_URL, LOCAL_MODEL_PATH, LOCAL_PREPROCESSOR_PATH)
    
    if model is None or preprocessor is None:
        raise RuntimeError("Could not load ML components. Server cannot start.")
    
    indexed_data = load_and_index_data(TRANSACTION_FILE)
    if indexed_data is None:
        raise RuntimeError("Could not load transaction data. Server cannot start.")
    
    # Store the loaded assets in the app's state
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.indexed_data = indexed_data
    
    print("✅ Server startup complete. Ready to accept requests.")


# --- API ENDPOINTS ---
@app.get("/reputation/{user_id}")
def get_user_reputation(user_id: str, request: Request):
    """
    Get the reputation for a single user ID.
    Returns a JSON object with a risk_percentage (0-100).
    """
    print(f"Received request for user_id: {user_id}")

    # Get the loaded assets from the app's state
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor
    indexed_data = request.app.state.indexed_data
    
    if user_id not in indexed_data:
        print(f"User ID not found: {user_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"User ID '{user_id}' not found in transaction history."
        )

    # Get the max fraud score (e.g., 0.925)
    max_prob, count = check_user_reputation(user_id, model, preprocessor, indexed_data)
    
    # Convert to a percentage (e.g., 92.5)
    risk_percentage = round(max_prob * 100, 2)
    
    # Define a risk level and message for the frontend
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
    
    # Return the new JSON response
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
    # Get the port from the environment variable (Render sets this)
    # Default to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)

