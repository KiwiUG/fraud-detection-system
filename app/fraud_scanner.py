import joblib
import pandas as pd
import os
import cv2 # Added for reading images
from pyzbar.pyzbar import decode # Added for decoding QR codes

# --- CONFIGURATION ---
TRANSACTION_FILE = "data/user_data.csv"
MODEL_FILE = os.path.join("chosen_model", "rf_fraud_model.joblib")
PREPROCESSOR_FILE = os.path.join("chosen_model", "preprocessor_rf.joblib")

# --- NEW: QR CODE READER ---
def read_qr_code_from_file(filepath: str):
    """
    Reads a QR code image from the given file and returns the decoded string.
    """
    try:
        image = cv2.imread(filepath)
        if image is None:
            print(f"ERROR: Could not read image from {filepath}. Check file path.")
            return None
            
        qr_codes = decode(image)
        
        if not qr_codes:
            print(f"ERROR: No QR code found in {filepath}.")
            return None
            
        # Return the data from the first QR code found
        qr_data = qr_codes[0].data.decode("utf-8")
        return qr_data
        
    except Exception as e:
        print(f"ERROR: Failed to read QR code from {filepath}. Error: {e}")
        return None

# --- DATA LOADING AND INDEXING ---
def load_and_index_data(file_path: str):
    """
    Loads all transactions from the CSV and indexes them by the SENDER's ID ('nameOrig').
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Transaction file '{file_path}' not found.")
        return None
    
    df.rename(columns={'nameOrig': 'user_id'}, inplace=True)

    indexed_data = {}
    for user_id, group in df.groupby('user_id'):
        group_records = group.to_dict("records")
        for record in group_records:
            record['nameOrig'] = record.pop('user_id') 
        indexed_data[user_id] = group_records

    print(f"✅ Loaded {df.shape[0]} transactions for {len(indexed_data)} unique users.")
    return indexed_data

# --- ML COMPONENT LOADING ---
def load_fraud_model_components(model_filepath: str, preprocessor_filepath: str):
    """
    Loads the trained model and the separate preprocessor object.
    """
    try:
        model = joblib.load(model_filepath)
        preprocessor = joblib.load(preprocessor_filepath)
        print(f"✅ Loaded model and preprocessor.")
        return model, preprocessor

    except FileNotFoundError:
        print("\n" + "="*70)
        print("CRITICAL ERROR: Model or Preprocessor files not found.")
        print(f"Missing files: {model_filepath} and/or {preprocessor_filepath}")
        print("="*70)
        return None, None

# --- FEATURE ENGINEERING (Copied from your preprocess.py) ---
def single_instance_feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering logic used during training.
    """
    df = df_raw.copy()

    df.rename(columns={
        'nameOrig': 'sender',
        'nameDest': 'receiver',
        'oldbalanceOrg': 'sender_old_bal',
        'newbalanceOrig': 'sender_new_bal',
        'oldbalanceDest': 'receiver_old_bal',
        'newbalanceDest': 'receiver_new_bal',
    }, inplace=True)

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

# --- SCANNING AND PREDICTION LOGIC ---
def scan_user_reputation(scanned_data: str, model, preprocessor, indexed_data):
    """
    Main function to extract ID, fetch data, process history, and predict risk.
    """
    if scanned_data is None:
        print("Skipping scan due to previous QR read error.")
        return "ERROR: QR Read Failed"

    print(f"\n--- SCANNING USER REPUTATION: '{scanned_data}' ---")
    is_scam_detected = False

    try:
        prefix, user_id = scanned_data.split('|')
        if prefix != "USER_ID": raise ValueError("Invalid format prefix.")
        print(f"  > Extracted User ID: {user_id}")
    except (ValueError, TypeError):
        print("ERROR: Scanned data format is incorrect. Expected 'USER_ID|ID_VALUE'.")
        return "ERROR: Invalid QR Code Data"

    transaction_history = indexed_data.get(user_id)
    if not transaction_history:
        print(f"  > User ID '{user_id}' not found in transaction history.")
        return "UNKNOWN USER"

    print(f"  > Analyzing {len(transaction_history)} historical transactions...")

    for i, transaction_dict in enumerate(transaction_history):
        try:
            df_raw = pd.DataFrame([transaction_dict])
            X_engineered = single_instance_feature_engineering(df_raw)
            X_transformed = preprocessor.transform(X_engineered)
            prediction = model.predict(X_transformed)[0]

            if prediction == 1:
                is_scam_detected = True
                print(f"    - Txn #{i + 1}: 🚨 HIGH RISK 🚨 (Flagged by model)")
                break 
            
        except Exception as e:
            print(f"    - ERROR: Failed to process Txn #{i+1} due to pipeline error: {e}")
            continue

    print("\n" + "="*70)
    if is_scam_detected:
        final_result = "🚨 HIGH RISK USER (Flagged on one or more past transactions) 🚨"
    else:
        final_result = "✅ SAFE USER (No high-risk transactions found in history) ✅"
    
    print(f"FINAL REPUTATION for User {user_id}: {final_result}")
    print("="*70)
    return final_result

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Initializing User Reputation System...")
    
    # 1. Load data and ML components
    indexed_data = load_and_index_data(TRANSACTION_FILE)
    if indexed_data is None:
        exit()
        
    model, preprocessor = load_fraud_model_components(MODEL_FILE, PREPROCESSOR_FILE)
    if model is None or preprocessor is None:
        exit()

    print("\n--- REPUTATION SCANNER READY ---")
    
    # 2. Define the *filepaths* to your existing QR codes
    #    (These are generated by qr_generator.py)
    safe_user_qr_file = os.path.join("qrcodes", "qr_U-1001.png")
    scammer_qr_file = os.path.join("qrcodes", "qr_S-2002.png")
    
    print("\n" + "="*70)
    print("SIMULATION: SCANNING PROCESS")
    print("="*70)

    # 3. Simulate scanning the Safe User's QR image
    print(f"\nSimulating scan for the **SAFE USER (from {safe_user_qr_file})**:")
    safe_user_qr_data = read_qr_code_from_file(safe_user_qr_file)
    scan_user_reputation(safe_user_qr_data, model, preprocessor, indexed_data)

    # 4. Simulate scanning the Scammer's QR image
    print(f"\nSimulating scan for the **SCAMMER (from {scammer_qr_file})**:")
    scammer_qr_data = read_qr_code_from_file(scammer_qr_file)
    scan_user_reputation(scammer_qr_data, model, preprocessor, indexed_data)

    print("\n--- SIMULATION COMPLETE ---")