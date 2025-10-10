# preprocess.py 
import pandas as pd

def preprocess_data(csv_filepath):
    """
    Loads data from a CSV file, engineers features from the timestamp,
    preprocesses it for model training, and separates features from the target variable.

    Args:
        csv_filepath (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the features DataFrame (X) and the target Series (y).
    """
    print("Loading and preprocessing data...")
    # Load the dataset
    df = pd.read_csv(csv_filepath)

    # --- TIMESTAMP FEATURE ENGINEERING ---
    # Convert 'timestamp' column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract new time-based features
    # Note: Your original data already has hour, day_of_week, is_weekend
    # We can add more granular features.
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    
    # Drop the original timestamp and transaction_id columns
    df = df.drop(['transaction_id', 'timestamp'], axis=1)
    # --- END OF NEW CODE ---

    # Identify categorical columns for one-hot encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features (X) and target (y)
    X = df.drop('fraud_flag', axis=1)
    y = df['fraud_flag']
    
    print("Preprocessing complete with new time features.")
    return X, y
