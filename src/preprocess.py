import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_preprocess_data(file_path: str):
    """Load, clean, and preprocess UPI transaction dataset."""

    # ---- Load ----
    df = pd.read_csv(file_path)

    # ---- Rename columns ----
    df.rename(columns={
        'nameOrig': 'sender',
        'nameDest': 'receiver',
        'oldbalanceOrg': 'sender_old_bal',
        'newbalanceOrig': 'sender_new_bal',
        'oldbalanceDest': 'receiver_old_bal',
        'newbalanceDest': 'receiver_new_bal',
        'isFraud': 'fraud',
        'isFlaggedFraud': 'flagged'
    }, inplace=True)

    # ---- Feature Engineering ----
    df['sender_balance_delta'] = df['sender_new_bal'] - df['sender_old_bal']
    df['receiver_balance_delta'] = df['receiver_new_bal'] - df['receiver_old_bal']

    df['sender_expected_new'] = df['sender_old_bal'] - df['amount']
    df['sender_diff_expected'] = df['sender_new_bal'] - df['sender_expected_new']

    df['sender_zero_balance'] = (df['sender_old_bal'] == 0).astype(int)
    df['receiver_zero_balance'] = (df['receiver_old_bal'] == 0).astype(int)

    # ---- Drop unnecessary columns ----
    X = df.drop(columns=['fraud', 'flagged', 'sender', 'receiver'])
    y = df['fraud']

    # ---- Define features ----
    num_features = [
        'amount', 'sender_old_bal', 'sender_new_bal',
        'receiver_old_bal', 'receiver_new_bal',
        'sender_balance_delta', 'receiver_balance_delta',
        'sender_diff_expected'
    ]
    cat_features = ['type']

    # ---- Column Transformer ----
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    return X, y, preprocessor