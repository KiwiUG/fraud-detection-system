import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop('Class', axis=1))
    return df_scaled, df['Class']
