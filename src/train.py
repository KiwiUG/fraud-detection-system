import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_data, preprocess_data

def train_model(data_path="data/raw/creditcard.csv", model_path="model.pkl"):
    df = load_data(data_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    model = RandomForestClassifier(class_weight="balanced", n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    train_model()
