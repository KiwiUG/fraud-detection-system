# QR-Based User Reputation Scanner for Fraud Detection

This project implements a system to assess a user's risk profile by scanning a QR code. It uses a machine learning model trained on a financial transaction dataset to analyze a user's entire transaction history. If any past activity is deemed fraudulent, the user is flagged as high-risk.



## 📋 Table of Contents

- [About The Project](#about-the-project)
- [System Workflow](#system-workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [1. Model Training](#1-model-training)
  - [2. QR Code Generation](#2-qr-code-generation)
  - [3. Reputation Scanning](#3-reputation-scanning)
- [Future Improvements](#future-improvements)

## 🌟 About The Project

The core idea is to provide a "reputation check" before engaging in a transaction with a new user. Instead of predicting the risk of a single *new* transaction, this system checks the user's *past* behavior.

- **Input**: A QR code image containing a User ID.
- **Process**: The system retrieves the user's entire transaction history from a local data source (`user_data.csv`). It then runs a pre-trained Random Forest model on every single historical transaction.
- **Output**: A final verdict:
    - `✅ SAFE USER`: If no past transactions were flagged as fraudulent.
    - `🚨 HIGH RISK USER`: If one or more past transactions were flagged as fraudulent.

## ⚙️ System Workflow

The project is divided into three main stages:

1.  **Training**: A Random Forest Classifier is trained on the `AIML_Dataset.csv` to learn the patterns of fraudulent transactions. A preprocessor and the trained model are saved.
2.  **QR Generation**: Unique QR codes, containing formatted User IDs (e.g., `USER_ID|S-2002`), are generated for each user.
3.  **Scanning**: The `fraud_scanner.py` script reads a user's QR code, looks up their history in `user_data.csv`, and uses the trained model to analyze their past behavior.

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

- Python 3.9+
- `pip` package manager
- **For macOS users**: [Homebrew](https://brew.sh/) to install the `zbar` library.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv fraud-env
    source fraud-env/bin/activate
    ```
    *On Windows, use: `fraud-env\Scripts\activate`*

3.  **Install the ZBar shared library (Required for QR code reading):**
    -   **On macOS (using Homebrew):**
        ```bash
        brew install zbar
        ```
    -   **On Debian/Ubuntu:**
        ```bash
        sudo apt-get install libzbar0
        ```

4.  **Install Python dependencies:**
    ```bash
    pip install pandas scikit-learn imbalanced-learn joblib qrcode opencv-python pyzbar
    ```

## ▶️ Usage

Run the scripts in the following order:

1.  **Train the Model**:
    This script will process the data, train the Random Forest model and preprocessor, and save them to the `chosen_model/` directory.
    ```bash
    python app/train.py
    ```

2.  **Generate Sample QR Codes**:
    This will create QR code images for the sample users (`U-1001`, `S-2002`) and save them in the `qrcodes/` directory.
    ```bash
    python app/qr_generator.py
    ```

3.  **Run the Fraud Scanner**:
    This is the main application. It will simulate scanning the generated QR codes and print the final reputation assessment for each user.
    ```bash
    python app/fraud_scanner.py
    ```
    You should see output indicating whether "U-1001" is safe and "S-2002" is a high-risk user.

## 🧠 How It Works

### 1. Model Training

- **Feature Engineering**: New features are created from the raw data, such as `sender_balance_delta` and `sender_diff_expected`, to help the model find predictive patterns.
- **Handling Imbalance**: The training dataset is highly imbalanced (very few fraud cases). **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset by creating synthetic examples of the minority class (fraud).
- **Model**: A `RandomForestClassifier` is used for its robustness and good performance on tabular data.

### 2. QR Code Generation

The `qr_generator.py` script uses the `qrcode` library to create standard QR code images. Each QR code is encoded with a string in the format `USER_ID|{id}`, which is easily parsable by the scanner.

### 3. Reputation Scanning

The `fraud_scanner.py` script is the core of the application logic:
1.  **Read QR Code**: It uses `opencv-python` to load the QR image and `pyzbar` to decode it, extracting the `user_id`.
2.  **Fetch History**: It loads the `user_data.csv` file into a pandas DataFrame and retrieves all rows corresponding to the extracted `user_id`.
3.  **Iterative Prediction**: It loops through every historical transaction of the user. For each transaction, it applies the same feature engineering and preprocessing steps used during training.
4.  **Verdict**: The pre-trained model predicts if the transaction was fraudulent. If even one transaction is flagged as fraud (`prediction == 1`), the loop stops, and the user is marked as "HIGH RISK".

## 💡 Future Improvements

- [ ] **Develop a REST API**: Instead of reading from CSVs, build an API where an endpoint receives a User ID and returns the risk score.
- [ ] **Use a Real Database**: Replace `user_data.csv` with a database like PostgreSQL or MySQL for more efficient data retrieval.
- [ ] **Build a Simple UI**: Create a web or mobile front-end that allows a user to upload a QR code image and see the result.
- [ ] **Experiment with Models**: Try more advanced models like XGBoost, LightGBM, or CatBoost, which often perform better on tabular fraud data.