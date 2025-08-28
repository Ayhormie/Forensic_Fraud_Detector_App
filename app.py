import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('xgboost_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Forensic Fraud Detection Model")

# Sidebar for file upload and threshold
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)  # Adjustable threshold

# Process only if a file is uploaded
if uploaded_file is not None:
    # Load and preprocess the uploaded dataset
    df = pd.read_csv(uploaded_file)

    # Preprocess data
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['oldbalanceOrg'] = pd.to_numeric(df['oldbalanceOrg'], errors='coerce')
    df['newbalanceOrig'] = pd.to_numeric(df['newbalanceOrig'], errors='coerce')
    df['oldbalanceDest'] = pd.to_numeric(df['oldbalanceDest'], errors='coerce')
    df['newbalanceDest'] = pd.to_numeric(df['newbalanceDest'], errors='coerce')

    # Calculate derived features
    df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_change_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['step_week'] = df['step'] // 168

    # Infer transaction type (simplified mapping)
    def infer_transaction_type(type_str):
        type_map = {'CASH_IN': 'PAYMENT', 'CASH_OUT': 'TRANSFER', 'DEBIT': 'DEBIT', 'PAYMENT': 'PAYMENT', 'TRANSFER': 'TRANSFER'}
        return type_map.get(type_str, 'DEBIT')

    df['type'] = df['type'].apply(infer_transaction_type)

    # Benford's Law deviation (group-based)
    def benford_dev(amounts):
        if len(amounts) < 10:
            return np.nan
        leads = amounts[amounts > 0].apply(lambda x: int(str(abs(x)).lstrip('0')[0]) if str(abs(x)).lstrip('0') else 0)
        freq = leads.value_counts(normalize=True)
        expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}
        return sum((freq.get(d, 0) - expected[d])**2 for d in expected)

    # Compute benford_dev and handle potential issues
    df['benford_dev'] = df.groupby(['step_week', 'type'])['amount'].transform(benford_dev).fillna(0)

    # One-hot encode type and ensure all expected categories are present
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    expected_types = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in expected_types:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0

    # Normalize amount to amountZ using scaler's mean_ and scale_ for amountZ
    mean_amountZ = scaler.mean_[0]  # Training mean for amountZ
    scale_amountZ = scaler.scale_[0]  # Training std for amountZ
    df['amountZ'] = (df['amount'] - mean_amountZ) / scale_amountZ

    # Select features (match training features)
    features = ['amountZ', 'benford_dev', 'step', 'step_week', 'balance_change_orig', 'balance_change_dest',
                'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    X = df[features].fillna(0)

    # Scale all features using the loaded scaler
    X_scaled = scaler.transform(X)

    # Predict probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Apply custom threshold for predictions
    df['predicted_fraud'] = (probabilities >= threshold).astype(int)
    df['fraud_probability'] = probabilities

    # Flag potential illegal charges (e.g., high debits with anomalies)
    df['illegal_charge_flag'] = ((df['amount'].abs() > 200000) & 
                                (df['type_TRANSFER'] == 1) & 
                                (df['benford_dev'] > 0.1)) * 1

    # Display results
    st.header("Prediction Results")
    st.write("Total Transactions:", len(df))
    st.write("Fraudulent Transactions (Predicted):", len(df[df['predicted_fraud'] == 1]))
    st.write("Potential Illegal Charges:", len(df[df['illegal_charge_flag'] == 1]))

    # Interactive filters
    st.subheader("Transaction Details")
    fraud_filter = st.checkbox("Show Only Fraudulent Transactions", value=False)
    illegal_filter = st.checkbox("Show Only Illegal Charges", value=False)

    if fraud_filter:
        df_display = df[df['predicted_fraud'] == 1]
    elif illegal_filter:
        df_display = df[df['illegal_charge_flag'] == 1]
    else:
        df_display = df

    st.dataframe(df_display)

    # Download processed data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Processed Data as CSV",
        data=csv,
        file_name="processed_simulated_paysim.csv",
        mime="text/csv"
    )

    # Compare with true labels (if available)
    if 'isFraud' in df.columns:
        st.subheader("Model Performance")
        true_positives = len(df[(df['predicted_fraud'] == 1) & (df['isFraud'] == 1)])
        false_positives = len(df[(df['predicted_fraud'] == 1) & (df['isFraud'] == 0)])
        false_negatives = len(df[(df['predicted_fraud'] == 0) & (df['isFraud'] == 1)])
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
else:
    st.write("Please upload your CSV file to proceed.")