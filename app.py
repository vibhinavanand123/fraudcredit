import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('fraud_model.pkl')

st.title("💳 Credit Card Fraud Detector")
st.write("Enter transaction details below to evaluate risk.")

# Inputs
amount = st.number_input("Transaction Amount (Rs)", min_value=0.0, value=0.00)
distance = st.number_input("Distance from Home (km)", min_value=0.0, value=0.0)
hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
is_international = st.selectbox("Is International Transaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction
if st.button("Evaluate Transaction"):
    input_df = pd.DataFrame([{
        'amount': amount,
        'distance': distance,
        'hour': hour,
        'is_international': is_international
    }])
    
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if pred == 1:
        st.error(f"🚨 **FRAUD ALERT** (Risk Score: {prob:.1%})")
    else:
        st.success(f"✅ **LEGITIMATE TRANSACTION** (Risk Score: {prob:.1%})")
