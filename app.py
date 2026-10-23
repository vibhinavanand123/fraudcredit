import joblib
import pandas as pd

# 1. Load the trained model saved in Step 1
model = joblib.load('fraud_model.pkl')

def evaluate_transaction(amount, distance, hour, is_international):
    input_data = pd.DataFrame([{
        'amount': float(amount),
        'distance': float(distance),
        'hour': int(hour),
        'is_international': int(is_international)
    }])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        'status': 'FRAUD ALERT' if prediction == 1 else 'LEGITIMATE',
        'fraud_probability': f"{probability:.2%}"
    }

if __name__ == '__main__':
    # Test evaluation
    sample_result = evaluate_transaction(amount=250.0, distance=45.0, hour=2, is_international=1)
    print("Test Prediction Output:", sample_result)
