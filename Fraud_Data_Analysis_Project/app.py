from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

#Load Model
model = pickle.load(open("models/finalized_model_RF.sav", "rb"))

#Load Scalar
scalar = pickle.load(open("models/scaler.sav", "rb"))

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    amount = float(request.form["amount"])
    hour = int(request.form["hour"])
    account_age_days = int(request.form["account_age_days"])
    avg_amount_30d = float(request.form["avg_amount_30d"])
    merchant_risk_score = int(request.form["merchant_risk_score"])

    data = pd.DataFrame(
        np.array([[amount,hour,account_age_days,avg_amount_30d,merchant_risk_score]]),
        columns=["amount","hour","account_age_days","avg_amount_30d","merchant_risk_score"])

        # Scale using saved scaler
    input_scaled = scalar.transform(data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        result = "This is a Fraud Transaction" 
    else:
        result = "This is not a Fraud Transaction"

    return render_template("result.html", prediction = result)

if __name__ == "__main__":
    app.run(debug=True)