from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)
model = load_model("model/fraud_cnn.h5")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")
feature_order = joblib.load("model/feature_order.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Gather inputs
            input_data = {
                "cc_num": int(request.form["cc_num"]),
                "merchant": request.form["merchant"],
                "category": request.form["category"],
                "amt": float(request.form["amt"]),
                "first": request.form["first"],
                "last": request.form["last"],
                "gender": request.form["gender"],
                "street": request.form["street"],
                "city": request.form["city"],
                "state": request.form["state"],
                "zip": int(request.form["zip"]),
                "lat": float(request.form["lat"]),
                "long": float(request.form["long"]),
                "city_pop": int(request.form["city_pop"]),
                "job": request.form["job"],
                "dob": datetime.strptime(request.form["dob"], "%Y-%m-%d").timestamp(),
                "trans_num": request.form["trans_num"],
                "merch_lat": float(request.form["merch_lat"]),
                "merch_long": float(request.form["merch_long"]),
                "trans_date_trans_time": 0,  # dummy
                "unix_time": 0  # dummy
            }

            # Encode categorical features
            for col in encoders:
                input_data[col] = encoders[col].transform([input_data[col]])[0]

            # Fill missing fields
            for col in feature_order:
                if col not in input_data:
                    input_data[col] = 0

            # Arrange & scale
            input_df = pd.DataFrame([[input_data[col] for col in feature_order]], columns=feature_order)
            scaled = scaler.transform(input_df)
            final_input = scaled.reshape(1, scaled.shape[1], 1)

            # Predict
            prob = model.predict(final_input)[0][0]
            prediction = prob > 0.5
        except Exception as e:
            print("Prediction Error:", e)
            prediction = "Error"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
