from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ✅ Load model and scaler from single file
with open("model.pkl", "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
scaler = saved_objects["scaler"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Collect form inputs
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        # Define column names (must match training features)
        features = ['Global_reactive_power', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

        # Convert inputs to DataFrame
        df = pd.DataFrame(features_value, columns=features)

        # Scale inputs using the saved scaler
        df_scaled = scaler.transform(df)

        # Predict using the saved model
        output = model.predict(df_scaled)

        # Return prediction rounded to 2 decimals
        return render_template('result.html', result=round(output[0], 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
