from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)
# Load model from absolute path relative to current file
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model.pkl'))
model = joblib.load(model_path)
# model = joblib.load("../model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame(data)
    data = input_df[["HouseAge"]]
    prediction = model.predict(data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)