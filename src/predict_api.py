# src/predict_api.py

from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the model (from MLflow Registry)
model_name = "iris-best-model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

@app.route('/')
def home():
    return "Iris Model Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON and convert to DataFrame
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
