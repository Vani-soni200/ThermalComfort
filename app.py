# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:56:42 2025

@author: Vani Soni
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model
with open("thermal_Comfort_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route("/")  
def home():
    return "Welcome to the Thermal Comfort Prediction App"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging

        # Extract input values
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        air_speed = float(data.get("air_speed", 0))
        clothing = float(data.get("clothing", 0))
        activity = float(data.get("activity", 0))
        
        # **Fix: Include the missing feature**
        mean_radiant_temperature = float(data.get("mrt", 0))  # Adjust this if needed

        # Prepare input for the model
        input_features = np.array([[temperature, humidity, air_speed, clothing, activity, mean_radiant_temperature]])

        # Get prediction from the model
        prediction = model.predict(input_features)[0]

        # Map numerical predictions to human-readable categories
        comfort_mapping = {0: "Uncomfortable", 1: "Comfortable", 2: "Hot"}
        comfort_level = comfort_mapping.get(prediction, "Unknown")

        response = {"comfort_level": comfort_level}
        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)



'''
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("thermal_comfort_model.pkl", "rb") as file:
    model = pickle.load(file)
    
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route("/")  # This defines the homepage route
def home():
    return "Welcome to the Thermal Comfort Prediction App"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request
        data = request.json
        temp = float(data["temperature"])
        humidity = float(data["humidity"])
        air_speed = float(data["airSpeed"])

        # Convert input to numpy array
        features = np.array([[temp, humidity, air_speed]])

        # Get model prediction
        prediction = model.predict(features)[0]

        # Return response
        return jsonify({"comfort_level": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change port if needed
'''
'''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")  # This defines the homepage route
def home():
    return "Welcome to the Thermal Comfort Prediction App"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Data:", data)  # Debugging line

        # Check if 'air_speed' exists
        if "air_speed" not in data:
            return jsonify({"error": "Missing air_speed parameter"}), 400

        temperature = data["temperature"]
        humidity = data["humidity"]
        air_speed = data["air_speed"]  # This is causing the error
        clothing = data["clothing"]
        activity = data["activity"]

        # Dummy prediction logic (replace with your actual ML model)
        comfort_level = "Comfortable" if air_speed > 0.5 else "Uncomfortable"

        return jsonify({"comfort_level": comfort_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
'''