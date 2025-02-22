The file thermal_comfort_model.pkl is a serialized (saved) machine-learning model used to predict thermal comfort levels based on input parameters like temperature, humidity, airspeed, clothing level, activity level, and mean radiant temperature (MRT).

It is likely a Random Forest Classifier model, which takes these inputs and outputs a prediction (0, 1, or 2), which maps to:

0 → Uncomfortable
1 → Comfortable
2 → Hot
🔍 What Does This .pkl File Contain?
It contains a trained ML model that has been:

Trained on past data to recognize patterns.
Saved (pickled) using Python’s pickle module so it can be reused without retraining.
Loaded in Flask app to make predictions on new user inputs

The app.py file in the project is Flask backend that:

Loads the thermal_comfort_model.pkl machine learning model
Receives input data from the frontend (website)
Predicts the thermal comfort level using the model
Sends the prediction (Hot, Comfortable, or Uncomfortable) back to the frontend
