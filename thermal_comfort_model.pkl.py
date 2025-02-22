# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "ashrae_db2.01.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Step 1: Selecting important features
selected_columns = [
    "Air temperature (C)", "Radiant temperature (C)", "Relative humidity (%)", "Air velocity (m/s)",
    "Clo", "Met", "Thermal sensation"
]
df_selected = df[selected_columns]

# Step 2: Handling missing values (Fill with column mean)
df_selected.fillna(df_selected.mean(), inplace=True)

# Step 3: Define features (X) and target variable (y)
X = df_selected.drop(columns=["Thermal sensation"])
y = df_selected["Thermal sensation"]

# Convert thermal sensation (-3 to +3) into categories: Cold, Comfortable, Hot
def categorize_comfort(value):
    if value <= -1:
        return "Cold"
    elif value >= 1:
        return "Hot"
    else:
        return "Comfortable"

y = y.apply(categorize_comfort)

# Encode categorical labels (Cold=0, Comfortable=1, Hot=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 4: Splitting dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=250, random_state=42,max_features='sqrt',max_depth=None,min_samples_split=10)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cold", "Comfortable", "Hot"]))

# Step 9: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=["Cold", "Comfortable", "Hot"], yticklabels=["Cold", "Comfortable", "Hot"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
plt.show()


import pickle

with open("thermal_comfort_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully!")