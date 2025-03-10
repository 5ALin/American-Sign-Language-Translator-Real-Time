import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "asl_landmarks_final.csv"
df = pd.read_csv(data_path)

# Extract features (X) and labels (y)
X = df.iloc[:, :-1].values  # All columns except 'label'
y = df["label"].values  # The last column

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an SVM classifier
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "asl_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully.")
