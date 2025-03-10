import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load ASL dataset
data_path = "asl_landmarks_final.csv"
df = pd.read_csv(data_path)

# Extract features (X) and labels (y)
X = df.iloc[:, :-1].values  # All landmark coordinates
y = df["label"].values  # ASL letters

# Encode labels (A-Z, space, del)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Reshape X for CNN + LSTM (samples, time steps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)

# Build CNN + LSTM model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1, X_reshaped.shape[2])),
    Flatten(),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")  # Output layer for classification
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save the model
model.save("asl_cnn_lstm_model.h5")

# Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
