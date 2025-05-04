import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from src.feature_extraction import extract_features

def load_data(csv_path, audio_dir, is_train=True):
    df = pd.read_csv(csv_path)
    X, y, filenames = [], [], []

    for _, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        features = extract_features(file_path)
        X.append(features)
        filenames.append(row['filename'])
        if is_train:
            y.append(row['label'])

    if is_train:
        return np.array(X), np.array(y)
    else:
        return np.array(X), filenames

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, 'grammar_scoring_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Training RMSE: {rmse:.4f}")
    return model, scaler

def predict(model, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions
