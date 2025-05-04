import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import joblib

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

def evaluate_model(model, X, y):
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    return rmse

def cross_validate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    return np.sqrt(-scores.mean())

def save_model(model, path):
    joblib.dump(model, path)

