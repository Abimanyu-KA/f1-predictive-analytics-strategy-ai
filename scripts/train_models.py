import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    mean_absolute_error, classification_report
)

# 1. SETUP PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'modeling'
MODEL_EXPORT_PATH = ROOT_DIR / 'models'
MODEL_EXPORT_PATH.mkdir(exist_ok=True)

def engineer_advanced_features(df, historical_df=None):
    """Calculates F1-specific logic for track history and passability."""
    ref_df = historical_df if historical_df is not None else df
    
    # Circuit Overtake-ability
    circuit_pass = ref_df.groupby('race_name')['pos_gain_loss'].mean().to_dict()
    df['circuit_passability'] = df['race_name'].map(circuit_pass).fillna(0)

    # Driver Track History
    driver_track = ref_df.groupby(['driverRef', 'race_name'])['positionOrder'].mean().reset_index()
    driver_track = driver_track.rename(columns={'positionOrder': 'driver_track_history'})
    df = df.merge(driver_track, on=['driverRef', 'race_name'], how='left')
    df['driver_track_history'] = df['driver_track_history'].fillna(12)

    # Team Historical Strength
    team_hist = ref_df.groupby('constructor_name')['positionOrder'].mean().to_dict()
    df['team_history'] = df['constructor_name'].map(team_hist).fillna(12)
    
    return df

def optimize_hyperparameters(X, y, model_type='clf'):
    """Finds the best settings for the Random Forest to improve accuracy."""
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    }
    base = RandomForestClassifier(random_state=42) if model_type == 'clf' else RandomForestRegressor(random_state=42)
    
    # GridSearchCV finds the 'sweet spot' for model parameters
    grid = GridSearchCV(base, param_grid, cv=3, n_jobs=-1, scoring='accuracy' if model_type == 'clf' else 'neg_mean_absolute_error')
    grid.fit(X, y)
    return grid.best_estimator_

def train_and_export():
    print("--- [1/5] Loading & Engineering Data ---")
    train_df = pd.read_csv(DATA_DIR / 'train_f1_2017_2022.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_f1_2023_2024.csv')

    train_df = engineer_advanced_features(train_df)
    test_df = engineer_advanced_features(test_df, historical_df=train_df)

    FEATURES = ['grid', 'driver_form', 'driver_track_history', 'team_form', 'team_history', 
                'circuit_passability', 'driver_points_pre_race', 'driver_pos_pre_race']

    X_train, X_test = train_df[FEATURES], test_df[FEATURES]

    # --- MODEL 1: PODIUM CLASSIFIER (Optimized) ---
    print("\n--- [2/5] Optimizing Podium Classifier ---")
    podium_clf = optimize_hyperparameters(X_train, train_df['top3'], model_type='clf')
    p_preds = podium_clf.predict(X_test)
    print(f"Podium Accuracy: {accuracy_score(test_df['top3'], p_preds):.2%}")
    print(f"Podium Precision: {precision_score(test_df['top3'], p_preds):.2f}")
    print(f"Podium Recall: {recall_score(test_df['top3'], p_preds):.2f}")

    # --- MODEL 2: POINTS CLASSIFIER ---
    print("\n--- [3/5] Training Points Classifier ---")
    points_clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    points_clf.fit(X_train, train_df['top10'])
    print(f"Points Accuracy: {accuracy_score(test_df['top10'], points_clf.predict(X_test)):.2%}")

    # --- MODEL 3: POSITION REGRESSOR (Cross-Validated) ---
    print("\n--- [4/5] Training Position Regressor ---")
    pos_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    
    # Cross-validation proves the model works across different data slices
    cv_scores = cross_val_score(pos_reg, X_train, train_df['positionOrder'], cv=5, scoring='neg_mean_absolute_error')
    print(f"Mean Training Error (Cross-Val): {np.abs(cv_scores).mean():.2f} positions")
    
    pos_reg.fit(X_train, train_df['positionOrder'])
    mae = mean_absolute_error(test_df['positionOrder'], pos_reg.predict(X_test))
    print(f"Final Test Error (MAE): {mae:.2f} positions")

    # Feature Importance - What drives the F1 model?
    print("\n--- Top Performance Drivers ---")
    importances = pd.Series(pos_reg.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importances)

    # --- EXPORT ---
    print(f"\n--- [5/5] Exporting Assets to {MODEL_EXPORT_PATH} ---")
    joblib.dump(podium_clf, MODEL_EXPORT_PATH / 'podium_clf.joblib')
    joblib.dump(points_clf, MODEL_EXPORT_PATH / 'points_clf.joblib')
    joblib.dump(pos_reg, MODEL_EXPORT_PATH / 'pos_reg.joblib')
    test_df.to_csv(MODEL_EXPORT_PATH / 'app_data_advanced.csv', index=False)
    print("Export Complete. Models are production-ready!")

if __name__ == "__main__":
    train_and_export()