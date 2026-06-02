import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from data_collector import fetch_historical_games
from preprocessor import build_features

def train_model():
    # Obtener el histórico de entrenamiento (ej. Temporadas completas pasadas)
    df_2024 = fetch_historical_games("04/01/2024", "10/30/2024")
    df_2025 = fetch_historical_games("04/01/2025", "10/30/2025")
    
    if df_2024.empty and df_2025.empty:
        print("Error: Sin datos históricos suficientes.")
        return
        
    df_all = pd.concat([df_2024, df_2025], ignore_index=True)
    print(f"Base de entrenamiento total: {len(df_all)} juegos procesados.")
    
    featured_df = build_features(df_all)
    
    feature_cols = [
        'home_rolling_runs_scored', 'home_rolling_runs_allowed', 'home_rolling_win_rate', 'home_rest_days',
        'away_rolling_runs_scored', 'away_rolling_runs_allowed', 'away_rolling_win_rate', 'away_rest_days',
        'diff_win_rate', 'diff_runs_scored', 'diff_runs_allowed', 'diff_rest_days'
    ]
    
    X = featured_df[feature_cols]
    y = featured_df['winner_home']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("Ajustando modelo de Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=4,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- EVALUACIÓN DE PRUEBA OUT-OF-SAMPLE ---")
    print(f"Precisión General (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nDetalle de Métricas:")
    print(classification_report(y_test, y_pred))
    
    model_data = {
        'model': model,
        'features': feature_cols
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Modelo exportado en: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
