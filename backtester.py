import numpy as np
import pandas as pd
import joblib
import os
from config import MODEL_PATH, KELLY_FRACTION, BANKROLL_INICIAL, DATA_DIR

def american_to_decimal(am_odds):
    return (am_odds / 100) + 1 if am_odds > 0 else (100 / abs(am_odds)) + 1

def decimal_to_american(dec_odds):
    return int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))

def run_backtest():
    if not os.path.exists(MODEL_PATH):
        print("Entrena primero el modelo ejecutando train.py.")
        return
        
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_cols = model_data['features']
    
    from data_collector import fetch_historical_games
    from preprocessor import build_features
    
    print("Descargando segmento histórico de control para Backtesting...")
    # Probando con un subset del cierre de la temporada 2025
    df_test = fetch_historical_games("08/01/2025", "10/01/2025")
    featured_df = build_features(df_test)
    
    if featured_df.empty:
        return
        
    X = featured_df[feature_cols]
    y = featured_df['winner_home'].values
    prob_home = model.predict_proba(X)[:, 1]
    
    bankroll = BANKROLL_INICIAL
    total_bets = 0
    wins = 0
    history = []
    
    for idx, row in featured_df.iterrows():
        p_home = prob_home[idx]
        p_away = 1 - p_home
        
        # Simular cuotas del mercado usando las ventanas móviles de winrate + un margen/vig de 4.5%
        h_wr = row['home_rolling_win_rate'] if pd.notna(row['home_rolling_win_rate']) else 0.5
        a_wr = row['away_rolling_win_rate'] if pd.notna(row['away_rolling_win_rate']) else 0.5
        total_wr = h_wr + a_wr
        p_home_market = np.clip((h_wr / total_wr) + np.random.normal(0, 0.02), 0.1, 0.9) if total_wr > 0 else 0.5
        p_away_market = 1 - p_home_market
        
        home_dec = 1 / (p_home_market * 1.045)
        away_dec = 1 / (p_away_market * 1.045)
        
        # Calcular Expected Value (EV)
        ev_home = (p_home * home_dec) - 1
        ev_away = (p_away * away_dec) - 1
        
        bet_placed = False
        target_odds = 0
        won = False
        ev_taken = 0
        
        # Buscar Valor Positivo (EV > 2%)
        if ev_home > 0.02:
            bet_placed = "Home"
            target_odds = home_dec
            ev_taken = ev_home
            won = (y[idx] == 1)
        elif ev_away > 0.02:
            bet_placed = "Away"
            target_odds = away_dec
            ev_taken = ev_away
            won = (y[idx] == 0)
            
        if bet_placed:
            # Fórmula de Kelly: f = EV / (DecOdds - 1)
            b = target_odds - 1
            f_kelly = (ev_taken / b) * KELLY_FRACTION if b > 0 else 0
            bet_amount = min(bankroll * f_kelly, bankroll * 0.05) # Límite estricto de riesgo: máximo 5% de banca por apuesta
            
            if bet_amount > 0:
                total_bets += 1
                if won:
                    bankroll += bet_amount * (target_odds - 1)
                    wins += 1
                else:
                    bankroll -= bet_amount
                    
        history.append(bankroll)
        
    print("\n============ MÉTRICAS FINANCIERAS (BACKTEST) ============")
    print(f"Bankroll Inicial: {BANKROLL_INICIAL:.2f}")
    print(f"Bankroll Final: {bankroll:.2f}")
    print(f"Retorno Absoluto: {((bankroll - BANKROLL_INICIAL)/BANKROLL_INICIAL)*100:.2f}%")
    print(f"Frecuencia de Apuestas: {total_bets} de {len(featured_df)} partidos evaluados.")
    if total_bets > 0:
        print(f"Porcentaje de Aciertos en Apuestas: {(wins/total_bets)*100:.2f}%")
    print("=========================================================\n")

if __name__ == "__main__":
    run_backtest()
