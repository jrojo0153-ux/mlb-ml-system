import os
import joblib
import pandas as pd
import numpy as np
import statsapi
from datetime import datetime, timedelta
from config import MODEL_PATH, KELLY_FRACTION
from data_collector import fetch_espn_live_odds

TEAM_MAP = {
    'Arizona Diamondbacks': 109, 'D-backs': 109, 'ARI': 109,
    'Atlanta Braves': 144, 'ATL': 144,
    'Baltimore Orioles': 110, 'BAL': 110,
    'Boston Red Sox': 111, 'BOS': 111,
    'Chicago Cubs': 112, 'CHC': 112,
    'Chicago White Sox': 145, 'CWS': 145, 'CHW': 145,
    'Cincinnati Reds': 113, 'CIN': 113,
    'Cleveland Guardians': 114, 'CLE': 114,
    'Colorado Rockies': 115, 'COL': 115,
    'Detroit Tigers': 116, 'DET': 116,
    'Houston Astros': 117, 'HOU': 117,
    'Kansas City Royals': 118, 'KC': 118, 'KCR': 118,
    'Los Angeles Angels': 108, 'LAA': 108,
    'Los Angeles Dodgers': 119, 'LAD': 119,
    'Miami Marlins': 146, 'MIA': 146,
    'Milwaukee Brewers': 158, 'MIL': 158,
    'Minnesota Twins': 142, 'MIN': 142,
    'New York Mets': 121, 'NYM': 121,
    'New York Yankees': 147, 'NYY': 147,
    'Oakland Athletics': 133, 'Athletics': 133, 'OAK': 133,
    'Philadelphia Phillies': 143, 'PHI': 143,
    'Pittsburgh Pirates': 134, 'PIT': 134,
    'San Diego Padres': 135, 'SD': 135, 'SDG': 135,
    'San Francisco Giants': 137, 'SF': 137, 'SFG': 137,
    'Seattle Mariners': 136, 'SEA': 136,
    'St. Louis Cardinals': 138, 'STL': 138,
    'Tampa Bay Rays': 139, 'TB': 139, 'TBR': 139,
    'Texas Rangers': 140, 'TEX': 140,
    'Toronto Blue Jays': 141, 'TOR': 141,
    'Washington Nationals': 120, 'WSH': 120, 'WAS': 120
}

def resolve_team_id(name: str) -> int:
    if not name:
        return None
    clean = name.strip()
    if clean in TEAM_MAP:
        return TEAM_MAP[clean]
    for key, value in TEAM_MAP.items():
        if key.lower() in clean.lower() or clean.lower() in key.lower():
            return value
    return None

def fetch_live_rolling_metrics(team_id: int, ref_date_str: str) -> dict:
    # Consulta los encuentros reales jugados el último mes
    start_date = (datetime.strptime(ref_date_str, "%Y-%m-%d") - timedelta(days=35)).strftime("%Y-%m-%d")
    end_date = (datetime.strptime(ref_date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        games = statsapi.schedule(start_date=start_date, end_date=end_date, team=team_id)
    except Exception:
        games = []
        
    completed = []
    for g in games:
        if g.get('status') == 'Final' and g.get('game_type') in ['R', 'F', 'D', 'L', 'W']:
            is_home = g['home_id'] == team_id
            runs_scored = g['home_score'] if is_home else g['away_score']
            runs_allowed = g['away_score'] if is_home else g['home_score']
            won = 1 if runs_scored > runs_allowed else 0
            completed.append({
                'date': g['game_date'],
                'runs_scored': runs_scored,
                'runs_allowed': runs_allowed,
                'won': won
            })
            
    completed = sorted(completed, key=lambda x: x['date'])[-10:]
    
    if len(completed) < 3:
        return {'rolling_runs_scored': 4.5, 'rolling_runs_allowed': 4.5, 'rolling_win_rate': 0.5, 'rest_days': 1}
        
    df_temp = pd.DataFrame(completed)
    last_game_date = pd.to_datetime(df_temp['date'].iloc[-1])
    current_game_date = pd.to_datetime(ref_date_str)
    rest_days = min(max((current_game_date - last_game_date).days, 1), 10)
    
    return {
        'rolling_runs_scored': df_temp['runs_scored'].mean(),
        'rolling_runs_allowed': df_temp['runs_allowed'].mean(),
        'rolling_win_rate': df_temp['won'].mean(),
        'rest_days': rest_days
    }

def parse_live_american_odds(odds_details: str, home_name: str, away_name: str):
    default_home, default_away = -110, -110
    if not odds_details or odds_details == "N/A" or " " not in odds_details:
        return default_home, default_away
    try:
        parts = odds_details.split(" ")
        fav_abbr = parts[0]
        fav_odds = int(parts[1])
        
        fav_id = resolve_team_id(fav_abbr)
        home_id = resolve_team_id(home_name)
        away_id = resolve_team_id(away_name)
        
        underdog_odds = -fav_odds - 20 if fav_odds < 0 else -fav_odds + 20
        underdog_odds = max(underdog_odds, 100) if underdog_odds > 0 else min(underdog_odds, -100)
            
        if fav_id == home_id:
            return fav_odds, underdog_odds
        elif fav_id == away_id:
            return underdog_odds, fav_odds
    except Exception:
        pass
    return default_home, default_away

def am_to_dec(am_odds):
    return (am_odds / 100) + 1 if am_odds > 0 else (100 / abs(am_odds)) + 1

def run_predictions():
    if not os.path.exists(MODEL_PATH):
        print("Modelo ausente. Corre primero train.py para entrenarlo.")
        return
        
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_cols = model_data['features']
    
    print("Capturando datos actuales de ESPN...")
    today_games = fetch_espn_live_odds()
    
    if not today_games:
        print("No se encontraron partidos programados para hoy en ESPN.")
        return
        
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Analizando {len(today_games)} partidos para la jornada: {today_str}\n")
    
    results = []
    for g in today_games:
        home_name, away_name = g['home_name'], g['away_name']
        home_id, away_id = resolve_team_id(home_name), resolve_team_id(away_name)
        
        if not home_id or not away_id:
            continue
            
        h_stats = fetch_live_rolling_metrics(home_id, today_str)
        a_stats = fetch_live_rolling_metrics(away_id, today_str)
        
        feat_vector = {
            'home_rolling_runs_scored': h_stats['rolling_runs_scored'],
            'home_rolling_runs_allowed': h_stats['rolling_runs_allowed'],
            'home_rolling_win_rate': h_stats['rolling_win_rate'],
            'home_rest_days': h_stats['rest_days'],
            'away_rolling_runs_scored': a_stats['rolling_runs_scored'],
            'away_rolling_runs_allowed': a_stats['rolling_runs_allowed'],
            'away_rolling_win_rate': a_stats['rolling_win_rate'],
            'away_rest_days': a_stats['rest_days'],
            'diff_win_rate': h_stats['rolling_win_rate'] - a_stats['rolling_win_rate'],
            'diff_runs_scored': h_stats['rolling_runs_scored'] - a_stats['rolling_runs_scored'],
            'diff_runs_allowed': h_stats['rolling_runs_allowed'] - a_stats['rolling_runs_allowed'],
            'diff_rest_days': h_stats['rest_days'] - a_stats['rest_days']
        }
        
        df_feat = pd.DataFrame([feat_vector])[feature_cols]
        prob_home = model.predict_proba(df_feat)[0, 1]
        prob_away = 1 - prob_home
        
        home_am, away_am = parse_live_american_odds(g['odds_details'], home_name, away_name)
        home_dec, away_dec = am_to_dec(home_am), am_to_dec(away_am)
        
        ev_home = (prob_home * home_dec) - 1
        ev_away = (prob_away * away_dec) - 1
        
        reco = "Pasar (No hay EV+)"
        kelly_pct = 0.0
        
        if ev_home > 0.02:
            reco = f"Apostar Local ({home_name})"
            kelly_pct = (ev_home / (home_dec - 1)) * KELLY_FRACTION
        elif ev_away > 0.02:
            reco = f"Apostar Visita ({away_name})"
            kelly_pct = (ev_away / (away_dec - 1)) * KELLY_FRACTION
            
        results.append({
            'Partido': f"{away_name} @ {home_name}",
            'Prob_Local': f"{prob_home:.1%}",
            'Prob_Visita': f"{prob_away:.1%}",
            'Línea_Local': f"{home_am:+d}",
            'Línea_Visita': f"{away_am:+d}",
            'Decisión': reco,
            'Sugerido_Kelly': f"{kelly_pct:.1%}" if kelly_pct > 0 else "0.0%"
        })
        
    if results:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("====================== RECOMENDACIONES DE LA JORNADA ======================")
        print(pd.DataFrame(results).to_string(index=False))
        print("==========================================================================")
        print(f"* Basado en un control del Criterio de Kelly al {KELLY_FRACTION:.0%}.\n")
    else:
        print("No se pudieron generar predicciones debido a problemas con la resolución de nombres.")

if __name__ == "__main__":
    run_predictions()
