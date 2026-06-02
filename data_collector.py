import pandas as pd
import requests
import statsapi
import os
from datetime import datetime
from config import DATA_DIR

def fetch_historical_games(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene el histórico de partidos de la MLB desde la API oficial (via MLB-StatsAPI).
    """
    print(f"Obteniendo partidos desde {start_date} hasta {end_date}...")
    try:
        games = statsapi.schedule(start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"Error llamando a MLB StatsAPI: {e}")
        return pd.DataFrame()
    
    cleaned_games = []
    for g in games:
        # Filtrar solo juegos completados ('Final') de temporada regular o postemporada
        if g.get('status') == 'Final' and g.get('game_type') in ['R', 'F', 'D', 'L', 'W']:
            cleaned_games.append({
                'game_id': g.get('game_id'),
                'date': g.get('game_date'),
                'home_id': g.get('home_id'),
                'home_name': g.get('home_name'),
                'home_score': g.get('home_score'),
                'away_id': g.get('away_id'),
                'away_name': g.get('away_name'),
                'away_score': g.get('away_score'),
                'winner_home': 1 if g.get('home_score') > g.get('away_score') else 0
            })
            
    df = pd.DataFrame(cleaned_games)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    return df

def fetch_espn_live_odds() -> list:
    """
    Consulta la API pública de ESPN para capturar los partidos del día de hoy con sus cuotas.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception as e:
        print(f"Error consultando ESPN API: {e}")
        return []
    
    live_games = []
    events = data.get('events', [])
    for event in events:
        competition = event.get('competitions', [{}])[0]
        odds_data = competition.get('odds', [{}])
        
        odds_details = "N/A"
        over_under = None
        if odds_data:
            odds_details = odds_data[0].get('details', 'N/A')
            over_under = odds_data[0].get('overUnder', None)
            
        home_team = None
        away_team = None
        
        competitors = competition.get('competitors', [])
        for comp in competitors:
            if comp.get('homeAway') == 'home':
                home_team = comp.get('team', {}).get('displayName')
            else:
                away_team = comp.get('team', {}).get('displayName')
                
        live_games.append({
            'game_id': event.get('id'),
            'date': event.get('date'),
            'home_name': home_team,
            'away_name': away_team,
            'odds_details': odds_details,
            'over_under': over_under
        })
    return live_games
