import pandas as pd
import numpy as np
from config import ROLLING_WINDOW

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas móviles de desempeño por equipo previas al encuentro a modelar.
    """
    if df.empty:
        return df
    
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Transformar a formato largo para calcular métricas por equipo de manera secuencial
    records = []
    for idx, row in df.iterrows():
        records.append({
            'game_id': row['game_id'],
            'date': row['date'],
            'team_id': row['home_id'],
            'team_name': row['home_name'],
            'is_home': 1,
            'runs_scored': row['home_score'],
            'runs_allowed': row['away_score'],
            'won': 1 if row['home_score'] > row['away_score'] else 0
        })
        records.append({
            'game_id': row['game_id'],
            'date': row['date'],
            'team_id': row['away_id'],
            'team_name': row['away_name'],
            'is_home': 0,
            'runs_scored': row['away_score'],
            'runs_allowed': row['home_score'],
            'won': 1 if row['away_score'] > row['home_score'] else 0
        })
        
    team_df = pd.DataFrame(records)
    team_df = team_df.sort_values(by=['team_id', 'date']).reset_index(drop=True)
    
    # Desplazar (shift) para excluir el juego actual del cálculo móvil
    team_df['rolling_runs_scored'] = team_df.groupby('team_id')['runs_scored'].shift(1).rolling(window=ROLLING_WINDOW, min_periods=3).mean()
    team_df['rolling_runs_allowed'] = team_df.groupby('team_id')['runs_allowed'].shift(1).rolling(window=ROLLING_WINDOW, min_periods=3).mean()
    team_df['rolling_win_rate'] = team_df.groupby('team_id')['won'].shift(1).rolling(window=ROLLING_WINDOW, min_periods=3).mean()
    
    # Calcular días de descanso
    team_df['prev_date'] = team_df.groupby('team_id')['date'].shift(1)
    team_df['rest_days'] = (team_df['date'] - team_df['prev_date']).dt.days
    team_df['rest_days'] = team_df['rest_days'].fillna(7).clip(upper=10) # Suavizar valores atípicos
    
    # Re-estructurar a formato ancho
    home_feats = team_df[team_df['is_home'] == 1].copy()
    away_feats = team_df[team_df['is_home'] == 0].copy()
    
    home_feats = home_feats.rename(columns={
        'rolling_runs_scored': 'home_rolling_runs_scored',
        'rolling_runs_allowed': 'home_rolling_runs_allowed',
        'rolling_win_rate': 'home_rolling_win_rate',
        'rest_days': 'home_rest_days'
    })
    
    away_feats = away_feats.rename(columns={
        'rolling_runs_scored': 'away_rolling_runs_scored',
        'rolling_runs_allowed': 'away_rolling_runs_allowed',
        'rolling_win_rate': 'away_rolling_win_rate',
        'rest_days': 'away_rest_days'
    })
    
    merged = pd.merge(
        df,
        home_feats[['game_id', 'home_rolling_runs_scored', 'home_rolling_runs_allowed', 'home_rolling_win_rate', 'home_rest_days']],
        on='game_id'
    )
    merged = pd.merge(
        merged,
        away_feats[['game_id', 'away_rolling_runs_scored', 'away_rolling_runs_allowed', 'away_rolling_win_rate', 'away_rest_days']],
        on='game_id'
    )
    
    # Generar variables diferenciales
    merged['diff_win_rate'] = merged['home_rolling_win_rate'] - merged['away_rolling_win_rate']
    merged['diff_runs_scored'] = merged['home_rolling_runs_scored'] - merged['away_rolling_runs_scored']
    merged['diff_runs_allowed'] = merged['home_rolling_runs_allowed'] - merged['away_rolling_runs_allowed']
    merged['diff_rest_days'] = merged['home_rest_days'] - merged['away_rest_days']
    
    return merged.dropna().reset_index(drop=True)
