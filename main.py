import datetime
import json
import os
import sys
import logging
import statsapi
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HISTORICO_FILE = 'historico_predicciones.csv'
PREDICCIONES_FILE = 'predicciones_mlb.csv'
WEIGHTS_FILE = 'model_weights.json'
BANKROLL_FILE = 'bankroll.json'
JUICE_RATE = 0.04
KELLY_FRACTION = 0.25
DEFAULT_BANKROLL = 1000.0
DEFAULT_ERA = 4.50

class MLBPredictor:
    def __init__(self):
        self.weights = self.load_weights()
        self.bankroll = self.load_bankroll()
        self.historical_data = self.load_historical()
        
    def load_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading weights: {e}. Using defaults.")
        
        return {
            'peso_pitcher': 1.0,
            'peso_localia': 1.0,
            'peso_historial': 0.5
        }
    
    def save_weights(self):
        try:
            with open(WEIGHTS_FILE, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.info(f"Weights saved: {self.weights}")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    def load_bankroll(self):
        if os.path.exists(BANKROLL_FILE):
            try:
                with open(BANKROLL_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('amount', DEFAULT_BANKROLL)
            except Exception as e:
                logger.warning(f"Error loading bankroll: {e}. Using default.")
        
        return DEFAULT_BANKROLL
    
    def save_bankroll(self, amount):
        try:
            with open(BANKROLL_FILE, 'w') as f:
                json.dump({'amount': round(amount, 2)}, f, indent=2)
            self.bankroll = amount
            logger.info(f"Bankroll updated: ${amount:.2f}")
        except Exception as e:
            logger.error(f"Error saving bankroll: {e}")
    
    def load_historical(self):
        if os.path.exists(HISTORICO_FILE):
            try:
                return pd.read_csv(HISTORICO_FILE)
            except Exception as e:
                logger.warning(f"Error loading historical data: {e}")
        
        return pd.DataFrame()
    
    def initialize_historical(self):
        if not os.path.exists(HISTORICO_FILE):
            columns = [
                'Fecha', 'ID_Partido', 'Equipo_Visita', 'Equipo_Casa',
                'Prob_Visita', 'Prob_Casa', 'Cuota_Visita', 'Cuota_Casa',
                'Kelly_Visita', 'Kelly_Casa', 'Resultado_Real', 'Pronostico_Correcto', 'ROI_Generado'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(HISTORICO_FILE, index=False)
            self.historical_data = df
    
    def update_historical_results(self):
        try:
            self.historical_data = pd.read_csv(HISTORICO_FILE)
        except Exception as e:
            logger.error(f"Error reading historical file: {e}")
            return
        
        pending_games = self.historical_data[
            (self.historical_data['Resultado_Real'].isna()) | 
            (self.historical_data['Resultado_Real'] == '')
        ].copy()
        
        if pending_games.empty:
            logger.info("No pending games to update.")
            return
        
        logger.info(f"Found {len(pending_games)} pending games to update.")
        
        for idx, game in pending_games.iterrows():
            game_id = game['ID_Partido']
            try:
                game_data = statsapi.get('game', {'gamePk': int(game_id)})
                
                if game_data.get('gameData', {}).get('status', {}).get('detailedState') == 'Game Over':
                    home_team = game['Equipo_Casa']
                    away_team = game['Equipo_Visita']
                    
                    home_score = game_data.get('liveData', {}).get('linescore', {}).get('teams', {}).get('home', {}).get('runs', 0)
                    away_score = game_data.get('liveData', {}).get('linescore', {}).get('teams', {}).get('away', {}).get('runs', 0)
                    
                    if home_score > away_score:
                        resultado = home_team
                    elif away_score > home_score:
                        resultado = away_team
                    else:
                        resultado = 'Empate'
                    
                    self.historical_data.loc[idx, 'Resultado_Real'] = resultado
                    
                    pronostico = 1 if (
                        (resultado == home_team and self.historical_data.loc[idx, 'Prob_Casa'] > self.historical_data.loc[idx, 'Prob_Visita']) or
                        (resultado == away_team and self.historical_data.loc[idx, 'Prob_Visita'] > self.historical_data.loc[idx, 'Prob_Casa'])
                    ) else 0
                    
                    self.historical_data.loc[idx, 'Pronostico_Correcto'] = pronostico
                    
                    kelly_bet = self.historical_data.loc[idx, 'Kelly_Casa'] if resultado == home_team else self.historical_data.loc[idx, 'Kelly_Visita']
                    roi = (kelly_bet * self.historical_data.loc[idx, 'Cuota_Casa' if resultado == home_team else 'Cuota_Visita']) - kelly_bet if pronostico else -kelly_bet
                    
                    self.historical_data.loc[idx, 'ROI_Generado'] = roi
                    self.bankroll += roi
                    
                    logger.info(f"Updated game {game_id}: {resultado}, Correct={pronostico}, ROI={roi:.2f}")
                    
            except Exception as e:
                logger.warning(f"Error fetching game {game_id}: {e}")
        
        self.historical_data.to_csv(HISTORICO_FILE, index=False)
        self.save_bankroll(self.bankroll)
    
    def calculate_model_bias(self):
        if self.historical_data.empty or 'Pronostico_Correcto' not in self.historical_data.columns:
            return None
        
        completed = self.historical_data[self.historical_data['Resultado_Real'].notna()]
        
        if len(completed) < 5:
            return None
        
        accuracy = completed['Pronostico_Correcto'].mean()
        
        visited_bias = completed[completed['Pronostico_Correcto'] == 0]['Prob_Visita'].mean()
        home_bias = completed[completed['Pronostico_Correcto'] == 0]['Prob_Casa'].mean()
        
        return {
            'accuracy': accuracy,
            'visited_bias': visited_bias,
            'home_bias': home_bias
        }
    
    def adjust_weights(self):
        bias = self.calculate_model_bias()
        
        if bias is None:
            logger.info("Insufficient data for weight adjustment.")
            return
        
        logger.info(f"Model Bias Analysis: {bias}")
        
        if bias['accuracy'] < 0.45:
            adjustment = 1.02
        elif bias['accuracy'] > 0.60:
            adjustment = 0.98
        else:
            adjustment = 1.0
        
        if bias['visited_bias'] and not pd.isna(bias['visited_bias']) and bias['visited_bias'] > 50:
            self.weights['peso_pitcher'] *= adjustment
        elif bias['visited_bias'] and not pd.isna(bias['visited_bias']) and bias['visited_bias'] < 40:
            self.weights['peso_pitcher'] /= adjustment
        
        if bias['home_bias'] and not pd.isna(bias['home_bias']) and bias['home_bias'] > 55:
            self.weights['peso_localia'] *= adjustment
        elif bias['home_bias'] and not pd.isna(bias['home_bias']) and bias['home_bias'] < 45:
            self.weights['peso_localia'] /= adjustment
        
        self.save_weights()
    
    def simulate_odds(self, prob_win):
        prob_loss = 1 - prob_win
        
        decimal_odds = (1 + JUICE_RATE) / prob_win
        
        return round(decimal_odds, 2)
    
    def calculate_implied_probability(self, decimal_odds):
        return 1 / decimal_odds
    
    def has_positive_expected_value(self, our_prob, decimal_odds):
        implied_prob = self.calculate_implied_probability(decimal_odds)
        return our_prob > implied_prob
    
    def kelly_bet_size(self, our_prob, decimal_odds, unit_bankroll=None):
        if unit_bankroll is None:
            unit_bankroll = self.bankroll
        
        if unit_bankroll <= 0:
            return 0
        
        implied_prob = self.calculate_implied_probability(decimal_odds)
        
        if our_prob <= implied_prob:
            return 0
        
        win_prob = our_prob
        loss_prob = 1 - our_prob
        win_amount = decimal_odds - 1
        
        kelly_fraction = (win_prob * win_amount - loss_prob) / win_amount
        
        if kelly_fraction <= 0:
            return 0
        
        fractional_kelly = kelly_fraction * KELLY_FRACTION
        
        bet_size = unit_bankroll * fractional_kelly
        
        return round(max(0, bet_size), 2)
    
    def get_pitcher_stats(self, pitcher_id):
        try:
            stats_data = statsapi.player_stat_data(pitcher_id, group="pitching", type="season")
            if stats_data.get('stats') and len(stats_data['stats']) > 0:
                era = float(stats_data['stats'][0]['stats'].get('era', DEFAULT_ERA))
                wins = int(stats_data['stats'][0]['stats'].get('wins', 0))
                innings_pitched = float(stats_data['stats'][0]['stats'].get('inningsPitched', 0))
                return {
                    'era': era,
                    'wins': wins,
                    'innings_pitched': innings_pitched
                }
        except Exception as e:
            logger.warning(f"Error fetching pitcher {pitcher_id}: {e}")
        
        return {
            'era': DEFAULT_ERA,
            'wins': 0,
            'innings_pitched': 0
        }
    
    def calculate_prediction(self, home_pitcher_id, away_pitcher_id, home_name, away_name):
        home_stats = self.get_pitcher_stats(home_pitcher_id)
        away_stats = self.get_pitcher_stats(away_pitcher_id)
        
        era_home = home_stats['era']
        era_away = away_stats['era']
        
        total_era = era_home + era_away
        if total_era == 0:
            total_era = 9.0
        
        prob_home_base = (era_away / total_era) * 100
        prob_away_base = (era_home / total_era) * 100
        
        home_advantage = 4.0 * self.weights['peso_localia']
        prob_home = prob_home_base + home_advantage
        prob_away = prob_away_base - home_advantage
        
        total_prob = prob_home + prob_away
        if total_prob == 0:
            total_prob = 100
        
        prob_home = (prob_home / total_prob) * 100
        prob_away = (prob_away / total_prob) * 100
        
        prob_home = round(min(99.0, max(1.0, prob_home)), 2)
        prob_away = round(min(99.0, max(1.0, prob_away)), 2)
        
        return {
            'prob_home': prob_home,
            'prob_away': prob_away,
            'era_home': round(era_home, 2),
            'era_away': round(era_away, 2)
        }
    
    def process_games(self, target_date=None):
        if target_date is None:
            target_date = datetime.date.today().strftime('%Y-%m-%d')
        
        logger.info(f"Processing games for {target_date}")
        
        try:
            games = statsapi.schedule(date=target_date)
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return []
        
        if not games:
            logger.info(f"No games scheduled for {target_date}")
            # Crear archivo vacío para evitar errores en git add
            empty_df = pd.DataFrame(columns=[
                'Fecha', 'ID_Partido', 'Equipo_Visita', 'Equipo_Casa',
                'Prob_Visita', 'Prob_Casa', 'Cuota_Visita', 'Cuota_Casa',
                'Kelly_Visita', 'Kelly_Casa'
            ])
            empty_df.to_csv(PREDICCIONES_FILE, index=False)
            logger.info(f"Created empty predictions file: {PREDICCIONES_FILE}")
            return []
        
        predictions = []
        
        for game in games:
            try:
                game_id = game.get('game_pk')
                home_name = game.get('home_name')
                away_name = game.get('away_name')
                
                home_pitcher_id = game.get('home_probable_pitcher_id')
                away_pitcher_id = game.get('away_probable_pitcher_id')
                
                if not all([game_id, home_name, away_name, home_pitcher_id, away_pitcher_id]):
                    logger.warning(f"Incomplete data for game {game_id}")
                    continue
                
                logger.info(f"Analyzing: {away_name} @ {home_name}")
                
                prediction = self.calculate_prediction(
                    home_pitcher_id, away_pitcher_id, home_name, away_name
                )
                
                prob_home = prediction['prob_home'] / 100
                prob_away = prediction['prob_away'] / 100
                
                odds_home = self.simulate_odds(prob_home)
                odds_away = self.simulate_odds(prob_away)
                
                kelly_home = self.kelly_bet_size(prob_home, odds_home)
                kelly_away = self.kelly_bet_size(prob_away, odds_away)
                
                has_value_home = self.has_positive_expected_value(prob_home, odds_home)
                has_value_away = self.has_positive_expected_value(prob_away, odds_away)
                
                if has_value_home:
                    logger.info(f"Value found: {home_name} @ {odds_home} (prob={prob_home:.2%})")
                if has_value_away:
                    logger.info(f"Value found: {away_name} @ {odds_away} (prob={prob_away:.2%})")
                
                prediction_row = {
                    'Fecha': target_date,
                    'ID_Partido': game_id,
                    'Equipo_Visita': away_name,
                    'Equipo_Casa': home_name,
                    'Prob_Visita': prediction['prob_away'],
                    'Prob_Casa': prediction['prob_home'],
                    'Cuota_Visita': odds_away,
                    'Cuota_Casa': odds_home,
                    'Kelly_Visita': kelly_away,
                    'Kelly_Casa': kelly_home,
                    'Resultado_Real': '',
                    'Pronostico_Correcto': '',
                    'ROI_Generado': ''
                }
                
                predictions.append(prediction_row)
                
            except Exception as e:
                logger.error(f"Error processing game: {e}")
                continue
        
        if predictions:
            new_predictions_df = pd.DataFrame(predictions)
            
            if os.path.exists(HISTORICO_FILE) and not self.historical_data.empty:
                self.historical_data = pd.concat([self.historical_data, new_predictions_df], ignore_index=True)
            else:
                self.historical_data = new_predictions_df
            
            self.historical_data.to_csv(HISTORICO_FILE, index=False)
            logger.info(f"Saved {len(predictions)} predictions to {HISTORICO_FILE}")
            
            # Guardar predicciones del día en archivo separado para git push
            new_predictions_df.to_csv(PREDICCIONES_FILE, index=False)
            logger.info(f"Saved today's predictions to {PREDICCIONES_FILE}")
            
            return predictions
        else:
            logger.info("No valid predictions generated.")
            # Crear archivo vacío como fallback
            empty_df = pd.DataFrame(columns=[
                'Fecha', 'ID_Partido', 'Equipo_Visita', 'Equipo_Casa',
                'Prob_Visita', 'Prob_Casa', 'Cuota_Visita', 'Cuota_Casa',
                'Kelly_Visita', 'Kelly_Casa'
            ])
            empty_df.to_csv(PREDICCIONES_FILE, index=False)
            return []

def main():
    logger.info("Starting MLB Prediction System")
    
    predictor = MLBPredictor()
    
    predictor.initialize_historical()
    
    predictor.update_historical_results()
    
    predictor.adjust_weights()
    
    predictor.process_games()
    
    logger.info(f"Current Bankroll: ${predictor.bankroll:.2f}")
    logger.info("MLB Prediction System completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
