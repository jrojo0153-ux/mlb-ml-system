import datetime
import statsapi
import pandas as pd

def obtener_partidos_y_predicciones():
    # 1. Obtener la fecha de hoy en formato local (YYYY-MM-DD)
    hoy = datetime.date.today().strftime('%Y-%m-%d')
    print(f"--- PROCESANDO PARTIDOS PARA LA FECHA: {hoy} ---")
    
    # 2. Descargar los partidos programados para el día de hoy
    partidos = statsapi.schedule(date=hoy)
    
    if not partidos:
        print("No hay partidos programados para el día de hoy.")
        return

    datos_predicciones = []

    # 3. Analizar cada partido de forma individual
    for partido in partidos:
        id_partido = partido['game_pk']
        equipo_casa = partido['home_name']
        equipo_visita = partido['away_name']
        
        print(f"\nAnalizando: {equipo_visita} vs {equipo_casa}")
        
        # Obtener los IDs de los lanzadores abridores programados
        try:
            id_pitcher_casa = partido['home_probable_pitcher_id']
            id_pitcher_visita = partido['away_probable_pitcher_id']
        except KeyError:
            print(f"Lanzadores no confirmados aún para este encuentro. Saltando...")
            continue
            
        if not id_pitcher_casa or not id_pitcher_visita:
            print("Falta confirmar uno o ambos lanzadores abridores. Saltando...")
            continue

        # 4. Extracción de estadísticas reales desde la API de MLB
        try:
            # Estadísticas del lanzador de casa
            stats_casa = statsapi.player_stat_data(id_pitcher_casa, group="pitching", type="season")
            era_casa = float(stats_casa['stats'][0]['stats']['era']) if stats_casa['stats'] else 4.50
            
            # Estadísticas del lanzador visitante
            stats_visita = statsapi.player_stat_data(id_pitcher_visita, group="pitching", type="season")
            era_visita = float(stats_visita['stats'][0]['stats']['era']) if stats_visita['stats'] else 4.50
        except Exception:
            # Si es debutante o no tiene datos de la temporada, asignamos un ERA promedio de la liga
            era_casa = 4.50
            era_visita = 4.50

        # 5. Algoritmo de Ventaja Matemática (A menor ERA, mejor lanzador)
        # Calculamos una probabilidad base simplificada pero real basada en el pitcheo abridor
        total_era = era_casa + era_visita
        if total_era == 0: total_era = 9.0
        
        # A menor efectividad del pitcher rival, mayor probabilidad de ganar
        prob_casa = round((era_visita / total_era) * 100, 2)
        prob_visita = round((era_casa / total_era) * 100, 2)
        
        # Ajuste por ventaja de localía estándar en MLB (+4%)
        prob_casa += 4.0
        prob_visita -= 4.0
        
        # Re-normalizar a base 100
        suma_prob = prob_casa + prob_visita
        prob_casa = round((prob_casa / suma_prob) * 100, 2)
        prob_visita = round((prob_visita / suma_prob) * 100, 2)

        print(f" -> {equipo_visita} (Pitcher ERA: {era_visita}) | Probabilidad: {prob_visita}%")
        print(f" -> {equipo_casa} (Pitcher ERA: {era_casa}) | Probabilidad: {prob_casa}%")
        
        datos_predicciones.append({
            'Fecha': hoy,
            'Visita': equipo_visita,
            'Casa': equipo_casa,
            'Prob_Visita_%': prob_visita,
            'Prob_Casa_%': prob_casa
        })

    # 6. Guardar los resultados del día en un reporte CSV
    df = pd.DataFrame(datos_predicciones)
    df.to_csv('predicciones_mlb.csv', index=False)
    print("\nReporte 'predicciones_mlb.csv' generado con éxito en el servidor.")

if __name__ == "__main__":
    obtener_partidos_y_predicciones()
