import os

# Directorios de datos y modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mlb_model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)

# Parámetros del modelo
ROLLING_WINDOW = 10  # Cantidad de juegos previos para calcular estadísticas móviles
TEST_SIZE = 0.2     # Tamaño del set de prueba para evaluación
RANDOM_STATE = 42

# Parámetros cuantitativos de apuestas
KELLY_FRACTION = 0.1      # Kelly fraccional para mitigar la varianza (10% de Kelly total)
BANKROLL_INICIAL = 1000.0 # Unidad monetaria simulada para backtesting
