"""Configuration partagée du service de prédiction ML."""
import os

# Clé API Zafronix WC (free tier). Surchargée par la variable d'env WC_API_KEY.
WC_API_KEY = os.getenv("WC_API_KEY", "zwc_free_76e29d8b3d1531a27d30fe8d")
WC_BASE_URL = os.getenv(
    "WC_BASE_URL", "https://api.zafronix.com/fifa/worldcup/v1"
)

# Coupes du Monde historiques utilisées pour l'entraînement (tous les 4 ans).
# 1942 et 1946 n'ont pas eu lieu (2nde guerre mondiale) -> exclues.
TRAIN_YEARS = [y for y in range(1930, 2023, 4) if y not in (1942, 1946)]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
STATS_PATH = os.path.join(os.path.dirname(__file__), "stats.pkl")

# Origines autorisées pour CORS. Dev local + domaines de prod ajoutés via la
# variable d'env CORS_ORIGINS (liste séparée par des virgules), ex :
#   CORS_ORIGINS="https://mon-app.vercel.app,https://www.mon-domaine.com"
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://world-cup2026.alwaysdata.net"
] + [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
