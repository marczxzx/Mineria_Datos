import pandas as pd
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────

def load_movielens(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset MovieLens 100K.
    El archivo tiene formato: userId  movieId  rating  timestamp
    separados por comas.
    """
    df = pd.read_csv(filepath, sep=",")
    return df[["userId", "movieId", "rating"]]

def build_user_ratings(df: pd.DataFrame) -> dict:
    """
    Convierte el DataFrame en un diccionario:
        { userId: { movieId: rating, ... }, ... }

    Esta estructura es más eficiente para calcular similitudes
    porque evita iterar sobre películas que ningún usuario calificó.
    """
    user_ratings = defaultdict(dict) # ayuda a crear diccionarios facilmente
    for row in df.itertuples(index=False):
        user_ratings[row.userId][row.movieId] = float(row.rating)
    return dict(user_ratings)

def load_movie_titles(movies_path: str) -> dict:
    """Carga título -> movieId (opcional, para legibilidad)"""

    df = pd.read_csv(movies_path, sep=",")
    return dict(zip(df["movieId"], df["title"]))
