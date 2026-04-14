import pandas as pd
from collections import defaultdict

def cargar_calificaciones(ruta_csv: str) -> pd.DataFrame:
    """Carga calificaciones y convierte timestamps a fechas reales."""
    try:
        df = pd.read_csv(ruta_csv, sep=",")
        # FIX: Eliminar NaNs incluyendo la nueva columna timestamp
        df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"])
        
        # Convertir el timestamp gigante a una fecha real legible
        df['fecha'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df[["userId", "movieId", "rating", "fecha"]]
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {ruta_csv}")
        return pd.DataFrame(columns=["userId", "movieId", "rating", "fecha"])

def construir_historial_usuarios(df: pd.DataFrame) -> dict:
    """Construye un diccionario con el historial, notas y fechas."""
    historial = defaultdict(dict)
    for fila in df.itertuples(index=False):
        # Ahora guardamos un sub-diccionario con la nota y la fecha
        historial[fila.userId][fila.movieId] = {
            'nota': float(fila.rating),
            'fecha': fila.fecha
        }
    return dict(historial)

def cargar_peliculas(ruta_csv: str) -> dict:
    """Carga el catálogo mapeando ID a título y lista de géneros."""
    try:
        df = pd.read_csv(ruta_csv, sep=",")
        catalogo = {}
        for fila in df.itertuples(index=False):
            catalogo[fila.movieId] = {
                'titulo': fila.title,
                # Separamos "Action|Sci-Fi" en una lista real ['Action', 'Sci-Fi']
                'generos': fila.genres.split('|') if isinstance(fila.genres, str) else []
            }
        return catalogo
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el catálogo en {ruta_csv}")
        return {}

def cargar_etiquetas(ruta_csv: str) -> pd.DataFrame:
    """NUEVA: Carga los tags para el análisis semántico del influencer."""
    try:
        df = pd.read_csv(ruta_csv, sep=",")
        df = df.dropna(subset=["userId", "movieId", "tag"])
        
        # Pasamos los tags a minúsculas y quitamos espacios para evitar duplicados ('Sci-Fi' vs 'sci-fi')
        df['tag'] = df['tag'].str.lower().str.strip()
        
        return df[["userId", "movieId", "tag"]]
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo de etiquetas en {ruta_csv}")
        return pd.DataFrame(columns=["userId", "movieId", "tag"])