import numpy as np

# 2. SIMILITUD (MÁXIMA PRECISIÓN)


def obtener_peliculas_comunes(perfil_x: dict, perfil_y: dict) -> list:
    """Retorna una lista con los IDs de las películas que ambos usuarios han visto."""
    return list(set(perfil_x.keys()) & set(perfil_y.keys()))

def similitud_coseno(perfil_x: dict, perfil_y: dict) -> float:
    comunes = obtener_peliculas_comunes(perfil_x, perfil_y)
    if not comunes:
        return 0.0

    # FIX: Ahora extraemos explícitamente la ['nota'] del diccionario
    producto = sum(perfil_x[m]['nota'] * perfil_y[m]['nota'] for m in comunes)
    norma_x = np.sqrt(sum(perfil_x[m]['nota'] ** 2 for m in comunes))
    norma_y = np.sqrt(sum(perfil_y[m]['nota'] ** 2 for m in comunes))

    if norma_x == 0 or norma_y == 0:
        return 0.0
    return float(producto / (norma_x * norma_y))

def similitud_euclidiana(perfil_x: dict, perfil_y: dict) -> float:
    comunes = obtener_peliculas_comunes(perfil_x, perfil_y)
    if not comunes:
        return 0.0
        
    diferencia_cuadrada = sum((perfil_x[m]['nota'] - perfil_y[m]['nota']) ** 2 for m in comunes)
    return 1 / (1 + np.sqrt(diferencia_cuadrada))

def similitud_pearson(perfil_x: dict, perfil_y: dict) -> float:
    comunes = obtener_peliculas_comunes(perfil_x, perfil_y)
    if len(comunes) < 2:
        return 0.0

    vec_x = np.array([perfil_x[m]['nota'] for m in comunes])
    vec_y = np.array([perfil_y[m]['nota'] for m in comunes])
    media_x, media_y = np.mean(vec_x), np.mean(vec_y)
    
    x_centrado, y_centrado = vec_x - media_x, vec_y - media_y
    numerador = np.dot(x_centrado, y_centrado)
    denominador = np.sqrt(np.dot(x_centrado, x_centrado)) * np.sqrt(np.dot(y_centrado, y_centrado))

    if denominador == 0:
        return 0.0
        
    # FIX: np.clip forzado para evitar floats como 1.000000002
    return float(np.clip(numerador / denominador, -1.0, 1.0))

def similitud_manhattan(perfil_x: dict, perfil_y: dict) -> float:
    comunes = obtener_peliculas_comunes(perfil_x, perfil_y)
    if not comunes:
        return 0.0
        
    distancia = sum(abs(perfil_x[m]['nota'] - perfil_y[m]['nota']) for m in comunes)
    return 1 / (1 + distancia)

# Diccionario centralizado para llamar a las funciones fácilmente desde otros archivos
METRICAS_SIMILITUD = {
    "coseno":     similitud_coseno,
    "euclidiana": similitud_euclidiana,
    "pearson":    similitud_pearson,
    "manhattan":  similitud_manhattan,
}