import numpy as np

# ─────────────────────────────────────────────
# 2. SIMILITUD
# ─────────────────────────────────────────────

def get_common_movies(ratings_x: dict, ratings_y: dict) -> list:
    # Devuelve la lista de películas calificadas por AMBOS usuarios.
    return list(set(ratings_x.keys()) & set(ratings_y.keys()))


def cosine_similarity(ratings_x: dict, ratings_y: dict) -> float:
    """
    Similitud del Coseno
    --------------------
    Mide el ángulo entre los vectores de calificaciones.
    
    Fórmula:
        cos(x, y) =     (x · y)
                    ───────────────
                    (||x|| * ||y||)
    
    Donde:
        x · y   = suma de productos de calificaciones comunes
        ||x||   = norma (longitud) del vector x
        ||y||   = norma (longitud) del vector y

    Rango: [-1, 1]  (con calificaciones positivas: [0, 1])
    1  = idéntica dirección (muy similares)
    0  = ortogonales (sin relación)
    -1 = opuestos
    """
    common = get_common_movies(ratings_x, ratings_y)
    if not common:
        return 0.0

    # Producto punto entre los vectores de calificaciones comunes
    product = sum(ratings_x[m] * ratings_y[m] for m in common)

    # Normas usando TODAS las películas de cada usuario (no solo las comunes)
    # Esto refleja mejor el "tamaño" real del vector de cada usuario
    norm_x = np.sqrt(sum(r ** 2 for r in ratings_x.values()))
    norm_y = np.sqrt(sum(r ** 2 for r in ratings_y.values()))

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return product / (norm_x * norm_y)


def euclidean_similarity(ratings_x: dict, ratings_y: dict) -> float:
    """
    Distancia Euclidiana 
    -----------------------------------------------
    Fórmula de distancia:
        dm(x, y) = sqrt( Σ (x_i - y_i)² )  para películas comunes i

    Convertida a similitud:
        sim(x, y) = 1 / (1 + d(x, y))

    Rango de similitud: (0, 1]
    1   = distancia 0 (calificaciones idénticas)
    → 0 = distancia infinita (muy diferentes)
    """
    common = get_common_movies(ratings_x, ratings_y)
    if not common:
        return 0.0

    # Suma de diferencias al cuadrado sobre películas comunes
    squared_diff = sum((ratings_x[m] - ratings_y[m]) ** 2 for m in common)
    distance = np.sqrt(squared_diff)

    # Convertir distancia → similitud en rango (0, 1]
    return 1 / (1 + distance)
    #return -distance


def pearson_similarity(ratings_x: dict, ratings_y: dict) -> float:
    """
    Correlación de Pearson
    -----------------------
    Mide la correlación lineal entre las calificaciones, 
    centradas en la media de cada usuario.

    Fórmula:
        sim(u, v) = Σ (x_i - x)(y_i - y)
                    ─────────────────────────────────────────
                    sqrt(Σ(x_i - x)²) * sqrt(Σ(y_i - y)²)

    Donde X y Y son las medias de calificaciones de X y Y
    calculadas SOLO sobre las películas comunes.

    Rango: [-1, 1]
    1   = correlación perfecta positiva
    0   = sin correlación
    -1  = correlación perfecta negativa
    """
    common = get_common_movies(ratings_x, ratings_y)
    if len(common) < 2:
        return 0.0

    # Vectores de calificaciones en películas comunes
    vec_x = np.array([ratings_x[m] for m in common])
    vec_y = np.array([ratings_y[m] for m in common])

    # Medias de cada usuario (solo sobre películas comunes)
    avg_x = np.mean(vec_x)
    avg_y = np.mean(vec_y)

    # Centrar: restar la media a cada calificación
    centered_x = vec_x - avg_x
    centered_y = vec_y - avg_y

    # Numerador: covarianza entre los vectores centrados
    numerator = np.dot(centered_x, centered_y)

    # Denominador: producto de normas de vectores centrados
    denom_x = np.sqrt(np.dot(centered_x, centered_x))
    denom_y = np.sqrt(np.dot(centered_y, centered_y))
    denominator = denom_x * denom_y

    if denominator == 0:
        return 0.0

    # Clampear entre -1 y 1 para evitar errores numéricos
    #return float(np.clip(numerator / denominator, -1.0, 1.0))
    return float(numerator / denominator)


def manhattan_similarity(ratings_x: dict, ratings_y: dict) -> float:
    """
    Distancia Manhattan 
    ---------------------------------------------
    También llamada distancia L1 o "city block distance".
    Suma las diferencias absolutas entre calificaciones,
    en lugar de elevarlas al cuadrado como hace Euclidiana.

    Fórmula de distancia:
        d(x, y) = Σ |x_i - y_i|   para películas comunes i


    Rango de similitud: (0, 1]
    1   = distancia 0 (calificaciones idénticas)
    → 0 = distancia muy grande (muy diferentes)
    """
    common = get_common_movies(ratings_x, ratings_y)
    if not common:
        return 0.0

    # Suma de diferencias absolutas (no al cuadrado, esa es la clave)
    distance = sum(abs(ratings_x[m] - ratings_y[m]) for m in common)

    # Convertir distancia → similitud en rango (0, 1]
    return 1 / (1 + distance)
    #return -distance

METRICS = {
    "cosine":     cosine_similarity,
    "euclidean":  euclidean_similarity,
    "pearson":    pearson_similarity,
    "manhattan":  manhattan_similarity,
}