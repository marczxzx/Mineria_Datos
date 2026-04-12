import numpy as np

# ─────────────────────────────────────────────
# 2. SIMILITUD
# ─────────────────────────────────────────────

def get_common_movies(ratings_u: dict, ratings_v: dict) -> list:
    """
    Devuelve la lista de películas calificadas por AMBOS usuarios.
    """
    return list(set(ratings_u.keys()) & set(ratings_v.keys()))


def cosine_similarity(ratings_u: dict, ratings_v: dict) -> float:
    """
    Similitud del Coseno
    --------------------
    Mide el ángulo entre los vectores de calificaciones.
    
    Fórmula:
        sim(u, v) = (u · v) / (||u|| * ||v||)
    
    Donde:
        u · v   = suma de productos de calificaciones comunes
        ||u||   = norma (longitud) del vector u
        ||v||   = norma (longitud) del vector v

    Rango: [-1, 1]  (con calificaciones positivas: [0, 1])
    1  = idéntica dirección (muy similares)
    0  = ortogonales (sin relación)
    -1 = opuestos

    """
    common = get_common_movies(ratings_u, ratings_v)
    if not common:
        return 0.0

    # Producto punto entre los vectores de calificaciones comunes
    dot_product = sum(ratings_u[m] * ratings_v[m] for m in common)

    # Normas usando TODAS las películas de cada usuario (no solo las comunes)
    # Esto refleja mejor el "tamaño" real del vector de cada usuario
    norm_u = np.sqrt(sum(r ** 2 for r in ratings_u.values()))
    norm_v = np.sqrt(sum(r ** 2 for r in ratings_v.values()))

    if norm_u == 0 or norm_v == 0:
        return 0.0

    return dot_product / (norm_u * norm_v)


def euclidean_similarity(ratings_u: dict, ratings_v: dict) -> float:
    """
    Distancia Euclidiana → convertida a Similitud
    -----------------------------------------------
    Mide la distancia geométrica directa entre los vectores.

    Fórmula de distancia:
        d(u, v) = sqrt( Σ (u_i - v_i)² )  para películas comunes i

    Convertida a similitud:
        sim(u, v) = 1 / (1 + d(u, v))

    Rango de similitud: (0, 1]
    1   = distancia 0 (calificaciones idénticas)
    → 0 = distancia infinita (muy diferentes)
    """
    common = get_common_movies(ratings_u, ratings_v)
    if not common:
        return 0.0

    # Suma de diferencias al cuadrado sobre películas comunes
    squared_diff = sum((ratings_u[m] - ratings_v[m]) ** 2 for m in common)
    distance = np.sqrt(squared_diff)

    # Convertir distancia → similitud en rango (0, 1]
    return 1 / (1 + distance)


def pearson_similarity(ratings_u: dict, ratings_v: dict) -> float:
    """
    Correlación de Pearson
    -----------------------
    Mide la correlación lineal entre las calificaciones, 
    centradas en la media de cada usuario.

    Fórmula:
        sim(u, v) = Σ (u_i - ū)(v_i - v̄)
                    ─────────────────────────────────────────
                    sqrt(Σ(u_i - ū)²) * sqrt(Σ(v_i - v̄)²)

    Donde ū y v̄ son las medias de calificaciones de u y v
    calculadas SOLO sobre las películas comunes.

    Rango: [-1, 1]
    1   = correlación perfecta positiva
    0   = sin correlación
    -1  = correlación perfecta negativa
    """
    common = get_common_movies(ratings_u, ratings_v)
    if len(common) < 2:
        return 0.0

    # Vectores de calificaciones en películas comunes
    vec_u = np.array([ratings_u[m] for m in common])
    vec_v = np.array([ratings_v[m] for m in common])

    # Medias de cada usuario (solo sobre películas comunes)
    mean_u = np.mean(vec_u)
    mean_v = np.mean(vec_v)

    # Centrar: restar la media a cada calificación
    centered_u = vec_u - mean_u
    centered_v = vec_v - mean_v

    # Numerador: covarianza entre los vectores centrados
    numerator = np.dot(centered_u, centered_v)

    # Denominador: producto de normas de vectores centrados
    denom_u = np.sqrt(np.dot(centered_u, centered_u))
    denom_v = np.sqrt(np.dot(centered_v, centered_v))
    denominator = denom_u * denom_v

    if denominator == 0:
        return 0.0

    # Clampear entre -1 y 1 para evitar errores numéricos
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def manhattan_similarity(ratings_u: dict, ratings_v: dict) -> float:
    """
    Distancia Manhattan → convertida a Similitud
    ---------------------------------------------
    También llamada distancia L1 o "city block distance".
    Suma las diferencias absolutas entre calificaciones,
    en lugar de elevarlas al cuadrado como hace Euclidiana.

    Fórmula de distancia:
        d(u, v) = Σ |u_i - v_i|   para películas comunes i

    Convertida a similitud:
        sim(u, v) = 1 / (1 + d(u, v))

    Rango de similitud: (0, 1]
    1   = distancia 0 (calificaciones idénticas)
    → 0 = distancia muy grande (muy diferentes)
    """
    common = get_common_movies(ratings_u, ratings_v)
    if not common:
        return 0.0

    # Suma de diferencias absolutas (no al cuadrado, esa es la clave)
    distance = sum(abs(ratings_u[m] - ratings_v[m]) for m in common)

    # Convertir distancia → similitud en rango (0, 1]
    return 1 / (1 + distance)

METRICS = {
    "cosine":     cosine_similarity,
    "euclidean":  euclidean_similarity,
    "pearson":    pearson_similarity,
    "manhattan":  manhattan_similarity,
}