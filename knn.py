from similarity import METRICS, get_common_movies

# ─────────────────────────────────────────────
# 3. KNN DESDE CERO
# ─────────────────────────────────────────────

def find_similar_users(
    target_user: int,
    user_ratings: dict,
    metric: str = "pearson",
    k: int = 10,
    min_common: int = 0,
) -> list[tuple]:
    """
    Encuentra los K usuarios más similares al usuario objetivo.
    Implementación manual de KNN.

    Parámetros:
    -----------
    target_user : int
        ID del usuario del que queremos encontrar vecinos.
    user_ratings : dict
        Diccionario { userId: { movieId: rating } }
    metric : str
        Métrica a usar: 'cosine', 'euclidean' o 'pearson'
    k : int
        Número de vecinos a retornar
    min_common : int
        Mínimo de películas en común para considerar el par válido.
        Pares con menos películas comunes son poco confiables.

    Retorna:
    --------
    Lista de tuplas (userId, similitud, películas_comunes)
    ordenada de mayor a menor similitud.
    """
    if target_user not in user_ratings:
        raise ValueError(f"Usuario {target_user} no encontrado en el dataset.")

    sim_fn = METRICS.get(metric)
    if sim_fn is None:
        raise ValueError(f"Métrica '{metric}' no válida. Opciones: {list(METRICS.keys())}")

    target_ratings = user_ratings[target_user]
    similarities = []

    for other_user, other_ratings in user_ratings.items():
        if other_user == target_user:
            continue

        # Filtro mínimo de películas comunes
        common = get_common_movies(target_ratings, other_ratings)
        if len(common) < min_common:
            continue

        sim = sim_fn(target_ratings, other_ratings)
        similarities.append((other_user, round(sim, 4), len(common)))

    # Ordenar por similitud descendente (KNN: los K más cercanos)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]
