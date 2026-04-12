import numpy as np

def predict_ratings_for_user(
    target_user: int,
    user_ratings: dict,
    neighbors: list[tuple],
    min_neighbors_for_pred: int = 3
) -> dict:
    """
    Predice calificaciones para películas NO vistas por el usuario objetivo.
    Fórmula estándar de Filtrado Colaborativo basado en Usuario:
        r̂_u,i = ū + [ Σ sim(u,v)·(r_v,i - v̄) ] / [ Σ |sim(u,v)| ]
    
    Parámetros:
    -----------
    min_neighbors_for_pred : int
        Mínimo de vecinos que deben haber calificado una película para
        generar una predicción confiable. Evita predicciones basadas en 1-2 usuarios.
    
    Retorna:
    --------
    dict { movieId: predicted_rating }
    """
    if target_user not in user_ratings:
        raise ValueError("Usuario no encontrado en el dataset.")
    if not neighbors:
        return {}

    target_rated = set(user_ratings[target_user].keys())
    target_avg = np.mean(list(user_ratings[target_user].values()))

    # 1. Recopilar candidatas: todas las películas de los vecinos
    candidate_movies = set()
    for neighbor_id, _, _ in neighbors:
        candidate_movies.update(user_ratings.get(neighbor_id, {}).keys())

    # 2. Excluir las que el usuario YA calificó
    candidate_movies -= target_rated

    predictions = {}
    for movie in candidate_movies:
        num, den = 0.0, 0.0
        neighbors_considered = 0

        for neighbor_id, sim, _ in neighbors:
            if movie in user_ratings.get(neighbor_id, {}):
                neighbor_avg = np.mean(list(user_ratings[neighbor_id].values()))
                # Contribución del vecino (centrada en su media)
                num += sim * (user_ratings[neighbor_id][movie] - neighbor_avg)
                den += abs(sim)
                neighbors_considered += 1

        # 3. Validar soporte mínimo y calcular predicción
        if den > 0 and neighbors_considered >= min_neighbors_for_pred:
            pred = target_avg + (num / den)
            # Clip para mantener en rango realista del dataset [1.0, 5.0]
            predictions[movie] = float(np.clip(pred, 1.0, 5.0))

    return predictions


def get_recommendations(
    target_user: int,
    user_ratings: dict,
    neighbors: list[tuple],
    threshold: float = 3.0,
    top_n: int = 10,
    min_neighbors_for_pred: int = 3,
    movie_titles: dict = None
) -> list[tuple]:
    """
    Genera recomendaciones filtrando por umbral y ordenando por predicción.
    
    Retorna:
    --------
    lista de tuplas [(movieId o título, predicción), ...]
    """
    preds = predict_ratings_for_user(
        target_user, user_ratings, neighbors, min_neighbors_for_pred
    )
    
    # Filtrar por umbral (> threshold)
    filtered = [(mid, pred) for mid, pred in preds.items() if pred > threshold]
    
    # Ordenar descendente por predicción
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Tomar top-N
    top_recs = filtered[:top_n]
    
    # Mapear a títulos si se proporciona el diccionario
    if movie_titles:
        return [(movie_titles.get(mid, f"Desconocida ({mid})"), round(pred, 2)) for mid, pred in top_recs]
    
    return [(mid, round(pred, 2)) for mid, pred in top_recs]