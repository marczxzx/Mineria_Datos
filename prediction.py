# prediction.py
import numpy as np
from knn import find_similar_users

def predict_rating_mean_centered(
    target_user: int,
    movie_id: int,
    user_ratings: dict,
    neighbors: list[tuple]
) -> float:
    """
    Predice la calificación de una película para el usuario objetivo.
    Usa vecinos ponderados y corrección por media centrada.
    """
    if not neighbors:
        return 0.0

    target_mean = np.mean(list(user_ratings[target_user].values()))
    numerator = 0.0
    denominator = 0.0

    for neighbor_id, sim, _ in neighbors:
        if movie_id in user_ratings[neighbor_id]:
            neighbor_mean = np.mean(list(user_ratings[neighbor_id].values()))
            # (calificación_vecino - su_media) * similitud
            numerator += sim * (user_ratings[neighbor_id][movie_id] - neighbor_mean)
            denominator += abs(sim)

    if denominator == 0:
        return round(target_mean, 2)  # Fallback: su propio promedio

    predicted = target_mean + (numerator / denominator)
    return round(np.clip(predicted, 1.0, 5.0), 2)  # Limitar a rango [1, 5]


def get_recommendations(
    target_user: int,
    user_ratings: dict,
    movie_titles: dict,
    neighbors: list[tuple], 
    top_n: int = 10,
    threshold: float = 3.0
) -> list[tuple[str, float]]:
    """
    Genera las top_N recomendaciones para un usuario.
    Retorna: [(título_película, predicción), ...]
    """
    # 1. Obtener vecinos más similares
    # neighbors = find_similar_users(
    #     target_user, user_ratings, metric=metric, k=k, min_common=min_common
    # )

    if not neighbors:
        print("No se encontraron vecinos válidos.")
        return []

    # 2. Identificar películas NO vistas por el usuario objetivo
    target_movies = set(user_ratings[target_user].keys())
    all_movies = set()
    for u_ratings in user_ratings.values():
        all_movies.update(u_ratings.keys())
    unseen_movies = all_movies - target_movies

    # 3. Predecir calificación para cada película no vista
    predictions = []
    for movie_id in unseen_movies:
        pred = predict_rating_mean_centered(target_user, movie_id, user_ratings, neighbors)
        if pred > threshold:  # Solo guardar si hay predicción válida
            title = movie_titles.get(movie_id, f"ID: {movie_id}")
            predictions.append((title, pred))

    # 4. Ordenar por predicción descendente y retornar top_N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]