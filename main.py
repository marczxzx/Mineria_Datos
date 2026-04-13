import numpy as np
from data_loader import load_movielens, build_user_ratings, load_movie_titles
from knn import find_similar_users
from prediction import get_recommendations
from visualization import print_results, compare_metrics, print_recommendations

# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    users = {1: {1: 3.5, 2: 2.0, 3: 4.5, 4: 5.0, 5: 1.5, 6: 2.5, 7: 2.0},
         2:{1: 2.0, 2: 3.5, 8: 4.0, 4: 2.0, 5: 3.5, 7: 3.0},
         3: {1: 5.0, 2: 1.0, 8: 1.0, 3: 3.0, 4: 5, 5: 1.0},
         4: {1: 3.0, 2: 4.0, 8: 4.5, 4: 3.0, 5: 4.5, 6: 4.0, 7: 2.0},
         5: {2: 4.0, 8: 1.0, 3: 4.0, 6: 4.0, 7: 1.0},
         6:  {2: 4.5, 8: 4.0, 3: 5.0, 4: 5.0, 5: 4.5, 6: 4.0, 7: 4.0},
         7: {1: 5.0, 2: 2.0, 3: 3.0, 4: 5.0, 5: 4.0, 6: 5.0},
         8: {1: 3.0, 3: 5.0, 4: 4.0, 5: 2.5, 6: 3.0}
        }
    musica = {1:"Blues Traveler",2:"Broken Bells",3:"Norah Jones", 4:"Phoenix",5:"Slightly Stoopid",6:"The Strokes",7:"Vampire Weekend",8:8}

    # ── Configuración ──────────────────────────────
    THRESHOLD   = 3.0 
    MOVIES_PATH = "ml-latest-small/movies.csv"    # Ruta al archivo de MovieLens-Movies 100K
    DATA_PATH   = "ml-latest-small/ratings.csv"   # Ruta al archivo de MovieLens-Ratings 100K
    TARGET_USER = 1         # Usuario del que queremos encontrar vecinos
    K           = 10        # Número de vecinos a encontrar
    MIN_COMMON  = 2
             # Mínimo de películas en común
    # ───────────────────────────────────────────────

    # ─── Cargando Datos ───────────────────────────────────
    print("Cargando dataset MovieLens...")
    df = load_movielens(DATA_PATH)
    print(f"  {len(df):,} calificaciones | {df['userId'].nunique()} usuarios | {df['movieId'].nunique()} películas")

    print("\nConstruyendo estructura de datos...")
    user_ratings = build_user_ratings(df) # { userId: { movieId: rating, ... }, ... }
    movie_titles = load_movie_titles(MOVIES_PATH)
    # ───────────────────────────────────────────────────────

    # ─── Informacion del Usuario objetivo ─────────────────
    target_info = user_ratings[TARGET_USER] # { movieId: rating, ... }
    avg_rating = np.mean(list(target_info.values()))
    print(f"\nUsuario {TARGET_USER}: {len(target_info)} películas calificadas | promedio: {avg_rating:.2f} ★")
    # ──────────────────────────────────────────────────────

    # ── Resultados por cada métrica ─────────────
    metrics_similarity = ["cosine", "euclidean", "pearson", "manhattan"]
    #metrics_similarity = ["pearson"]

    for metric in metrics_similarity:
        results = find_similar_users(
            target_user=TARGET_USER,
            user_ratings=user_ratings,
            metric=metric,
            k=K,
            min_common=MIN_COMMON,
        )
        print_results(TARGET_USER, results, metric)

        # Generar recomendaciones
        recs = get_recommendations(
            target_user=TARGET_USER,
            user_ratings=user_ratings,
            movie_titles=movie_titles,
            k=K,
            min_common=MIN_COMMON,
            metric=metric,  # ← Mejor opción
            top_n=10
        )

        print_recommendations(TARGET_USER, recs, metric, THRESHOLD)

    # for metric in metrics_similarity:
    #     results = find_similar_users(
    #         target_user= 1,
    #         user_ratings=users,
    #         metric=metric,
    #         k=8,
    #         min_common=0,
    #     )
    #     print_results(TARGET_USER, results, metric)
    #     recs = get_recommendations(
    #         target_user=1,
    #         user_ratings=users,
    #         movie_titles=musica,
    #         k=8,
    #         min_common=0,
    #         metric=metric,  # ← Mejor opción
    #         top_n=7
    #     )

    #     print_recommendations(TARGET_USER, recs, metric, THRESHOLD)
    # ── Comparación lado a lado ─────────────────
    # compare_metrics(TARGET_USER, user_ratings, k=5,min=MIN_COMMON)
