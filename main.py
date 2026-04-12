import numpy as np
import pandas as pd
from data_loader import load_movielens, build_user_ratings
from knn import find_similar_users
from prediction import get_recommendations
from visualization import print_results, compare_metrics, print_recommendations

def load_movie_titles(movies_path: str) -> dict:
    """Carga título -> movieId (opcional, para legibilidad)"""
    try:
        df = pd.read_csv(movies_path, sep=",", encoding="latin-1")
        return dict(zip(df["movieId"], df["title"]))
    except FileNotFoundError:
        print("⚠️ No se encontró movies.csv. Se usarán IDs numéricos.")
        return {}
# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Configuración ──────────────────────────
    THRESHOLD   = 3.5 
    MOVIES_PATH = "ml-latest-small/movies.csv"
    DATA_PATH   = "ml-latest-small/ratings.csv"   # Ruta al archivo de MovieLens 100K
    TARGET_USER = 10         # Usuario del que queremos encontrar vecinos
    K           = 10         # Número de vecinos a encontrar
    MIN_COMMON  = 3          # Mínimo de películas en común
    # ───────────────────────────────────────────

    print("Cargando dataset MovieLens...")
    df = load_movielens(DATA_PATH)
    print(f"  {len(df):,} calificaciones | {df['userId'].nunique()} usuarios | {df['movieId'].nunique()} películas")

    print("\nConstruyendo estructura de datos...")
    user_ratings = build_user_ratings(df)
    movie_titles = load_movie_titles(MOVIES_PATH)

    target_info = user_ratings[TARGET_USER]
    avg_rating = np.mean(list(target_info.values()))
    print(f"\nUsuario {TARGET_USER}: {len(target_info)} películas calificadas | promedio: {avg_rating:.2f} ★")

    # ── Resultados por cada métrica ─────────────
    for metric in ["cosine", "euclidean", "pearson", "manhattan"]:
        results = find_similar_users(
            target_user=TARGET_USER,
            user_ratings=user_ratings,
            metric=metric,
            k=K,
            min_common=MIN_COMMON,
        )
        print_results(TARGET_USER, results, metric)

        recs = get_recommendations(
            target_user=TARGET_USER,
            user_ratings=user_ratings,
            neighbors=results,
            threshold=THRESHOLD,
            top_n=10,
            min_neighbors_for_pred=3,
            movie_titles=movie_titles
        )
        
        if recs:
            print_recommendations(TARGET_USER, recs, metric, THRESHOLD)
        else:
            print(f"No se encontraron películas con predicción > {THRESHOLD} usando {metric}.\n")
    # ── Comparación lado a lado ─────────────────
    compare_metrics(TARGET_USER, user_ratings, k=5,min=MIN_COMMON)


    # neighbors = find_similar_users(
    #     target_user=TARGET_USER,
    #     user_ratings=user_ratings,
    #     metric="pearson",
    #     k=K,
    #     min_common=MIN_COMMON
    # )
    # print_results(TARGET_USER, neighbors, "pearson")

    # # 2. Generar recomendaciones
    # print(f"Generando recomendaciones (umbral > {THRESHOLD})...")
    # recommendations = get_recommendations(
    #     target_user=TARGET_USER,
    #     user_ratings=user_ratings,
    #     neighbors=neighbors,
    #     threshold=THRESHOLD,
    #     top_n=10,
    #     min_neighbors_for_pred=3,
    #     movie_titles=movie_titles
    # )

    # # 3. Mostrar resultados
    # print(f"\n{'='*70}")
    # print(f"  TOP {len(recommendations)} RECOMENDACIONES PARA USUARIO {TARGET_USER}")
    # print(f"{'='*70}")
    # print(f"  {'#':<4} {'Película':<45} {'Predicción':<10}")
    # print(f"  {'-'*65}")
    # for i, (title, pred) in enumerate(recommendations, 1):
    #     # Barra de confianza
    #     bar = "█" * int(pred * 4)
    #     print(f"  {i:<4} {title:<45} {pred:.2f} ★  {bar}")
    # print(f"{'='*70}\n")