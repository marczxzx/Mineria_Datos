import os
import pandas as pd
from prediction import get_recommendations
from knn import find_similar_users
from visualization import print_results, print_recommendations

# ─────────────────────────────────────────────
# 6. CRUD INFLUENCER
# ─────────────────────────────────────────────

INFLUENCER_PATH = "influencer.csv"


# ── Utilidades de persistencia ────────────────────────────

def _load_influencer_csv() -> pd.DataFrame:
    """Carga el CSV del influencer si existe, si no retorna DataFrame vacío."""
    if os.path.exists(INFLUENCER_PATH):
        return pd.read_csv(INFLUENCER_PATH)
    return pd.DataFrame(columns=["userId", "movieId", "rating"])


def _save_influencer_csv(df: pd.DataFrame):
    """Guarda el DataFrame en el CSV del influencer."""
    df.to_csv(INFLUENCER_PATH, index=False)


def _get_influencer_id(user_ratings: dict) -> int | None:
    """
    Retorna el userId del influencer si existe en el CSV,
    None si aún no hay influencer creado.
    """
    df = _load_influencer_csv()
    if df.empty:
        return None
    return int(df["userId"].iloc[0])


def _generate_new_id(user_ratings: dict) -> int:
    """Genera el siguiente ID disponible (max existente + 1)."""
    max_existing = max(user_ratings.keys())
    df = _load_influencer_csv()
    if not df.empty:
        max_existing = max(max_existing, int(df["userId"].max()))
    return max_existing + 1


# ── Operaciones CRUD ──────────────────────────────────────

def crear_influencer(user_ratings: dict, movie_titles: dict):
    """
    C — Crea un nuevo influencer con un ID autogenerado.
    No se ingresan ratings manualmente; se agregan luego con agregar_rating.
    """
    existing_id = _get_influencer_id(user_ratings)
    if existing_id is not None:
        print(f"\n⚠ Ya existe un influencer (ID {existing_id}). Elimínalo primero para crear uno nuevo.\n")
        return

    nombre = input("  Nombre del influencer: ").strip()
    if not nombre:
        print("  Nombre no puede estar vacío.\n")
        return

    new_id = _generate_new_id(user_ratings)

    # Guardar en CSV con una fila de metadata (nombre en movieId=-1)
    df = _load_influencer_csv()
    meta_row = pd.DataFrame([{"userId": new_id, "movieId": -1, "rating": nombre}])
    df = pd.concat([df, meta_row], ignore_index=True)
    _save_influencer_csv(df)

    print(f"\n✔ Influencer '{nombre}' creado con ID {new_id}.")
    print(f"  Ahora puedes agregar ratings con la opción 3.\n")


def ver_perfil(user_ratings: dict, movie_titles: dict):
    """
    R — Muestra el perfil del influencer y su historial de ratings.
    """
    df = _load_influencer_csv()
    if df.empty:
        print("\n  No hay influencer creado aún.\n")
        return

    influencer_id = int(df["userId"].iloc[0])
    nombre_row = df[df["movieId"] == -1]
    nombre = nombre_row["rating"].values[0] if not nombre_row.empty else "Sin nombre"

    ratings_df = df[df["movieId"] != -1].copy()

    print(f"\n{'─'*55}")
    print(f"  Influencer: {nombre}  |  ID: {influencer_id}")
    print(f"  Total ratings: {len(ratings_df)}")
    print(f"{'─'*55}")

    if ratings_df.empty:
        print("  Aún no tiene películas calificadas.")
    else:
        print(f"  {'#':<4} {'Película':<35} {'Rating'}")
        print(f"  {'─'*50}")
        for i, row in enumerate(ratings_df.itertuples(), 1):
            title = movie_titles.get(int(row.movieId), f"Movie_{row.movieId}")
            print(f"  {i:<4} {title:<35} {float(row.rating):.1f} ★")

    print(f"{'─'*55}\n")


def agregar_o_editar_rating(user_ratings: dict, movie_titles: dict):
    """
    U — Agrega un nuevo rating o edita uno existente para el influencer.
    Busca la película por nombre parcial para facilitar la búsqueda.
    """
    influencer_id = _get_influencer_id(user_ratings)
    if influencer_id is None:
        print("\n  No hay influencer creado. Crea uno primero.\n")
        return

    # Buscar película por nombre
    query = input("  Buscar película (escribe parte del nombre): ").strip().lower()
    matches = [
        (mid, title) for mid, title in movie_titles.items()
        if query in title.lower()
    ]

    if not matches:
        print("  No se encontraron películas con ese nombre.\n")
        return

    print(f"\n  Resultados ({len(matches)} encontrados):")
    for i, (mid, title) in enumerate(matches[:10], 1):
        print(f"  {i}. {title} (ID {mid})")

    try:
        opcion = int(input("\n  Selecciona número de película: ")) - 1
        if opcion < 0 or opcion >= len(matches[:10]):
            print("  Opción inválida.\n")
            return
        selected_id, selected_title = matches[opcion]
    except ValueError:
        print("  Entrada inválida.\n")
        return

    try:
        rating = float(input(f"  Rating para '{selected_title}' (0.5 - 5.0): "))
        if not (0.5 <= rating <= 5.0):
            print("  Rating fuera de rango.\n")
            return
    except ValueError:
        print("  Rating inválido.\n")
        return

    df = _load_influencer_csv()

    # Si ya existe ese movieId para este influencer, editar
    mask = (df["userId"] == influencer_id) & (df["movieId"] == selected_id)
    if mask.any():
        df.loc[mask, "rating"] = rating
        accion = "actualizado"
    else:
        nueva_fila = pd.DataFrame([{"userId": influencer_id, "movieId": selected_id, "rating": rating}])
        df = pd.concat([df, nueva_fila], ignore_index=True)
        accion = "agregado"

    _save_influencer_csv(df)
    print(f"\n  ✔ Rating {accion}: '{selected_title}' → {rating} ★\n")


def eliminar_influencer(user_ratings: dict, movie_titles: dict):
    """
    D — Elimina completamente al influencer y su historial del CSV.
    """
    influencer_id = _get_influencer_id(user_ratings)
    if influencer_id is None:
        print("\n  No hay influencer que eliminar.\n")
        return

    confirmacion = input(f"  ¿Confirmas eliminar al influencer ID {influencer_id}? (s/n): ").strip().lower()
    if confirmacion != "s":
        print("  Operación cancelada.\n")
        return

    # Limpiar CSV completamente
    df = pd.DataFrame(columns=["userId", "movieId", "rating"])
    _save_influencer_csv(df)
    print(f"\n  ✔ Influencer ID {influencer_id} eliminado correctamente.\n")


def recomendar_para_influencer(user_ratings: dict, movie_titles: dict):
    """
    Carga los ratings del influencer al user_ratings en memoria,
    busca sus vecinos y genera recomendaciones.
    """
    influencer_id = _get_influencer_id(user_ratings)
    if influencer_id is None:
        print("\n  No hay influencer creado.\n")
        return

    df = _load_influencer_csv()
    ratings_df = df[df["movieId"] != -1]

    if len(ratings_df) < 5:
        print(f"\n  ⚠ El influencer necesita al menos 5 ratings para recibir recomendaciones. Tiene {len(ratings_df)}.\n")
        return

    # Inyectar ratings del influencer en user_ratings (solo en memoria)
    influencer_ratings = {
        int(row.movieId): float(row.rating)
        for row in ratings_df.itertuples()
    }
    user_ratings[influencer_id] = influencer_ratings

    # Parámetros
    try:
        k         = int(input("  Número de vecinos K (recomendado 20): ") or 20)
        min_common = int(input("  Mínimo de películas en común (recomendado 5): ") or 5)
        top_n     = int(input("  ¿Cuántas recomendaciones quieres ver? (recomendado 10): ") or 10)
    except ValueError:
        print("  Entrada inválida.\n")
        user_ratings.pop(influencer_id, None)
        return

    print("\n  Buscando vecinos con Pearson...")
    neighbors = find_similar_users(
        target_user=influencer_id,
        user_ratings=user_ratings,
        metric="pearson",
        k=k,
        min_common=min_common,
    )

    if not neighbors:
        print("  No se encontraron vecinos suficientes.\n")
        user_ratings.pop(influencer_id, None)
        return

    print_results(influencer_id, neighbors, "pearson")

    recs = get_recommendations(
        target_user=influencer_id,
        #neighbors=neighbors,
        user_ratings=user_ratings,
        movie_titles=movie_titles,
        k=k,
        min_common=min_common,
        threshold=3.0,
        top_n=top_n,
 
    )

    print_recommendations(influencer_id, recs, "pearson", 3.0)

    # Limpiar memoria (no queremos persistir en user_ratings)
    user_ratings.pop(influencer_id, None)


# ── Menú principal ────────────────────────────────────────

def menu_influencer(user_ratings: dict, movie_titles: dict):
    """Menú interactivo en consola para gestionar el influencer."""

    opciones = {
        "1": ("Crear influencer",                   crear_influencer),
        "2": ("Ver perfil e historial de ratings",  ver_perfil),
        "3": ("Agregar / Editar rating",            agregar_o_editar_rating),
        "4": ("Eliminar influencer",                eliminar_influencer),
        "5": ("Buscar vecinos y recomendar",        recomendar_para_influencer),
        "0": ("Salir",                              None),
    }

    while True:
        print(f"\n{'═'*45}")
        print("   MENÚ INFLUENCER")
        print(f"{'═'*45}")
        for key, (label, _) in opciones.items():
            print(f"   {key}. {label}")
        print(f"{'═'*45}")

        eleccion = input("  Selecciona una opción: ").strip()

        if eleccion == "0":
            print("  Saliendo del menú influencer.\n")
            break
        elif eleccion in opciones:
            _, fn = opciones[eleccion]
            fn(user_ratings, movie_titles)
        else:
            print("  Opción no válida. Intenta de nuevo.\n")