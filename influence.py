import numpy as np
from knn import find_similar_users
from prediction import get_recommendations, predict_rating_mean_centered  
from visualization import print_recommendations
from similarity import pearson_similarity

# ─────────────────────────────────────────────
# 7. MODELO DE INFLUENCIA
# ─────────────────────────────────────────────

def _get_influencer_ratings(user_ratings: dict, influencer_id: int) -> dict:
    """Retorna los ratings del influencer desde user_ratings."""
    return user_ratings.get(influencer_id, {})


def _compute_neighbor_influence(
    neighbor_id: int,
    user_ratings: dict,
    influencer_ratings: dict,
) -> float:
    """
    Calcula la similitud Pearson entre un vecino y el influencer.
    Este valor determina cuánto afecta el influencer al vecino:
    - similitud alta  → el vecino es muy cercano al influencer → más afectado
    - similitud baja  → el vecino es lejano al influencer     → poco afectado

    Retorna valor en [-1, 1], clampea a [0, 1] (ignoramos influencia negativa).
    """
    neighbor_ratings = user_ratings.get(neighbor_id, {})
    sim = pearson_similarity(influencer_ratings, neighbor_ratings)
    return max(0.0, sim)  # ignoramos correlación negativa


def _adjust_neighbor_ratings(
    neighbor_ratings: dict,
    influencer_ratings: dict,
    sim_with_influencer: float,
    alpha: float,
) -> dict:
    """
    Ajusta los ratings de un vecino según la influencia del influencer.

    Fórmula por película:
        r_ajustado = (1 - alpha * sim) * r_original
                   +      alpha * sim  * r_influencer

    Solo ajusta películas que TANTO el vecino como el influencer calificaron.
    Las películas que solo calificó el vecino quedan sin cambio.

    Parámetros:
    -----------
    neighbor_ratings      : { movieId: rating } del vecino
    influencer_ratings    : { movieId: rating } del influencer
    sim_with_influencer   : similitud Pearson entre vecino e influencer [0,1]
    alpha                 : fuerza global de influencia definida por el usuario

    Retorna:
    --------
    Nuevo diccionario de ratings ajustados.
    """
    adjusted = dict(neighbor_ratings)  # copia para no mutar el original

    influence_weight = alpha * sim_with_influencer  # peso efectivo de influencia

    for movie_id, original_rating in neighbor_ratings.items():
        if movie_id in influencer_ratings:
            influencer_rating = influencer_ratings[movie_id]

            # Interpolación lineal entre rating original y rating del influencer
            adjusted[movie_id] = (
                (1 - influence_weight) * original_rating
                + influence_weight     * influencer_rating
            )

    return adjusted


def _boost_influencer_similarity(
    neighbors: list[tuple],
    influencer_id: int,
    boost_factor: float,
) -> list[tuple]:
    """
    Aumenta el peso del influencer en el pool de vecinos
    multiplicando su similitud por boost_factor.

    Si el influencer no está en los vecinos, lo agrega con
    similitud = 1.0 * boost_factor (máxima confianza).

    Parámetros:
    -----------
    neighbors    : lista original de (userId, sim, common)
    influencer_id: ID del influencer
    boost_factor : multiplicador de similitud (ej. 1.5 = 50% más peso)

    Retorna:
    --------
    Lista de vecinos con el influencer boosteado, re-ordenada.
    """
    boosted = []
    influencer_found = False

    for user_id, sim, common in neighbors:
        if user_id == influencer_id:
            boosted.append((user_id, round(sim * boost_factor, 4), common))
            influencer_found = True
        else:
            boosted.append((user_id, sim, common))

    # Si el influencer no estaba en los K vecinos, lo insertamos
    if not influencer_found:
        boosted.append((influencer_id, round(1.0 * boost_factor, 4), 0))

    # Re-ordenar por similitud descendente
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def compute_influenced_ratings(
    neighbors: list[tuple],
    user_ratings: dict,
    influencer_id: int,
    alpha: float,
) -> dict:
    """
    Construye un user_ratings modificado donde cada vecino
    tiene sus ratings ajustados por la influencia del influencer.

    Solo modifica los vecinos en memoria, no toca el dataset original.

    Retorna:
    --------
    Copia de user_ratings con ratings ajustados para los vecinos.
    """
    influencer_ratings = _get_influencer_ratings(user_ratings, influencer_id)

    # Copia superficial del dict principal (no queremos mutar el original)
    influenced_ratings = dict(user_ratings)

    for neighbor_id, _, _ in neighbors:
        if neighbor_id == influencer_id:
            continue

        sim_with_influencer = _compute_neighbor_influence(
            neighbor_id, user_ratings, influencer_ratings
        )

        # Solo ajustar si hay alguna similitud real con el influencer
        if sim_with_influencer > 0:
            adjusted = _adjust_neighbor_ratings(
                neighbor_ratings       = user_ratings[neighbor_id],
                influencer_ratings     = influencer_ratings,
                sim_with_influencer    = sim_with_influencer,
                alpha                  = alpha,
            )
            influenced_ratings[neighbor_id] = adjusted

    return influenced_ratings


def run_influence_report(
    target_user: int,
    user_ratings: dict,
    movie_titles: dict,
    influencer_id: int,
    alpha: float       = 0.3,
    boost_factor: float = 1.5,
    k: int             = 20,
    min_common: int    = 5,
    top_n: int         = 10,
    threshold: float   = 3.0,
):
    """
    Genera el reporte comparativo:
        - Recomendaciones SIN influencer (baseline)
        - Recomendaciones CON influencer (modelo influenciado)
        - Diferencias: películas que subieron, bajaron o son nuevas

    Parámetros:
    -----------
    target_user  : usuario para quien se generan recomendaciones
    influencer_id: ID del influencer ya cargado en user_ratings
    alpha        : fuerza de influencia [0, 1]
    boost_factor : multiplicador de similitud del influencer (ej. 1.5)
    """

    print(f"\n{'═'*65}")
    print(f"  REPORTE DE INFLUENCIA")
    print(f"  Usuario objetivo : {target_user}")
    print(f"  Influencer ID    : {influencer_id}")
    print(f"  Alpha            : {alpha}  |  Boost factor: {boost_factor}")
    print(f"{'═'*65}")

    # ── BASELINE: sin influencer ──────────────────────────────
    print("\n  [1/2] Calculando recomendaciones BASE (sin influencer)...")
    neighbors_base = find_similar_users(
        target_user  = target_user,
        user_ratings = user_ratings,
        metric       = "pearson",
        k            = k,
        min_common   = min_common,
    )
    recs_base = get_recommendations(
        target_user  = target_user,
        neighbors    = neighbors_base,
        user_ratings = user_ratings,
        movie_titles = movie_titles,
        threshold    = threshold,
        top_n        = top_n,
    )

    # ── CON INFLUENCER ────────────────────────────────────────
    print("  [2/2] Calculando recomendaciones CON influencer...")

    # Paso 1: boost del influencer en el pool de vecinos
    neighbors_boosted = _boost_influencer_similarity(
        neighbors     = neighbors_base,
        influencer_id = influencer_id,
        boost_factor  = boost_factor,
    )

    # Paso 2: ratings de vecinos ajustados por influencia
    influenced_ratings = compute_influenced_ratings(
        neighbors      = neighbors_boosted,
        user_ratings   = user_ratings,
        influencer_id  = influencer_id,
        alpha          = alpha,
    )

    # Paso 3: recomendaciones con el dataset influenciado
    recs_influenced = get_recommendations(
        target_user  = target_user,
        neighbors    = neighbors_boosted,
        user_ratings = influenced_ratings,
        movie_titles = movie_titles,
        threshold    = threshold,
        top_n        = top_n,
    )

    # ── REPORTE COMPARATIVO ───────────────────────────────────
    _print_comparison_report(recs_base, recs_influenced, alpha, boost_factor)


def _print_comparison_report(
    recs_base: list[tuple],
    recs_influenced: list[tuple],
    alpha: float,
    boost_factor: float,
):
    """
    Imprime tabla comparativa lado a lado y resalta diferencias.

    Clasifica cada película en:
        ▲  subió su predicción
        ▼  bajó su predicción
        ★  nueva (no estaba en el baseline)
        ✖  desapareció (estaba en baseline pero ya no)
    """
    base_dict       = {title: pred for title, pred in recs_base}
    influenced_dict = {title: pred for title, pred in recs_influenced}
    all_movies      = sorted(
        set(base_dict) | set(influenced_dict),
        key=lambda t: influenced_dict.get(t, 0),
        reverse=True,
    )

    print(f"\n{'─'*75}")
    print(f"  {'Película':<35} {'Base':>7}  {'Influenciado':>12}  {'Δ':>7}  Estado")
    print(f"  {'─'*70}")

    for title in all_movies:
        base_val = base_dict.get(title)
        inf_val  = influenced_dict.get(title)

        base_str = f"{base_val:.2f} ★" if base_val is not None else "  —   "
        inf_str  = f"{inf_val:.2f} ★"  if inf_val  is not None else "  —   "

        if base_val is None:
            delta_str = "  —   "
            estado    = "★ NUEVA"
        elif inf_val is None:
            delta_str = "  —   "
            estado    = "✖ SALIÓ"
        else:
            delta = inf_val - base_val
            delta_str = f"{delta:+.2f}"
            if delta > 0.05:
                estado = "▲ subió"
            elif delta < -0.05:
                estado = "▼ bajó"
            else:
                estado = "  estable"

        print(f"  {str(title)[:34]:<35} {base_str:>7}  {inf_str:>12}  {delta_str:>7}  {estado}")

    print(f"{'─'*75}")

    # Resumen
    nuevas    = sum(1 for t in all_movies if t not in base_dict)
    salieron  = sum(1 for t in all_movies if t not in influenced_dict)
    subieron  = sum(1 for t in all_movies if t in base_dict and t in influenced_dict
                    and influenced_dict[t] - base_dict[t] > 0.05)
    bajaron   = sum(1 for t in all_movies if t in base_dict and t in influenced_dict
                    and influenced_dict[t] - base_dict[t] < -0.05)

    print(f"\n  RESUMEN  →  ▲ {subieron} subieron  |  ▼ {bajaron} bajaron  "
          f"|  ★ {nuevas} nuevas  |  ✖ {salieron} salieron")
    print(f"{'═'*65}\n")