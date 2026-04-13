from similarity import METRICS
from knn import find_similar_users

# ─────────────────────────────────────────────
# 4. UTILIDADES DE VISUALIZACIÓN
# ─────────────────────────────────────────────

def print_results(target_user: int, results: list, metric: str):
    metric_labels = {
        "cosine":    "Similitud Coseno",
        "euclidean": "Similitud Euclidiana",
        "pearson":   "Correlación de Pearson",
        "manhattan":  "Similitud Manhattan",
    }
    label = metric_labels[metric]

    print(f"\n{'─'*60}")
    print(f"  Usuario objetivo: {target_user}  |  Métrica: {label}")
    print(f"{'─'*60}")
    print(f"  {'#':<4} {'Usuario':<12} {label:<30} {'Películas comunes'}")
    print(f"  {'─'*55}")

    for i, (user, sim, common) in enumerate(results, 1):
        if i == 10:
            break
        # bar = "█" * int(abs(sim) * 20)
        sign = "+" if sim >= 0 else "-"
        # print(f"  {i:<4} Usuario {user:<5}  {sign}{abs(sim):.4f}  {bar:<22}  {common} películas")
        print(f"  {i:<4} Usuario {user:<5}  {sign}{abs(sim):.4f}  {common:>22} películas")

    print(f"{'─'*60}\n")


def compare_metrics(target_user: int, user_ratings: dict, k: int = 5,min: int = 3):
    """Compara los top-K vecinos según cada métrica."""
    print(f"\n{'═'*80}")
    print(f"  Comparación de métricas para Usuario {target_user}  (Top {k})")
    print(f"{'═'*80}")

    all_results = {}
    for metric in METRICS:
        results = find_similar_users(target_user, user_ratings, metric=metric, k=k,min_common=min)
        all_results[metric] = [r[0] for r in results]

    print(f"\n  {'#':<4} {'Coseno':<14} {'Euclidiana':<14} {'Pearson':<14} {'Manhattan'}")
    print(f"  {'─'*65}")
    for i in range(k):
        c = f"Usuario {all_results['cosine'][i]}" if i < len(all_results['cosine']) else "—"
        e = f"Usuario {all_results['euclidean'][i]}" if i < len(all_results['euclidean']) else "—"
        p = f"Usuario {all_results['pearson'][i]}" if i < len(all_results['pearson']) else "—"
        m = f"Usuario {all_results['manhattan'][i]}" if i < len(all_results['manhattan']) else "—"
        print(f"  {i+1:<4} {c:<14} {e:<14} {p:<14} {m}")
    print()


def print_recommendations(target_user: int, recs: list[tuple], metric: str, threshold: float):
    """Imprime las recomendaciones finales en formato tabla."""
    print(f"\nTOP RECOMENDACIONES (Métrica: {metric} | Umbral: > {threshold} ★)")
    print(f"{'─'*65}")
    print(f"  {'#':<4} {'Película / ID':<35} {'Predicción':<12} {'Confianza'}")
    print(f"  {'─'*65}")
    for i, (title, pred) in enumerate(recs, 1):
        bar = "█" * int(pred * 4)
        print(f"  {i:<4} {str(title):<35} {pred:.2f} ★  {bar}")
    print(f"{'─'*65}\n")
