from similarity import METRICAS_SIMILITUD, obtener_peliculas_comunes

# ─────────────────────────────────────────────
# 3. KNN DESDE CERO (OPTIMIZADO Y EN ESPAÑOL)
# ─────────────────────────────────────────────

def encontrar_usuarios_similares(
    usuario_objetivo: int,
    historial_global: dict,
    metrica: str = "pearson",
    k_vecinos: int = 10,
    min_comunes: int = 0,
) -> list[tuple]:
    """
    Calcula la similitud del usuario objetivo con el resto y retorna los K más similares.
    """
    if usuario_objetivo not in historial_global:
        raise ValueError(f"Usuario {usuario_objetivo} no encontrado en el dataset.")

    # Obtenemos la función matemática correspondiente desde similarity.py
    funcion_similitud = METRICAS_SIMILITUD.get(metrica)
    if funcion_similitud is None:
        raise ValueError(f"Métrica '{metrica}' no válida. Opciones: {list(METRICAS_SIMILITUD.keys())}")

    perfil_objetivo = historial_global[usuario_objetivo]
    similitudes = []

    for otro_usuario, perfil_otro in historial_global.items():
        if otro_usuario == usuario_objetivo:
            continue

        # Verificamos que tengan al menos el mínimo de películas en común (Soporte)
        comunes = obtener_peliculas_comunes(perfil_objetivo, perfil_otro)
        if len(comunes) < min_comunes:
            continue

        # Calculamos el nivel de similitud entre -1.0 y 1.0 (o 0.0 y 1.0)
        similitud = funcion_similitud(perfil_objetivo, perfil_otro)
        
        # Mantenemos la precisión máxima sin redondear prematuramente
        similitudes.append((otro_usuario, similitud, len(comunes)))

    # PASO DEL PIZARRÓN: Ordenamos los usuarios de mayor a menor similitud
    similitudes.sort(key=lambda x: x[1], reverse=True)
    
    # SALIDA: Retornamos únicamente los K vecinos más cercanos
    return similitudes[:k_vecinos]