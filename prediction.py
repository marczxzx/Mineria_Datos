import numpy as np
from knn import encontrar_usuarios_similares

# 4. PREDICCIÓN (CON FILTRO DE SOPORTE)

def predecir_nota_centrada_media(
    usuario_objetivo: int,
    id_pelicula: int,
    historial_global: dict,
    vecinos: list[tuple]
) -> tuple[float, int]:
    """
    Retorna tupla: (predicción_final, cantidad_de_vecinos_que_la_vieron)
    """
    if not vecinos:
        return 0.0, 0

    # FIX CRÍTICO: Extraemos específicamente las 'notas' del diccionario para sacar el promedio
    notas_objetivo = [datos['nota'] for datos in historial_global[usuario_objetivo].values()]
    media_objetivo = np.mean(notas_objetivo)
    
    numerador = 0.0
    denominador = 0.0
    conteo_soporte = 0

    for id_vecino, similitud, _ in vecinos:
        if id_pelicula in historial_global[id_vecino]:
            conteo_soporte += 1
            
            # Calculamos la media de ese vecino
            notas_vecino = [datos['nota'] for datos in historial_global[id_vecino].values()]
            media_vecino = np.mean(notas_vecino)
            
            # Extraemos la nota específica que le dio a esta película
            nota_vecino = historial_global[id_vecino][id_pelicula]['nota']
            
            numerador += similitud * (nota_vecino - media_vecino)
            denominador += abs(similitud)

    # Si ningún vecino la vio, no forzamos un falso promedio, descartamos la peli.
    if denominador == 0 or conteo_soporte == 0:
        return 0.0, 0

    prediccion = media_objetivo + (numerador / denominador)
    return float(np.clip(prediccion, 1.0, 5.0)), conteo_soporte


def generar_recomendaciones(
    usuario_objetivo: int,
    historial_global: dict,
    catalogo: dict,
    k_vecinos: int = 30,
    min_comunes: int = 5,
    metrica: str = "pearson",
    limite_top: int = 10,
    min_soporte: int = 2  # Filtro para evitar anomalías (Películas con 1 solo voto de 5.0)
) -> list[dict]:
    
    # 1. Obtenemos los mejores vecinos usando nuestra nueva función importada
    vecinos = encontrar_usuarios_similares(
        usuario_objetivo, historial_global, metrica=metrica, k_vecinos=k_vecinos, min_comunes=min_comunes
    )

    if not vecinos:
        print("No se encontraron vecinos válidos.")
        return []

    peliculas_vistas = set(historial_global[usuario_objetivo].keys())
    
    # OPTIMIZACIÓN EXTREMA: Recopilar SOLO las películas que vieron los VECINOS
    candidatas = set()
    for id_vecino, _, _ in vecinos:
        candidatas.update(historial_global[id_vecino].keys())
        
    candidatas_no_vistas = candidatas - peliculas_vistas

    predicciones = []
    for id_pelicula in candidatas_no_vistas:
        pred, soporte = predecir_nota_centrada_media(
            usuario_objetivo, id_pelicula, historial_global, vecinos
        )
        
        # Exigir nota válida y respaldo mínimo de la comunidad
        if pred > 0 and soporte >= min_soporte:
            # Extraemos el título de nuestro nuevo diccionario de catálogo (o mostramos el ID si falla)
            titulo = catalogo.get(id_pelicula, {}).get('titulo', f"ID: {id_pelicula}")
            
            predicciones.append({
                "titulo": titulo,
                "prediccion": round(pred, 2), # Redondeo SOLO para la vista del usuario
                "soporte": soporte
            })

    # Ordenamos de mayor a menor predicción
    predicciones.sort(key=lambda x: x["prediccion"], reverse=True)
    return predicciones[:limite_top]