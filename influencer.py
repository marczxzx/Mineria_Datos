import random
from collections import Counter
import pandas as pd

# ─────────────────────────────────────────────
# 🤖 ESTRATEGIA DE INFLUENCIA SEMÁNTICA (TAG-BASED)
# ─────────────────────────────────────────────

def descubrir_etiquetas_favoritas(perfil_usuario: dict, df_etiquetas: pd.DataFrame, tope_n: int = 5) -> list:
    """
    Analiza las películas que el usuario calificó alto y extrae los conceptos (tags) 
    que más se repiten. Esto define el 'gusto oculto' del usuario.
    """
    # 1. Obtener IDs de películas que el usuario premió (nota >= 4.0)
    ids_peliculas_amadas = [id_p for id_p, datos in perfil_usuario.items() if datos['nota'] >= 4.0]
    
    # 2. Filtrar los tags de esas películas en el DataFrame
    etiquetas_usuario = df_etiquetas[df_etiquetas['movieId'].isin(ids_peliculas_amadas)]['tag'].tolist()
    
    if not etiquetas_usuario:
        return []
        
    # 3. Contar y retornar los N tags más frecuentes (ej: ['distopia', 'espacio', 'oscuro'])
    conteo = Counter(etiquetas_usuario)
    return [etiqueta for etiqueta, _ in conteo.most_common(tope_n)]

def inyectar_influencer_organico(id_objetivo: int, historial_global: dict, catalogo: dict, df_etiquetas: pd.DataFrame) -> dict:
    """
    Crea un perfil de usuario artificial diseñado para maximizar la similitud de Pearson
    con el objetivo y 'empujar' recomendaciones específicas basadas en etiquetas.
    """
    perfil_objetivo = historial_global.get(id_objetivo, {})
    if not perfil_objetivo:
        return {}

    # --- FASE 1: PERFILAMIENTO ---
    etiquetas_clave = descubrir_etiquetas_favoritas(perfil_objetivo, df_etiquetas)
    
    # Identificamos qué ha visto el usuario para no repetir o para usar de cebo
    vistas_objetivo = set(perfil_objetivo.keys())
    amadas_objetivo = [id_p for id_p, d in perfil_objetivo.items() if d['nota'] >= 4.0]
    odiadas_objetivo = [id_p for id_p, d in perfil_objetivo.items() if d['nota'] <= 2.5]
    
    perfil_bot = {}
    fecha_actual = pd.Timestamp.now()

    # --- FASE 2: EL CEBO (Similitud Matemática) ---
    # El bot califica igual que el usuario sus películas favoritas para forzar un Pearson de 1.0
    muestras_cebo = random.sample(amadas_objetivo, min(5, len(amadas_objetivo)))
    for id_p in muestras_cebo:
        perfil_bot[id_p] = {'nota': 5.0, 'fecha': fecha_actual}
        
    # También odia lo que el usuario odia para reforzar el vínculo
    muestras_odio = random.sample(odiadas_objetivo, min(2, len(odiadas_objetivo)))
    for id_p in muestras_odio:
        perfil_bot[id_p] = {'nota': 1.0, 'fecha': fecha_actual}

    # --- FASE 3: RUIDO (Apariencia Humana) ---
    # Calificamos películas aleatorias con notas mediocres para no parecer un bot perfecto
    todas_ids = list(catalogo.keys())
    candidatas_ruido = [id_p for id_p in todas_ids if id_p not in vistas_objetivo]
    ruido = random.sample(candidatas_ruido, min(10, len(candidatas_ruido)))
    for id_p in ruido:
        perfil_bot[id_p] = {'nota': random.choice([3.0, 3.5, 4.0]), 'fecha': fecha_actual}

    # --- FASE 4: EL PAYLOAD (Influencia Semántica) ---
    # Buscamos películas que el usuario NO ha visto, pero que tienen sus etiquetas favoritas
    if etiquetas_clave:
        # Filtramos películas que tengan esos tags
        cine_recomendado = df_etiquetas[df_etiquetas['tag'].isin(etiquetas_clave)]['movieId'].unique()
        # Filtramos que no las haya visto el objetivo y no estén ya en el bot
        finales = [id_p for id_p in cine_recomendado if id_p not in vistas_objetivo and id_p not in perfil_bot]
        
        # Inyectamos las 3 mejores como "joyas" de 5 estrellas
        for id_p in finales[:3]:
            perfil_bot[id_p] = {'nota': 5.0, 'fecha': fecha_actual}
    else:
        # Si no hay tags, inyectamos 3 al azar como respaldo
        respaldo = random.sample(candidatas_ruido, 3)
        for id_p in respaldo:
            perfil_bot[id_p] = {'nota': 5.0, 'fecha': fecha_actual}

    return perfil_bot