import time
import sys
import pandas as pd
import numpy as np



def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    obj_id = id(obj)
    if obj_id in seen: return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
    return size

class ColectorMetricas:
    def __init__(self, num_registros):
        self.num_registros = num_registros
        self.t_inicio_total = time.time()
        
        # Métricas tipo pizarrón
        self.tiempo_subir_info = 0
        self.capacidad_almacenamiento = 0
        self.tiempo_knn = 0
        self.tiempo_recomendacion = 0
        
        # Métricas de precisión (Hold-out method)
        self.precision = 0.0

    def registrar_carga_datos(self, tiempo_segundos, estructura_datos):
        """Registra tiempo de carga y capacidad de almacenamiento real."""
        self.tiempo_subir_info = tiempo_segundos
        # Calcula tamaño real de la estructura de diccionarios anidados
        self.capacidad_almacenamiento = get_size(estructura_datos)

    def registrar_knn(self, tiempo_segundos):
        self.tiempo_knn = tiempo_segundos

    def registrar_recomendacion(self, tiempo_segundos):
        self.tiempo_recomendacion = tiempo_segundos
        
    def calcular_precision_holdout(self, usuario_id, historial_actual, funcion_prediccion, parametros_knn):
        """
        NUEVA: Métrica de precisión avanzada.
        Oculta el 20% de las películas vistas por el usuario (Hold-out)
        e intenta predecir sus notas para calcular la Precisión de Clasificación.
        """
        vistas = list(historial_actual.get(usuario_id, {}).keys())
        if len(vistas) < 10: return # No hay suficiente datos

        # Dividir datos (Train/Test)
        np.random.shuffle(vistas)
        ocultas = vistas[:max(1, int(len(vistas) * 0.2))] # Ocultamos 20%
        vistas_entrenamiento = vistas[max(1, int(len(vistas) * 0.2)):]
        
        # Creamos historial de entrenamiento sin las ocultas
        historial_entreno = {m: d for m, d in historial_actual[usuario_id].items() if m in vistas_entrenamiento}
        
        exitos = 0
        intentos = 0
        
        for peli_id in ocultas:
            nota_real = historial_actual[usuario_id][peli_id]['nota']
            
            # Intentamos predecir usando el resto de usuarios
            parametros_reales = parametros_knn.copy()
            parametros_reales.update({'usuario_objetivo': usuario_id, 'id_pelicula': peli_id})
            
            intentos += 1
            prediccion, _ = funcion_prediccion(**parametros_reales)
            
            # Precisión de Clasificación: Si la predicción está a +/- 1 estrella de la real
            if abs(prediccion - nota_real) <= 1.0:
                exitos += 1
        
        self.precision = (exitos / intentos) if intentos > 0 else 0.0

    def imprimir_tabla_comparativa_pizarron(self):
        """Replica la tabla final comparativa solicitada."""
        print(f"\n{'═'*90}")
        print(f" 📊 TABLA COMPARATIVA DE COMPLEJIDAD Y RENDIMIENTO ".center(90, "═"))
        print(f"{'═'*90}")
        print(f" {'Cant de Registro':<17} | {'Dist.':<10} | {'Tiempo Subir':<12} | {'Capacidad':<12} | {'Tiempo Recom.':<12} | {'Precisión'}")
        print(f" {'(Calificaciones)':<17} | {'Métrica':<10} | {'Info (s)':<12} | {'Almacena.':<12} | {'dación (s)':<12} | {'Recom.'}")
        print(f"{'─'*90}")
        
        # Formateo de capacidad (MB, KB o Bytes)
        if self.capacidad_almacenamiento > 1024*1024:
            cap_str = f"{self.capacidad_almacenamiento / (1024*1024):.1f} MB"
        elif self.capacidad_almacenamiento > 1024:
            cap_str = f"{self.capacidad_almacenamiento / 1024:.1f} KB"
        else:
            cap_str = f"{self.capacidad_almacenamiento} Bytes"
            
        print(f" {self.num_registros:,<17} | {'Pearson':<10} | {self.tiempo_subir_info:.5f}s     | {cap_str:<12} | {self.tiempo_recomendacion:.5f}s     | {self.precision*100:.1f}%")
        print(f"{'═'*90}\n")