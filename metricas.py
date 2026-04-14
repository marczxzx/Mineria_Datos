import time
import sys
import numpy as np


def get_size(obj, seen=None):
    """Calcula el tamaño real en memoria de una estructura."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)

    return size


class ColectorMetricas:

    def __init__(self, num_registros, distancia="Pearson"):
        self.num_registros = num_registros
        self.distancia = distancia

        # Tiempo global
        self.t_inicio_total = time.time()

        # Métricas principales
        self.tiempo_subir_info = 0
        self.tiempo_knn = 0
        self.tiempo_recomendacion = 0
        self.capacidad_almacenamiento = 0

        # Precisión
        self.precision = 0.0


    def registrar_carga_datos(self, tiempo_segundos, estructura_datos):
        """Tiempo de carga + memoria usada"""
        self.tiempo_subir_info = tiempo_segundos
        self.capacidad_almacenamiento = get_size(estructura_datos)

    def registrar_knn(self, tiempo_segundos):
        """Tiempo de cálculo de vecinos"""
        self.tiempo_knn = tiempo_segundos

    def registrar_recomendacion(self, tiempo_segundos):
        """Tiempo de generación de recomendaciones"""
        self.tiempo_recomendacion = tiempo_segundos

    def tiempo_total(self):
        """Tiempo total de ejecución"""
        return time.time() - self.t_inicio_total


    def calcular_precision_holdout(self, usuario_id, historial_actual, funcion_prediccion, parametros_knn):
        """
        Oculta el 20% del historial y mide qué tan bien predice.
        """

        vistas = list(historial_actual.get(usuario_id, {}).keys())

        if len(vistas) < 10:
            self.precision = 0.0
            return

        np.random.shuffle(vistas)

        corte = max(1, int(len(vistas) * 0.2))

        ocultas = vistas[:corte]
        vistas_entrenamiento = vistas[corte:]

        historial_entreno = {
            m: d for m, d in historial_actual[usuario_id].items()
            if m in vistas_entrenamiento
        }

        exitos = 0
        intentos = 0

        for peli_id in ocultas:
            nota_real = historial_actual[usuario_id][peli_id]['nota']

            parametros_reales = parametros_knn.copy()
            parametros_reales.update({
                'usuario_objetivo': usuario_id,
                'id_pelicula': peli_id
            })

            intentos += 1

            prediccion, _ = funcion_prediccion(**parametros_reales)

            if prediccion is not None and abs(prediccion - nota_real) <= 1.0:
                exitos += 1

        self.precision = (exitos / intentos) if intentos > 0 else 0.0

    def _formatear_memoria(self):
        if self.capacidad_almacenamiento > 1024 * 1024:
            return f"{self.capacidad_almacenamiento / (1024 * 1024):.2f} MB"
        elif self.capacidad_almacenamiento > 1024:
            return f"{self.capacidad_almacenamiento / 1024:.2f} KB"
        else:
            return f"{self.capacidad_almacenamiento} B"

    def obtener_metricas_dict(self):
        """Devuelve métricas listas para GUI"""
        return {
            "registros": self.num_registros,
            "distancia": self.distancia,
            "tiempo_subida": round(self.tiempo_subir_info, 4),
            "tiempo_knn": round(self.tiempo_knn, 4),
            "tiempo_recomendacion": round(self.tiempo_recomendacion, 4),
            "tiempo_total": round(self.tiempo_total(), 4),
            "memoria": self._formatear_memoria(),
            "precision": f"{self.precision * 100:.2f}%"
        }

    # =========================
    # IMPRESIÓN EN CONSOLA
    # =========================

    def imprimir_tabla(self):
        print(f"\n{'═'*120}")
        print(f" 📊 TABLA COMPLETA DE MÉTRICAS K-NN ".center(120, "═"))
        print(f"{'═'*120}")

        print(f"{'Registros':<12} | {'Distancia':<10} | {'Carga(s)':<10} | {'KNN(s)':<10} | "
              f"{'Recom(s)':<10} | {'Total(s)':<10} | {'Memoria':<12} | {'Precisión'}")

        print(f"{'─'*120}")

        print(f"{self.num_registros:<12} | {self.distancia:<10} | "
              f"{self.tiempo_subir_info:.4f} | {self.tiempo_knn:.4f} | "
              f"{self.tiempo_recomendacion:.4f} | {self.tiempo_total():.4f} | "
              f"{self._formatear_memoria():<12} | {self.precision * 100:.2f}%")