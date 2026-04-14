import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# MÓDULO VISUAL PROFESIONAL (PULIDO Y EN ESPAÑOL)
# ─────────────────────────────────────────────

# --- Estilos Globales para consistencia ---
TITULO_FONT = {'fontweight': 'bold', 'fontsize': 12}
ETIQUETA_FONT = {'fontweight': 'bold', 'fontsize': 10}
COLOR_BARRA_SIM = '#2ecc71' # Verde plano
COLOR_BARRA_REC = '#4A90E2' # Azul para recomendaciones

def graficar_perfil_pastel(usuario_objetivo: int, historial_global: dict, catalogo: dict):
    """
    Diagrama de pastel interactivo (Pie Chart).
    Permite visualizar la distribución de calificaciones del usuario.
    [ACTUALIZADO - COLORES PASTEL BONITOS con Set3 colormap]
    """
    perfil_objetivo = historial_global.get(usuario_objetivo, {})
    if not perfil_objetivo:
        print(f"Sin datos para graficar perfil del usuario {usuario_objetivo}")
        return
 
    # 1. Agrupar películas por nota
    grupos_notas = {}
    for id_p, datos in perfil_objetivo.items():
        nota = datos['nota']
        if nota not in grupos_notas:
            grupos_notas[nota] = []
 
        titulo = catalogo.get(id_p, {}).get('titulo', f"Película {id_p}")
        grupos_notas[nota].append(titulo)
 
    # 2. Preparar datos
    notas = sorted(grupos_notas.keys())
    cantidades = [len(grupos_notas[nota]) for nota in notas]
    etiquetas = [f"{nota} ★" for nota in notas]
 
    # --- SOLUCIÓN DE COLORES ---
    # Usamos 'Set3', mapa de color específico para categorías con colores pastel únicos
    num_categorias = len(notas)
    cmap = plt.get_cmap('Set3')
    colores_pastel = [cmap(i / num_categorias) for i in range(num_categorias)]
 
    # 3. Configurar figura profesional
    fig, ax = plt.subplots(figsize=(8, 6), num=f"Perfil Usuario {usuario_objetivo}")
    fig.patch.set_facecolor('white')  # Fondo limpio
 
    wedges, texts, autotexts = ax.pie(
        cantidades, labels=etiquetas, autopct='%1.1f%%',
        startangle=140, colors=colores_pastel,
        wedgeprops={'edgecolor': 'gray', 'linewidth': 0.5},
        textprops={'fontsize': 10}
    )
 
    ax.set_title(f'Perfil del Usuario {usuario_objetivo}\n(Distribución de sus {len(perfil_objetivo)} calificaciones)', **TITULO_FONT)
    plt.tight_layout()
 
    # 4. Tooltip (Recuadro interactivo al pasar el mouse)
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
        bbox=dict(boxstyle="round4,pad=0.5", fc="#FAFAFA", ec="black", lw=1, alpha=0.98),
        fontsize=9
    )
    annot.set_visible(False)
 
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for i, wedge in enumerate(wedges):
                cont, ind = wedge.contains(event)
                if cont:
                    nota = notas[i]
                    ejemplos = grupos_notas[nota][:5]
 
                    texto = f"Calificación: {nota} ★\n"
                    texto += f"Representa: {cantidades[i]} películas vistas\n"
                    texto += "─" * 35 + "\n"
                    texto += "Ejemplos en esta categoría:\n"
                    for ej in ejemplos:
                        ej_corto = str(ej)[:40] + "..." if len(str(ej)) > 40 else str(ej)
                        texto += f" • {ej_corto}\n"
 
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(texto.strip())
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()
 
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()
 
def graficar_similitud_vecinos(usuario_objetivo: int, vecinos: list, metrica: str):
    if not vecinos:
        print("No hay vecinos válidos para graficar similitud.")
        return

    # 1. Extraer datos del K-NN (IDs y Similitudes puras)
    vecinos_ids = [str(res[0]) for res in vecinos]
    similitudes = [res[1] for res in vecinos]
    k_vecinos = len(vecinos)

    # 2. Configurar la métrica para el título profesional
    nombres_metricas = {
        "coseno":    "Coseno",
        "euclidiana": "Euclidiana (1/(1+d))",
        "pearson":   "Pearson (Correlación)",
        "manhattan": "Manhattan (1/(1+d))",
    }
    label_metrica = nombres_metricas.get(metrica, metrica.capitalize())

    # 3. Crear gráfica (aspecto profesional con fondo blanco y grilla)
    fig, ax = plt.subplots(figsize=(10, 5), num=f"K-NN Usuario {usuario_objetivo}")
    fig.patch.set_facecolor('white') 
    
    # Crear barras verticales
    bars = ax.bar(vecinos_ids, similitudes, color=COLOR_BARRA_SIM, edgecolor='black', alpha=0.9)
    
    # Títulos y Ejes Profesional
    ax.set_title(f'Resultados K-NN ({k_vecinos} vecinos) - Métrica: {label_metrica}', **TITULO_FONT)
    ax.set_ylabel('Grado de Similitud (-1 a 1)', **ETIQUETA_FONT)
    ax.set_xlabel('ID de Usuario (Vecinos)', **ETIQUETA_FONT)
    
    # ───────────────────────────────────────────────────
    # 🚀 FIX: AÑADIR MARGEN SUPERIOR (SOLUCIÓN)
    # ───────────────────────────────────────────────────
    # Cambiamos el límite superior de 1.0 a 1.1 para dar "aire" a los números.
    ax.set_ylim(-1.0, 1.1) 
    
    ticks_y = np.arange(-1.0, 1.01, 0.25)
    ax.set_yticks(ticks_y)
    
    ax.grid(axis='y', linestyle='--', color='lightgray', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1) 
    ax.set_axisbelow(True) 
    
    # Añadir valores exactos sobre las barras
    for bar in bars:
        height = bar.get_height()
        va_pos = 'bottom' if height >= 0 else 'top'
        xy_pos = (bar.get_x() + bar.get_width() / 2, height)
        xy_text = (0, 3) if height >= 0 else (0, -12)
        
        ax.annotate(f'{height:.3f}', 
                    xy=xy_pos,
                    xytext=xy_text,
                    textcoords="offset points",
                    ha='center', va=va_pos, fontsize=8, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.show()


def graficar_top_recomendaciones(usuario_objetivo: int, recomendaciones: list, metrica: str):
    """
    Gráfico de barras horizontales profesional para Top de Recomendaciones.
    """
    if not recomendaciones: return

    # Preparar datos (acortar títulos largos)
    titulos = [r['titulo'][:40] + "..." if len(r['titulo']) > 40 else r['titulo'] for r in recomendaciones]
    predicciones = [r["prediccion"] for r in recomendaciones]

    # Invertir orden para que #1 esté arriba en la gráfica
    titulos = titulos[::-1]
    predicciones = predicciones[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), num=f"Recomendaciones Usuario {usuario_objetivo}")
    
    # Barras horizontales
    barras = ax.barh(titulos, predicciones, color=COLOR_BARRA_REC, edgecolor='gray', alpha=0.85)
    
    # Ejes profesionales
    ax.set_xlabel('Puntuación Predicha (Estrellas)', **ETIQUETA_FONT)
    ax.set_title(f'Top Recomendaciones para Usuario {usuario_objetivo}\n(Métrica: {metrica.capitalize()})', **TITULO_FONT)
    ax.set_xlim(0, 5.5) 

    # Score exacto al final de la barra
    for barra in barras:
        ancho = barra.get_width()
        ax.text(ancho + 0.05, barra.get_y() + barra.get_height()/2,
                 f'{ancho:.2f} ★', va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# NUEVAS FUNCIONES DE CONSOLA (TABLAS SE MANTIENEN IGUAL)
# ─────────────────────────────────────────────
def imprimir_tabla_recomendaciones(recomendaciones: list, metrica: str):
    """
    Imprime recomendaciones en consola con formato de tabla estructurada.
    """
    print(f"\n{'═'*85}")
    print(f" 🏆 TOP RECOMENDACIONES - MÉTRICA: {metrica.upper()} ".center(85, "═"))
    print(f"{'═'*85}")
    print(f" {'#':<3} | {'Película (ID)':<48} | {'Predicción':<10} | {'Soporte'}")
    print(f"{'─'*85}")
    
    if not recomendaciones:
        print(" Ninguna película superó los filtros mínimos.")
    else:
        for i, r in enumerate(recomendaciones, 1):
            # Acortamos títulos largos para no romper la tabla
            titulo_corto = r['titulo'][:45] + "..." if len(r['titulo']) > 48 else r['titulo']
            print(f" {i:<3} | {titulo_corto:<48} | {r['prediccion']:.2f} ★   | {r['soporte']} vecinos")
    print(f"{'═'*85}\n")

def graficar_ataque_influencer(perfil_bot: dict, vistas_usuario: set, catalogo: dict):
    """
    Gráfica exclusiva que muestra el 'Payload' del influencer: 
    Las películas que calificó con 5.0 y que el usuario OBJETIVO aún NO ha visto.
    """
    # Filtramos: Notas de 5.0 del bot que el usuario no conozca
    payload_ids = [id_p for id_p, datos in perfil_bot.items() if datos['nota'] == 5.0 and id_p not in vistas_usuario]
    
    if not payload_ids:
        print("El bot no inyectó ninguna película nueva.")
        return

    # Extraemos títulos limitando a las 3 principales que suele inyectar el bot
    payload_ids = payload_ids[:3] 
    titulos = [catalogo.get(id_p, {}).get('titulo', f"ID {id_p}")[:35] + "..." for id_p in payload_ids]
    notas = [5.0] * len(titulos) # Las notas siempre son 5.0 porque las está forzando
    
    fig, ax = plt.subplots(figsize=(8, 3.5), num="Ataque Semántico del Influencer")
    fig.patch.set_facecolor('#FFF9F9') # Fondo ligeramente rojizo
    
    # Usamos un color rojo/naranja para advertir que es contenido "inyectado"
    barras = ax.barh(titulos[::-1], notas[::-1], color='#E74C3C', edgecolor='darkred', alpha=0.85) 
    
    ax.set_title('Películas recomendadas por el Influencer', **TITULO_FONT)
    ax.set_xlabel('Nota del influencer', **ETIQUETA_FONT)
    ax.set_xlim(0, 5.5)
    
    for barra in barras:
        ancho = barra.get_width()
        ax.text(ancho + 0.1, barra.get_y() + barra.get_height()/2, 
                f'{ancho:.1f} ★', va='center', fontweight='bold', color='darkred')
                
    plt.tight_layout()
    plt.show()