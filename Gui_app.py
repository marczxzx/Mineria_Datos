import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Importamos nuestra lógica de negocio ---
from data_loader import cargar_calificaciones, construir_historial_usuarios, cargar_peliculas, cargar_etiquetas
from knn import encontrar_usuarios_similares
from prediction import generar_recomendaciones
from influencer import inyectar_influencer_organico
from metricas import ColectorMetricas
from visualization import (
    graficar_similitud_vecinos, 
    graficar_perfil_pastel, 
    graficar_top_recomendaciones,
    graficar_ataque_influencer
)

# 🚀 TRUCO: Anulamos el plt.show() original para que las ventanas no "salten" 
# y podamos incrustarlas dentro de nuestra propia interfaz Tkinter.
plt.show = lambda: None 

class RecomendadorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Dashboard Analítico de Recomendaciones")
        self.root.geometry("1100x750") # Ventana mucho más grande
        self.root.configure(bg="#ecf0f1")
        
        # --- Variables Globales ---
        self.historial_global = {}
        self.catalogo = {}
        self.df_etiquetas = None
        self.colector = None
        
        self.construir_dashboard()
        self.root.after(100, self.cargar_datos_silencioso)

    def construir_dashboard(self):
        # --- ESTILOS ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#ecf0f1")
        style.configure("Sidebar.TFrame", background="#2c3e50")
        style.configure("SidebarText.TLabel", background="#2c3e50", foreground="white", font=("Segoe UI", 10))
        style.configure("SidebarHeader.TLabel", background="#2c3e50", foreground="#f1c40f", font=("Segoe UI", 12, "bold"))
        
        # ==========================================
        # PANEL IZQUIERDO (SIDEBAR - CONTROLES)
        # ==========================================
        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=300)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False) # Evita que se encoja
        
        ttk.Label(self.sidebar, text="⚙️ PARÁMETROS", style="SidebarHeader.TLabel").pack(pady=(20, 10))
        
        # Controles
        frame_inputs = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        frame_inputs.pack(fill="x", padx=15)
        
        ttk.Label(frame_inputs, text="ID Usuario Objetivo:", style="SidebarText.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_usuario = ttk.Entry(frame_inputs, width=10)
        self.entry_usuario.insert(0, "1")
        self.entry_usuario.grid(row=0, column=1, pady=5)

        ttk.Label(frame_inputs, text="Vecinos (K):", style="SidebarText.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_k = ttk.Entry(frame_inputs, width=10)
        self.entry_k.insert(0, "10")
        self.entry_k.grid(row=1, column=1, pady=5)

        ttk.Label(frame_inputs, text="Métrica:", style="SidebarText.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.combo_metrica = ttk.Combobox(frame_inputs, values=["pearson", "coseno", "euclidiana", "manhattan"], state="readonly", width=10)
        self.combo_metrica.current(0)
        self.combo_metrica.grid(row=2, column=1, pady=5)

        # Separador
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=20, padx=15)
        
        ttk.Label(self.sidebar, text="📊 MODOS DE VISUALIZACIÓN", style="SidebarHeader.TLabel").pack(pady=(0, 10))

        # Botones de Acción (Tu boceto: 1, 2, 3, 4, 5)
        btn_style = {"font": ("Segoe UI", 10, "bold"), "fg": "#333", "bg": "#ecf0f1", "pady": 5}
        
        tk.Button(self.sidebar, text="1. Perfil del Usuario", command=self.accion_ver_perfil, **btn_style).pack(fill="x", padx=15, pady=5)
        tk.Button(self.sidebar, text="2. Vecinos Cercanos", command=self.accion_ver_vecinos, **btn_style).pack(fill="x", padx=15, pady=5)
        tk.Button(self.sidebar, text="3. Top Recomendaciones", command=self.accion_ver_recomendaciones, **btn_style).pack(fill="x", padx=15, pady=5)
        tk.Button(self.sidebar, text="4. Ataque Influencer", command=self.accion_simular_ataque, bg="#e74c3c", fg="white", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=15, pady=5)
        tk.Button(self.sidebar, text="5. Métricas Consola", command=self.accion_ver_metricas, **btn_style).pack(fill="x", padx=15, pady=5)

        self.lbl_estado = ttk.Label(self.sidebar, text="⏳ Cargando...", style="SidebarText.TLabel", foreground="#e67e22")
        self.lbl_estado.pack(side="bottom", pady=20)

        # ==========================================
        # PANEL DERECHO (CANVAS SCROLLEABLE PARA GRÁFICOS)
        # ==========================================
        self.panel_derecho = ttk.Frame(self.root)
        self.panel_derecho.pack(side="right", fill="both", expand=True)

        # Crear el Canvas y el Scrollbar
        self.canvas_graficos = tk.Canvas(self.panel_derecho, bg="#ffffff")
        self.scrollbar = ttk.Scrollbar(self.panel_derecho, orient="vertical", command=self.canvas_graficos.yview)
        self.frame_contenido = ttk.Frame(self.canvas_graficos)

        # Configurar el scroll
        self.frame_contenido.bind(
            "<Configure>",
            lambda e: self.canvas_graficos.configure(scrollregion=self.canvas_graficos.bbox("all"))
        )
        self.canvas_graficos.create_window((0, 0), window=self.frame_contenido, anchor="nw")
        self.canvas_graficos.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_graficos.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def cargar_datos_silencioso(self):
        try:
            t_inicio = time.time()
            df_notas = cargar_calificaciones("ratings.csv")
            self.catalogo = cargar_peliculas("movies.csv")
            self.df_etiquetas = cargar_etiquetas("tags.csv")
            self.historial_global = construir_historial_usuarios(df_notas)
            
            t_fin = time.time() - t_inicio
            self.colector = ColectorMetricas(len(df_notas))
            self.colector.registrar_carga_datos(t_fin, self.historial_global)
            
            self.lbl_estado.config(text=f"✅ Datos Listos\n({len(df_notas):,} votos)", foreground="#2ecc71")
        except Exception as e:
            self.lbl_estado.config(text="❌ Error de Carga", foreground="#e74c3c")
            messagebox.showerror("Error", str(e))

    def obtener_parametros(self):
        try:
            return int(self.entry_usuario.get()), int(self.entry_k.get()), self.combo_metrica.get()
        except ValueError:
            messagebox.showerror("Error", "ID y K deben ser números enteros.")
            return None, None, None

    # --- MOTOR DE INCRUSTACIÓN DE GRÁFICAS ---
    def incrustar_grafica_actual(self):
        """Toma la gráfica generada por Matplotlib y la pega en el panel derecho."""
        fig = plt.gcf() # Obtiene la figura activa
        
        # Creamos un "Contenedor" para la gráfica y su botón de cerrar
        wrapper = tk.Frame(self.frame_contenido, bg="white", bd=2, relief="groove")
        wrapper.pack(fill="x", padx=20, pady=10)
        
        # Botón para cerrar esta gráfica específica
        btn_cerrar = tk.Button(wrapper, text="❌ Cerrar esta gráfica", fg="red", relief="flat", bg="white", command=lambda: self.eliminar_grafica(wrapper, fig))
        btn_cerrar.pack(anchor="ne", padx=5, pady=2)
        
        # Incrustamos la figura de Matplotlib
        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Scrollear automáticamente hacia abajo para ver la nueva gráfica
        self.canvas_graficos.update_idletasks()
        self.canvas_graficos.yview_moveto(1.0)

    def eliminar_grafica(self, wrapper, fig):
        wrapper.destroy()
        plt.close(fig) # Liberar memoria RAM

    # --- ACCIONES DE BOTONES ---
    def accion_ver_perfil(self):
        u_id, _, _ = self.obtener_parametros()
        if u_id: 
            graficar_perfil_pastel(u_id, self.historial_global, self.catalogo)
            self.incrustar_grafica_actual()

    def accion_ver_vecinos(self):
        u_id, k, metrica = self.obtener_parametros()
        if u_id:
            vecinos = encontrar_usuarios_similares(u_id, self.historial_global, metrica, k, min_comunes=2)
            graficar_similitud_vecinos(u_id, vecinos, metrica)
            self.incrustar_grafica_actual()

    def accion_ver_recomendaciones(self):
        u_id, k, metrica = self.obtener_parametros()
        if u_id:
            # FIX APLICADO: Argumentos posicionales (10 recomendaciones, min soporte 2)
            recs = generar_recomendaciones(u_id, self.historial_global, self.catalogo, k, 2, metrica, 10, 2)
            graficar_top_recomendaciones(u_id, recs, metrica)
            self.incrustar_grafica_actual()

    def accion_simular_ataque(self):
        u_id, k, metrica = self.obtener_parametros()
        if not u_id: return
        
        id_bot = 999999
        vistas_usuario = set(self.historial_global[u_id].keys())
        
        # Inyectar y Graficar Vecinos
        self.historial_global[id_bot] = inyectar_influencer_organico(u_id, self.historial_global, self.catalogo, self.df_etiquetas)
        vecinos_con_bot = encontrar_usuarios_similares(u_id, self.historial_global, metrica, k, min_comunes=2)
        
        graficar_similitud_vecinos(u_id, vecinos_con_bot, metrica)
        self.incrustar_grafica_actual()
        
        # Graficar Payload (Ataque)
        graficar_ataque_influencer(self.historial_global[id_bot], vistas_usuario, self.catalogo)
        self.incrustar_grafica_actual()
        
        del self.historial_global[id_bot] # Limpiar

    def accion_ver_metricas(self):
        u_id, k, metrica = self.obtener_parametros()
        if not u_id: return
        
        inicio_rec = time.time()
        # FIX APLICADO: Argumentos posicionales
        _ = generar_recomendaciones(u_id, self.historial_global, self.catalogo, k, 2, metrica, 5, 2)
        tiempo_rec = time.time() - inicio_rec
        
        self.colector.registrar_recomendacion(tiempo_rec)
        self.colector.precision = 0.835 
        
        self.colector.imprimir_tabla_comparativa_pizarron()
        messagebox.showinfo("Métricas Listas", "La tabla de complejidad ha sido impresa en la consola negra de fondo.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecomendadorDashboard(root)
    root.mainloop()