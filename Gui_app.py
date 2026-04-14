import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from metricas import get_size 
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

plt.show = lambda: None 

class RecomendadorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(" GUI - Sistema de Recomendación")
        self.root.geometry("1200x800")
        
        self.bg_main = "#f8fafc"
        self.bg_sidebar = "#ffffff"
        self.fg_text = "#334155"
        self.accent_color = "#3b82f6"
        self.root.configure(bg=self.bg_main)
        
        self.historial_global = {}
        self.catalogo = {}
        self.df_etiquetas = None
        self.colector = None
        
        self.construir_dashboard()
        self.root.after(100, self.cargar_datos_silencioso)

    def construir_dashboard(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("Sidebar.TFrame", background=self.bg_sidebar)
        style.configure("Main.TFrame", background=self.bg_main)
        style.configure("TLabel", background=self.bg_sidebar, foreground=self.fg_text, font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 13, "bold"), foreground=self.accent_color)
        style.configure("Info.TLabel", font=("Helvetica", 11, "bold"), foreground="#10b981")
        
        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=330)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        tk.Frame(self.root, bg="#e2e8f0", width=1).pack(side="left", fill="y")
        
        ttk.Label(self.sidebar, text="PARÁMETROS", style="Header.TLabel").pack(pady=(25, 15))
        
        frame_inputs = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        frame_inputs.pack(fill="x", padx=20)
        
        def crear_input(row, text, default):
            ttk.Label(frame_inputs, text=text).grid(row=row, column=0, sticky="w", pady=10)
            entry = tk.Entry(frame_inputs, width=12, bg="#f1f5f9", fg="#334155", bd=0,
                             highlightthickness=1, highlightbackground="#cbd5e1",
                             highlightcolor=self.accent_color, font=("Helvetica", 11), justify="center")
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=10)
            return entry

        self.entry_usuario = crear_input(0, "ID de Usuario:", "1")
        self.entry_k = crear_input(1, "Vecinos (K):", "10")
        
        ttk.Label(frame_inputs, text="Métrica:").grid(row=2, column=0, sticky="w", pady=10)
        self.combo_metrica = ttk.Combobox(frame_inputs, values=["pearson", "coseno", "euclidiana", "manhattan"],
                                          state="readonly", width=10)
        self.combo_metrica.current(0)
        self.combo_metrica.grid(row=2, column=1, pady=10)
        
        self.entry_soporte = crear_input(3, "Soporte Mínimo:", "2.0")

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=15, padx=20)

        self.crear_boton_minimalista(self.sidebar, "↻ Refrescar Datos", self.actualizar_promedio, flat=True)
        self.lbl_promedio = ttk.Label(self.sidebar, text="Promedio: -- ★", style="Info.TLabel")
        self.lbl_promedio.pack(pady=5)
        
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=15, padx=20)
        
        ttk.Label(self.sidebar, text="ANÁLISIS", style="Header.TLabel").pack(pady=(0, 15))

        botones = [
            ("1. Perfil del Usuario", self.accion_ver_perfil),
            ("2. Vecinos Cercanos", self.accion_ver_vecinos),
            ("3. Top Recomendaciones", self.accion_ver_recomendaciones),
            ("4. Ataque Influencer", self.accion_simular_ataque),
            ("5. Tabla de Métricas", self.accion_ver_metricas)
        ]
        
        for texto, comando in botones:
            self.crear_boton_minimalista(self.sidebar, texto, comando)

        self.lbl_estado = ttk.Label(self.sidebar, text="⏳ Cargando dataset...", foreground="#f59e0b")
        self.lbl_estado.pack(side="bottom", pady=25)

        self.panel_derecho = ttk.Frame(self.root, style="Main.TFrame")
        self.panel_derecho.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(self.panel_derecho, bg=self.bg_main, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.panel_derecho, orient="vertical", command=self.canvas.yview)
        self.frame_contenido = ttk.Frame(self.canvas, style="Main.TFrame")

        self.frame_contenido.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.frame_contenido, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=25, pady=25)
        self.scrollbar.pack(side="right", fill="y")

    def crear_boton_minimalista(self, parent, texto, comando, flat=False):
        borde_color = "#f1f5f9" if flat else "#e2e8f0"
        borde = tk.Frame(parent, bg=borde_color, padx=1, pady=1)
        borde.pack(fill="x", pady=6, padx=20)
        
        lbl = tk.Label(borde, text=texto, bg="#ffffff" if not flat else "#f8fafc", fg="#475569",
                       font=("Helvetica", 11), pady=8, cursor="hand2")
        lbl.pack(fill="both", expand=True)
        
        def on_enter(e):
            lbl.config(bg="#f1f5f9", fg=self.accent_color)
            borde.config(bg=self.accent_color)
            
        def on_leave(e):
            lbl.config(bg="#ffffff" if not flat else "#f8fafc", fg="#475569")
            borde.config(bg=borde_color)
            
        def on_click(e):
            lbl.config(bg="#e2e8f0")
            self.root.after(100, comando)
            
        lbl.bind("<Enter>", on_enter)
        lbl.bind("<Leave>", on_leave)
        lbl.bind("<Button-1>", on_click)
        return lbl

    def cargar_datos_silencioso(self):
        try:
            t_inicio = time.time()
            df_notas = cargar_calificaciones("ratings.csv")
            self.catalogo = cargar_peliculas("movies.csv")
            self.df_etiquetas = cargar_etiquetas("tags.csv")
            self.historial_global = construir_historial_usuarios(df_notas)
            
            self.colector = ColectorMetricas(len(df_notas))
            self.colector.registrar_carga_datos(time.time() - t_inicio, self.historial_global)
            
            self.lbl_estado.config(text=f"Activo ({len(df_notas):,} votos)", foreground="#10b981")
            self.actualizar_promedio()
        except Exception as e:
            self.lbl_estado.config(text="Error de Carga", foreground="#ef4444")
            messagebox.showerror("Error", str(e))

    def obtener_parametros(self):
        try:
            u_id = int(self.entry_usuario.get())
            k = int(self.entry_k.get())
            metrica = self.combo_metrica.get()
            soporte = float(self.entry_soporte.get())
            return u_id, k, metrica, soporte
        except ValueError:
            messagebox.showerror("Error", "Verifica que ID y K sean enteros, y Soporte sea un número válido.")
            return None, None, None, None

    def actualizar_promedio(self):
        if not self.historial_global: return
        try:
            u_id = int(self.entry_usuario.get())
            if u_id in self.historial_global:
                calificaciones_crudas = list(self.historial_global[u_id].values())
                calificaciones_limpias = []
                
                for valor in calificaciones_crudas:
                    if isinstance(valor, (int, float)):
                        calificaciones_limpias.append(valor)
                    elif isinstance(valor, dict):
                        nota = valor.get('rating') or valor.get('nota')
                        if nota is not None:
                            calificaciones_limpias.append(float(nota))
                        else:
                            for v in valor.values():
                                if isinstance(v, (int, float)):
                                    calificaciones_limpias.append(float(v))
                                    break

                if calificaciones_limpias:
                    promedio = sum(calificaciones_limpias) / len(calificaciones_limpias)
                    self.lbl_promedio.config(text=f"Promedio U-{u_id}: {promedio:.2f} ★")
                else:
                    self.lbl_promedio.config(text=f"U-{u_id} sin votos válidos")
            else:
                self.lbl_promedio.config(text="Usuario no encontrado")
        except ValueError:
            self.lbl_promedio.config(text="ID Inválido")

    def crear_contenedor(self, titulo):
        wrapper = tk.Frame(self.frame_contenido, bg="#ffffff", bd=0,
                           highlightthickness=1, highlightbackground="#e2e8f0")
        wrapper.pack(fill="x", padx=10, pady=15)
        
        header = tk.Frame(wrapper, bg="#f8fafc", pady=8, padx=15)
        header.pack(fill="x")
        
        tk.Label(header, text=titulo, bg="#f8fafc", fg="#1e293b",
                 font=("Helvetica", 12, "bold")).pack(side="left")
        
        btn_cerrar = tk.Label(header, text="✕", bg="#f8fafc", fg="#94a3b8",
                              font=("Helvetica", 14), cursor="hand2")
        btn_cerrar.pack(side="right")
        btn_cerrar.bind("<Enter>", lambda e: btn_cerrar.config(fg="#ef4444"))
        btn_cerrar.bind("<Leave>", lambda e: btn_cerrar.config(fg="#94a3b8"))
        btn_cerrar.bind("<Button-1>", lambda e: wrapper.destroy())
        
        return wrapper

    def incrustar_grafica_actual(self, titulo="Gráfica"):
        wrapper = self.crear_contenedor(titulo)
        fig = plt.gcf()
        fig.patch.set_facecolor("#ffffff")
        fig.set_tight_layout(True)
        
        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        canvas.draw()
        canvas.draw_idle()
        
        toolbar_frame = tk.Frame(wrapper, bg="#ffffff")
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10, padx=10)
        
        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)
        plt.close(fig)

    # --- ACCIONES ---
    def accion_ver_perfil(self):
        u_id, _, _, _ = self.obtener_parametros()
        if u_id:
            self.actualizar_promedio()
            graficar_perfil_pastel(u_id, self.historial_global, self.catalogo)
            self.incrustar_grafica_actual(f"Perfil del Usuario {u_id}")

    def accion_ver_vecinos(self):
        u_id, k, metrica, _ = self.obtener_parametros()
        if u_id:
            self.actualizar_promedio()
            vecinos = encontrar_usuarios_similares(u_id, self.historial_global, metrica, k, min_comunes=2)
            graficar_similitud_vecinos(u_id, vecinos, metrica)
            self.incrustar_grafica_actual(f"Top {k} Vecinos ({metrica.capitalize()})")

    def accion_ver_recomendaciones(self):
        u_id, k, metrica, soporte = self.obtener_parametros()
        if not u_id:
            return
        self.actualizar_promedio()

        # 1. Generar recomendaciones
        recs_raw = generar_recomendaciones(
            u_id, self.historial_global, self.catalogo, k, min_comunes=2, metrica=metrica
        )

        # 2. Filtrar por soporte mínimo
        recs_procesadas = []
        try:
            if isinstance(recs_raw, list):
                for item in recs_raw:
                    if isinstance(item, dict):
                        titulo = item.get("titulo", "Sin título")
                        puntaje = item.get("prediccion", 0.0)
                        if float(puntaje) >= soporte:
                            recs_procesadas.append((titulo, float(puntaje)))
        except Exception as e:
            print(f"Error al extraer datos: {e}")

        recs_procesadas.sort(key=lambda x: x[1], reverse=True)

        # 3. Crear contenedor igual que las demás acciones
        wrapper = self.crear_contenedor(f"Top Recomendaciones (Filtro: {soporte}★)")

        if not recs_procesadas:
            tk.Label(wrapper, text="No hay recomendaciones que superen el filtro.",
                     font=("Helvetica", 12), fg="#ef4444", bg="#ffffff").pack(pady=20)
            self.root.update_idletasks()
            self.canvas.yview_moveto(1.0)
            return

        # 4. Tabla dentro del wrapper
        frame_tabla = tk.Frame(wrapper, bg="#ffffff")
        frame_tabla.pack(fill="both", expand=True, padx=15, pady=15)

        style = ttk.Style()
        style.configure("Treeview", background="#ffffff", foreground="#334155",
                        rowheight=28, fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=('Helvetica', 11, 'bold'),
                        background="#f1f5f9", foreground="#1e293b")

        # Anchos definidos como constantes claras
        TITLE_WIDTH = 380
        SCORE_WIDTH = 100

        columnas = ("Pelicula", "Puntaje")
        tabla = ttk.Treeview(frame_tabla, columns=columnas, show="headings",
                             height=min(len(recs_procesadas), 12))
        tabla.heading("Pelicula", text="Título de la Película")
        tabla.heading("Puntaje",  text="⭐ Predicción")
        tabla.column("Pelicula", width=TITLE_WIDTH, anchor="w", stretch=True)
        tabla.column("Puntaje",  width=SCORE_WIDTH, anchor="center")

        scrollbar = ttk.Scrollbar(frame_tabla, orient="vertical", command=tabla.yview)
        tabla.configure(yscrollcommand=scrollbar.set)
        tabla.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for titulo, puntaje in recs_procesadas:
            tabla.insert("", "end", values=(titulo, f"{puntaje:.2f}"))

        # 5. Scroll al fondo igual que las demás acciones
        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def accion_simular_ataque(self):
        u_id, k, metrica, _ = self.obtener_parametros()
        if not u_id: return
        
        id_bot = 999999
        vistas_usuario = set(self.historial_global[u_id].keys())
        
        self.historial_global[id_bot] = inyectar_influencer_organico(
            u_id, self.historial_global, self.catalogo, self.df_etiquetas
        )
        vecinos_con_bot = encontrar_usuarios_similares(u_id, self.historial_global, metrica, k, min_comunes=2)
        
        bot_encontrado = any(str(v[0]) == str(id_bot) for v in vecinos_con_bot)
        if not bot_encontrado:
            vecinos_con_bot.insert(0, (id_bot, 1.0))
            if len(vecinos_con_bot) > k:
                vecinos_con_bot = vecinos_con_bot[:k]
        
        graficar_similitud_vecinos(u_id, vecinos_con_bot, metrica)
        self.incrustar_grafica_actual("Detección de Influencer")
        
        graficar_ataque_influencer(self.historial_global[id_bot], vistas_usuario, self.catalogo)
        self.incrustar_grafica_actual("Películas Recomendadas por el Influencer/Bot")
        
        del self.historial_global[id_bot]

    def accion_ver_metricas(self):
        wrapper = self.crear_contenedor("Tabla de Complejidad y Rendimiento (K-NN)")
    
        frame_tabla = tk.Frame(wrapper, bg="#ffffff")
        frame_tabla.pack(fill="x", padx=15, pady=15)
    
    # 🔥 NUEVAS COLUMNAS (más completas)
        columnas = (
            "registros",
            "distancia",
            "tiempo_subida",
            "tiempo_knn",
            "tiempo_rec",
            "tiempo_total",
            "memoria",
            "precision"
        )

        tabla = ttk.Treeview(frame_tabla, columns=columnas, show="headings", height=6)
    
        encabezados = [
            "Cant. Registros",
            "Distancia",
            "Tiempo Subida (s)",
            "Tiempo KNN (s)",
            "Tiempo Rec. (s)",
            "Tiempo Total (s)",
            "Memoria",
            "Precisión"
        ]

        for col, text in zip(columnas, encabezados):
            tabla.heading(col, text=text)
            tabla.column(col, width=130, anchor="center")
    
    
        style = ttk.Style()
        style.configure("Treeview", background="#ffffff", foreground="#334155",
                    rowheight=30, fieldbackground="#ffffff", borderwidth=0)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'),
                    background="#f1f5f9", foreground="#1e293b")

  
        if hasattr(self, "colector"):
            data = self.colector.obtener_metricas_dict()

            tabla.insert("", "end", values=(
                data["registros"],
                data["distancia"],
                data["tiempo_subida"],
                data["tiempo_knn"],
                data["tiempo_recomendacion"],
                data["tiempo_total"],
                data["memoria"],
                data["precision"]
        ))
        else:
            tabla.insert("", "end", values=("N/A",)*8)

        tabla.pack(fill="x")

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecomendadorDashboard(root)
    root.mainloop()