# app.py - Streamlit Minimalista (Solo Análisis)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime
import warnings
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium  # <--- Necesario para mostrar mapas en Streamlit

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# =============================================================================
# CONFIGURACIÓN BÁSICA
# =============================================================================
st.set_page_config(page_title="Hubway Analytics", page_icon="🚲", layout="wide")

# =============================================================================
# FUNCIONES ESENCIALES (INTACTAS)
# =============================================================================
@st.cache_data
def load_and_preprocess(stations_path, trips_path):
    stations = pd.read_csv(stations_path)
    trips = pd.read_csv(trips_path)

    df = trips.dropna(axis=0)
    df['age'] = 2026 - df['birth_date'].values
    df.drop('birth_date', axis=1, inplace=True)

    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.dropna(subset=['start_date'])
    df['hour'] = df['start_date'].dt.hour
    df['day_of_week'] = df['start_date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

    stations.dropna(axis=0, inplace=True)
    return df, stations

@st.cache_data
def calculate_distances(trips, stations, center_lat=42.355, center_lon=-71.065):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    merged = trips.merge(
        stations[['id', 'lat', 'lng']], 
        how='left',
        left_on='end_statn',
        right_on='id'
    )
    merged['dist_to_center'] = haversine(
        merged['lat'], merged['lng'],
        center_lat, center_lon
    )
    return merged

# =============================================================================
# VISUALIZACIONES ORIGINALES (INTACTAS)
# =============================================================================
def plot_demographics(trips):
    gender_counts = np.unique(trips['gender'].values, return_counts=True)
    labels = gender_counts[0]
    counts = gender_counts[1]

    fig1, ax1 = plt.subplots(figsize=(10, 6))    
    ax1.bar(range(2), width=0.5, height=counts, color=['#e4a199', 'blue'], edgecolor='black')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_title('Distribución por Género', fontweight='bold')
    ax1.set_ylabel('Usuarios')

    total = counts.sum()
    for i in range(len(labels)):
        pct = counts[i] / total * 100
        ax1.text(i, counts[i] + total * 0.01, f'{pct:.1f}%', ha='center', fontsize=9)

   # Mostrar gráfico de género
    plt.show()

    # Gráfico 2: Distribución de Edades
    age_counts = np.unique(trips['age'], return_counts=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(age_counts[0], age_counts[1], align='center', width=0.8, alpha=0.6)
    ax2.axvline(x=np.mean(trips['age']), color='red', label='Promedio de edad')
    ax2.axvline(x=np.percentile(trips['age'], 25), color='red', linestyle='--', label='Cuartil inferior')
    ax2.axvline(x=np.percentile(trips['age'], 75), color='red', linestyle='--', label='Cuartil superior')
    ax2.set_xlabel('Edad (años)')
    ax2.set_ylabel('Número de viajes')
    ax2.set_title('Distribución de Edades', fontweight='bold')
    ax2.legend(fontsize=8)

    plt.show()
    return fig1, fig2


def plot_temporal(trips):
    # Gráfico 1: Viajes por Hora del Día
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    check_out_hours = trips['hour'].value_counts().sort_index()
    ax1.bar(check_out_hours.index, check_out_hours.values, align='center', width=0.4, alpha=0.6)
    ax1.set_xlim([-1, 24])
    ax1.set_xticks(range(24))
    ax1.set_xlabel('Hora del Día')
    ax1.set_ylabel('Número de Viajes (checkouts)')
    ax1.set_title('Viajes por Hora del Día', fontweight='bold')

    # Gráfico 2: Viajes por Día de la Semana
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    daily = trips['day_of_week'].value_counts().reindex(day_order)
    colors = ['#3498db'] * 5 + ['#e74c3c'] * 2
    ax2.bar(day_labels, daily.values, color=colors)
    ax2.set_title('📅 Viajes por Día de la Semana', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # Retornar ambas figuras por separado
    return fig1, fig2

def plot_heatmap(trips, stations):
    stations_gps = stations[['id','lat','lng']].rename(columns={'lat':'station_lat','lng':'station_lng'})
    trips_gps = trips.join(stations_gps.set_index('id'), on='strt_statn')
    heat_df = trips_gps[['station_lat','station_lng']].dropna()
    
    if heat_df.empty:
        st.warning("⚠️ No hay coordenadas válidas para el mapa de calor")
        return folium.Map(location=[42.3601,-71.0589], zoom_start=13)
    
    heat_points = heat_df.values.tolist()
    mapa = folium.Map(location=[42.3601, -71.0589], zoom_start=13)
    HeatMap(heat_points, radius=8, blur=15).add_to(mapa)
    return mapa

import plotly.graph_objects as go

def plot_distance_regression(df, stations):
    checkouts = df.groupby('end_statn').agg(
        checkouts=('end_statn', 'count'),
        dist_to_center=('dist_to_center', 'mean')
    ).reset_index().dropna()
    
    if len(checkouts) < 10:
        return None, None
    
    X = checkouts[['dist_to_center']]
    y = checkouts['checkouts']
    model = LinearRegression().fit(X, y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X['dist_to_center'], y)
    r2 = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    
    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

    # Agregar puntos de dispersión
    fig.add_trace(go.Scatter(
        x=checkouts['dist_to_center'],
        y=checkouts['checkouts'],
        mode='markers',
        marker=dict(color='#2980b9', size=8, opacity=0.6),
        name='Checkouts por Estación'
    ))

    # Línea de regresión
    x_line = np.linspace(X['dist_to_center'].min(), X['dist_to_center'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=model.predict(x_line.reshape(-1, 1)),
        mode='lines',
        line=dict(color='#c0392b', width=3),
        name=f'Regresión: y = {intercept:.0f} {slope:+.0f}x'
    ))

    # Títulos y etiquetas
    fig.update_layout(
        title=f"Modelo de Regresión: y = {intercept:.0f} {slope:+.0f}x",
        xaxis_title="Distancia al Centro (millas)",
        yaxis_title="Checkouts por Estación",
        showlegend=True,
        hovermode='closest'
    )

    # Mostrar estadísticas del modelo como texto
    model_stats = {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'mae': mae
    }
    
    return fig, model_stats
# =============================================================================
# NUEVAS VISUALIZACIONES (AÑADIDAS PARA LAS PREGUNTAS FALTANTES)
# =============================================================================
def plot_subscribers(trips):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(y='subsc_type', data=trips, palette='pastel', ax=ax)
    ax.set_title('Suscriptores vs Usuarios Ocasionales', fontweight='bold')
    ax.set_xlabel('Cantidad de Usuarios')
    ax.set_ylabel('Tipo de Usuario')
    plt.tight_layout()
    return fig

def plot_seasonality(trips):
    # Extraemos el mes para ver verano vs otoño
    trips['month'] = trips['start_date'].dt.month
    fig, ax = plt.subplots(figsize=(14, 4))
    month_counts = trips['month'].value_counts().sort_index()
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    ax.bar(month_counts.index, month_counts.values, color='#f1c40f', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses_nombres)
    ax.set_title('Volumen de Viajes por Mes (Verano vs Otoño)', fontweight='bold')
    ax.set_ylabel('Número de Viajes')
    plt.tight_layout()
    return fig

def plot_municipalities(trips, stations):
    # Cruzamos con stations para tener el municipio de origen
    merged = trips.merge(stations[['id', 'municipal']], left_on='strt_statn', right_on='id', how='left')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=merged, x='municipal', order=merged['municipal'].value_counts().index, palette='viridis', ax=ax)
    ax.set_title('Viajes por Municipio (Boston vs Cambridge)', fontweight='bold')
    ax.set_ylabel('Número de Viajes')
    ax.set_xlabel('Municipio')
    plt.tight_layout()
    return fig

# =============================================================================
# INTERFAZ PRINCIPAL CON LAYOUT PARA EXPOSICIÓN
# =============================================================================
def main():
    st.title("Hubway Analytics Dashboard 🚲")
    
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    stations_path = os.path.join(BASE_DIR, "Hubway_datasets/hubway_stations.csv")
    trips_path = os.path.join(BASE_DIR, "Hubway_datasets/hubway_trips.csv")
    
    # Manejo de error por si las rutas no coinciden
    try:
        trips, stations = load_and_preprocess(stations_path, trips_path)
        trips = calculate_distances(trips, stations)
        st.success(f"✅ {len(trips):,} viajes procesados exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar los datos. Verifica la ruta de los CSV. Detalles: {e}")
        return

    # KPIs
    st.subheader("Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Viajes (Filtrados)", f"{len(trips):,}")
    col2.metric("Duración Promedio", f"{trips['duration'].mean()/60:.1f} min")
    col3.metric("Edad Promedio", f"{trips['age'].mean():.1f} años")
    col4.metric("Estaciones Únicas", f"{trips['end_statn'].nunique()}")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Demografía", "Análisis Temporal", "Geografía", "Modelo & Distancias"])

    # --- TAB 1: DEMOGRAFÍA ---
    with tab1:
        # Gráfico de Género
        st.subheader("Distribución por Género")
        fig_gender = plot_demographics(trips)[0]  # Solo el primer gráfico (Género)
        st.pyplot(fig_gender)

        # Hallazgos sobre Género
        st.warning(
            "**Hallazgos sobre Género:**\n\n"
            "1. **Género:** Existe una dominancia clara de usuarios masculinos (~75%). Esto representa una oportunidad de marketing para atraer al público femenino."
        )

        # Gráfico de Edad
        st.subheader("Distribución de Edades")
        fig_age = plot_demographics(trips)[1]  # Solo el segundo gráfico (Edad)
        st.pyplot(fig_age)

        # Hallazgos sobre Edad
        st.warning(
            "**Hallazgos sobre Edad:**\n\n"
            "2. **Edad:** El pico principal de usuarios se encuentra alrededor de los 40 años. Hay un bajo uso en rangos muy jóvenes (menores de 25)."
        )

        st.subheader("Suscriptores vs Usurios")
        st.pyplot(plot_subscribers(trips))
        # Hallazgo sobre Fidelización
        st.warning(
            "3. **Fidelización:** La inmensa mayoría de los usuarios (con datos completos) son **Suscriptores Registrados**. Esto indica que el sistema se utiliza como un medio de transporte rutinario más que como una atracción turística de un solo uso."
        )
            
    # --- TAB 2: TIEMPO ---
    with tab2:
        # Gráfico de Temporalidad
        st.subheader("Distribución Temporal")
        st.pyplot(plot_temporal(trips)[0])
        st.success(
            "1. **Hora Punta (Transporte vs Recreación):** Vemos picos masivos a las **8:00 AM y 5:00 PM**. Esto responde a tu pregunta: las bicicletas se usan principalmente para el **transporte diario al trabajo (commuting)**, no por motivos puramente recreativos.\n"
        )
        st.pyplot(plot_temporal(trips)[1])
        st.success(
            "2. **Días de la semana:** Hay mayor uso de Lunes a Viernes que los fines de semana, confirmando la hipótesis del uso laboral.\n"
        )
        st.pyplot(plot_seasonality(trips))
        # Hallazgos sobre Tiempo
        st.success(
            "3. **Verano vs Otoño:** Existe una fuerte estacionalidad. Los meses cálidos (Junio a Septiembre) concentran la mayor parte de los viajes. En otoño e invierno el uso cae drásticamente debido al clima de Boston."
        )


    # --- TAB 3: GEOGRAFÍA (MAPA Y MUNICIPIOS) ---
    with tab3:
        # Gráfico de Municipalidades
        st.subheader("Distribución Geográfica")
         # Hallazgos sobre Geografía
        st.markdown("### Hallazgos Geográficos")
        st.warning(
            "**¿Dónde se prestan las bicicletas?**\n\n"
            "1. **Boston vs Cambridge:** El gráfico muestra que **Boston domina** ampliamente el volumen de viajes en comparación con Cambridge u otros municipios.\n"
            "2. **Zonas Comerciales vs Residenciales:** En el mapa de calor, los 'hotspots' (puntos rojos) se concentran fuertemente en el centro de la ciudad (Downtown Boston, zonas financieras y universidades). Esto indica una alta demanda en **zonas comerciales, educativas y laborales** más que en los suburbios puramente residenciales."
        )

        st.pyplot(plot_municipalities(trips, stations))

        # Mapa de Calor
        st.subheader("Mapa de Calor de Estaciones de Inicio")
        mapa = plot_heatmap(trips, stations)
        st_folium(mapa, width=700, height=500)

    # --- TAB 4: MODELO DE DISTANCIA ---
    with tab4:
        # Gráfico del Modelo de Distancia
        model_fig, model_stats = plot_distance_regression(trips, stations)
        if model_fig:
            # Mostrar gráfico interactivo con Plotly
            st.plotly_chart(model_fig)

        # Hallazgos sobre el Modelo
        if model_stats:
            st.markdown("### Modelo de Regresión")
            st.info(
                "**Análisis de la Distancia al Centro:**\n\n"
                "Este modelo valida nuestra hipótesis visual del mapa de calor con matemáticas:\n\n"
                f"- Existe una relación inversa. Por cada milla que nos alejamos del centro geográfico, perdemos aproximadamente **{abs(model_stats['slope']):,.0f} checkouts** por estación.\n"
                f"- El R² es de **{model_stats['r2']:.3f}**, lo que significa que la pura distancia al centro explica una parte razonable del éxito de una estación."
            )
if __name__ == "__main__":
    main()