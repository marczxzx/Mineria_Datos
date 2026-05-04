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
# FUNCIONES ESENCIALES
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
# VISUALIZACIONES
# =============================================================================
def plot_demographics(trips):
    gender_counts = np.unique(trips['gender'].values, return_counts=True)
    labels = gender_counts[0]
    counts = gender_counts[1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))    
    
    # 🔹 Género
    ax[0].bar(range(2), width=0.5, height=counts, color=['#e4a199','green'], edgecolor='black')
    ax[0].set_xticks([0,1])
    ax[0].set_xticklabels(labels)
    ax[0].set_title('Distribución por Género', fontweight='bold')
    ax[0].set_ylabel('Usuarios')

    total = counts.sum()
    for i in range(len(labels)):
        pct = counts[i] / total * 100
        ax[0].text(i, counts[i] + total*0.01, f'{pct:.1f}%', ha='center', fontsize=9)

    # 🔹 Edad
    age_counts = np.unique(trips['age'], return_counts=True)
    ax[1].bar(age_counts[0], age_counts[1], align='center', width=0.8, alpha=0.6)
    ax[1].axvline(x=np.mean(trips['age']), color='red', label='average age')
    ax[1].axvline(x=np.percentile(trips['age'], 25), color='red', linestyle='--', label='lower quartile')
    ax[1].axvline(x=np.percentile(trips['age'], 75), color='red', linestyle='--', label='upper quartile')
    ax[1].set_xlabel('Edad (años)')
    ax[1].set_ylabel('Number of checkouts')
    ax[1].set_title('Distribución de Edades', fontweight='bold')
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    return fig

def plot_temporal(trips):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    check_out_hours = trips['hour'].value_counts().sort_index()
    ax[0].bar(check_out_hours.index, check_out_hours.values, align='center', width=0.4, alpha=0.6)
    ax[0].set_xlim([-1,24])
    ax[0].set_xticks(range(24))
    ax[0].set_xlabel('Hora del dìa')
    ax[0].set_ylabel('Número de Viajes (checkouts)')
    ax[0].set_title('Viajes por Hora del Día', fontweight='bold')

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    daily = trips['day_of_week'].value_counts().reindex(day_order)
    colors = ['#3498db']*5 + ['#e74c3c']*2
    ax[1].bar(day_labels, daily.values, color=colors)
    ax[1].set_title('📅 Viajes por Día de la Semana', fontweight='bold')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_heatmap(trips, stations):
    # Renombrar para no chocar con columnas de trips
    stations_gps = stations[['id','lat','lng']].rename(columns={'lat':'station_lat','lng':'station_lng'})
    
    # Unir coordenadas de inicio de viaje
    trips_gps = trips.join(stations_gps.set_index('id'), on='strt_statn')
    
    # Filtrar viajes con coordenadas válidas
    heat_df = trips_gps[['station_lat','station_lng']].dropna()
    
    if heat_df.empty:
        st.warning("⚠️ No hay coordenadas válidas para el mapa de calor")
        return folium.Map(location=[42.3601,-71.0589], zoom_start=13)
    
    heat_points = heat_df.values.tolist()
    
    # Mapa centrado en Boston
    mapa = folium.Map(location=[42.3601, -71.0589], zoom_start=13)
    HeatMap(heat_points, radius=8, blur=15).add_to(mapa)
    
    return mapa

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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(checkouts['dist_to_center'], checkouts['checkouts'], alpha=0.6, s=30, color='#2980b9')
    
    x_line = np.linspace(X['dist_to_center'].min(), X['dist_to_center'].max(), 100)
    ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color='#c0392b', linewidth=2.5)
    
    ax.set_xlabel('🗺️ Distancia al Centro (millas)')
    ax.set_ylabel('🚲 Checkouts por Estación')
    ax.set_title(f'📈 Regresión: y = {intercept:.0f} {slope:+.0f}x', fontweight='bold')
    ax.text(0.02, 0.98, f'R² = {r2:.3f} | MAE = {mae:.0f}', 
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    stats_dict = {'slope': slope, 'intercept': intercept, 'r2': r2, 'mae': mae}
    return fig, stats_dict

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    st.title("Hubway Analytics Dashboard")
    
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    stations_path = os.path.join(BASE_DIR, "Hubway_datasets/hubway_stations.csv")
    trips_path = os.path.join(BASE_DIR, "Hubway_datasets/hubway_trips.csv")
    
    trips, stations = load_and_preprocess(stations_path, trips_path)
    trips = calculate_distances(trips, stations)
    st.success(f"✅ {len(trips):,} viajes procesados")

    # KPIs
    st.subheader("📈 Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Viajes", f"{len(trips):,}")
    col2.metric("Duración Promedio", f"{trips['duration'].mean()/60:.1f} min")
    col3.metric("Edad Promedio", f"{trips['age'].mean():.1f} años")
    col4.metric("Estaciones", f"{trips['end_statn'].nunique()}")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Demografía", "Tiempo", "Modelo","Mapa de Calor"])

    with tab1:
        st.pyplot(plot_demographics(trips))
        st.dataframe(pd.DataFrame({
            'Métrica': ['Edad Promedio', 'Edad Mediana', '% Hombres', '% Mujeres'],
            'Valor': [f"{trips['age'].mean():.1f}", f"{trips['age'].median():.1f}", 
                     f"{(trips['gender']=='Male').mean()*100:.1f}%", 
                     f"{(trips['gender']=='Female').mean()*100:.1f}%"]
        }), hide_index=True, use_container_width=True)
    
    with tab2:
        st.pyplot(plot_temporal(trips))
        col1, col2 = st.columns(2)
        col1.info(f"Hora pico: **{trips['hour'].mode()[0]}:00**")
        col2.success(f"Día pico: **{trips['day_of_week'].mode()[0]}**")
    
    with tab3:
        model_fig, model_stats = plot_distance_regression(trips, stations)
        if model_fig:
            st.pyplot(model_fig)
            if model_stats:
                st.info(f"Interpretación: Cada milla adicional al centro → {abs(model_stats['slope']):,.0f} checkouts menos por estación")
    with tab4:
        st.subheader("🌡️ Mapa de Calor de Viajes")
        mapa = plot_heatmap(trips, stations)
        st_folium(mapa, width=700, height=500)

    st.markdown("---")

if __name__ == "__main__":
    main()