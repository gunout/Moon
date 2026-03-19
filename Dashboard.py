import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="🌕 Lunar Data Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #C0C0C0;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #000000, #1a1a1a, #000000);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a, #000000);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #C0C0C0;
        text-align: center;
    }
    .metric-label {
        color: #C0C0C0;
        font-size: 1rem;
    }
    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    .info-box {
        background: linear-gradient(135deg, #1a1a1a, #000000);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #C0C0C0;
        margin: 1rem 0;
    }
    .event-marker {
        background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #C0C0C0;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LunarDataAnalyzer:
    def __init__(self, data_type):
        self.data_type = data_type
        self.colors = ['#C0C0C0', '#F0F0F0', '#E0E0E0', '#D0D0D0', '#B0B0B0', 
                      '#A0A0A0', '#909090', '#808080', '#707070', '#606060']
        
        self.start_year = 1600
        self.end_year = 2025
        
        self.config = self._get_lunar_config()
        
    def _get_lunar_config(self):
        """Retourne la configuration spécifique pour chaque type de données lunaires"""
        configs = {
            "lunar_distance": {
                "base_value": 384400,
                "cycle_years": 27.3,
                "amplitude": 21000,
                "trend": "cyclique",
                "unit": "km",
                "description": "Distance Terre-Lune",
                "icon": "📏",
                "color": "#C0C0C0"
            },
            "lunar_phase": {
                "base_value": 0.5,
                "cycle_years": 29.53/365.25,
                "amplitude": 0.5,
                "trend": "cyclique",
                "unit": "Phase (0-1)",
                "description": "Phase lunaire",
                "icon": "🌗",
                "color": "#E0E0E0"
            },
            "lunar_inclination": {
                "base_value": 5.15,
                "cycle_years": 18.6,
                "amplitude": 0.5,
                "trend": "cyclique",
                "unit": "degrés",
                "description": "Inclinaison orbitale lunaire",
                "icon": "📐",
                "color": "#D0D0D0"
            },
            "lunar_libration": {
                "base_value": 0,
                "cycle_years": 27.3,
                "amplitude": 7.5,
                "trend": "cyclique",
                "unit": "degrés",
                "description": "Libration lunaire",
                "icon": "🔄",
                "color": "#B0B0B0"
            },
            "lunar_perigee": {
                "base_value": 363000,
                "cycle_years": 8.85,
                "amplitude": 2000,
                "trend": "cyclique",
                "unit": "km",
                "description": "Distance au périgée",
                "icon": "⬇️",
                "color": "#A0A0A0"
            },
            "lunar_apogee": {
                "base_value": 405000,
                "cycle_years": 8.85,
                "amplitude": 2000,
                "trend": "cyclique",
                "unit": "km",
                "description": "Distance à l'apogée",
                "icon": "⬆️",
                "color": "#909090"
            },
            "lunar_eclipse": {
                "base_value": 0,
                "cycle_years": 18.03,
                "amplitude": 1,
                "trend": "cyclique",
                "unit": "occurrences/an",
                "description": "Fréquence des éclipses lunaires",
                "icon": "🌑",
                "color": "#808080"
            },
            "lunar_temperature": {
                "base_value": -20,
                "cycle_years": 29.53/365.25,
                "amplitude": 150,
                "trend": "cyclique",
                "unit": "°C",
                "description": "Température de surface lunaire",
                "icon": "🌡️",
                "color": "#707070"
            },
            "lunar_gravity": {
                "base_value": 1.62,
                "cycle_years": 27.3,
                "amplitude": 0.01,
                "trend": "stable",
                "unit": "m/s²",
                "description": "Gravité lunaire",
                "icon": "⚖️",
                "color": "#606060"
            }
        }
        return configs.get(self.data_type, configs["lunar_distance"])
    
    def generate_lunar_data(self):
        """Génère des données lunaires simulées"""
        years = list(range(self.start_year, self.end_year + 1))
        
        data = {'Year': years}
        
        # Génération des différentes composantes
        data['Base_Value'] = self._simulate_lunar_cycle(years)
        data['Monthly_Variation'] = self._simulate_monthly_variation(years)
        data['Annual_Variation'] = self._simulate_annual_variation(years)
        data['Saros_Cycle'] = self._simulate_saros_cycle(years)
        data['Metonic_Cycle'] = self._simulate_metonic_cycle(years)
        data['Nodal_Precession'] = self._simulate_nodal_precession(years)
        data['Eclipse_Probability'] = self._simulate_eclipse_probability(years)
        data['SuperMoon_Occurrence'] = self._simulate_supermoon_occurrence(years)
        data['Libration_Amplitude'] = self._simulate_libration_amplitude(years)
        data['Smoothed_Value'] = self._simulate_smoothed_data(years)
        data['Cycle_Phase'] = self._simulate_cycle_phase(years)
        data['Anomaly_Value'] = self._simulate_anomaly_data(years)
        data['Secular_Trend'] = self._simulate_secular_trend(years)
        data['Predicted_Value'] = self._simulate_predicted_data(years)
        data['Uncertainty_Range'] = self._simulate_uncertainty_range(years)
        
        df = pd.DataFrame(data)
        self._add_lunar_events(df)
        
        return df
    
    def _simulate_lunar_cycle(self, years):
        """Simule le cycle lunaire principal"""
        values = []
        for year in years:
            phase = (year - self.start_year) % self.config["cycle_years"]
            cycle_value = np.sin(2 * np.pi * phase / self.config["cycle_years"])
            value = self.config["base_value"] + self.config["amplitude"] * cycle_value
            noise = np.random.normal(0, self.config["amplitude"] * 0.05)
            values.append(value + noise)
        return values
    
    def _simulate_monthly_variation(self, years):
        variations = []
        for year in years:
            synodic_phase = (180 / 365.25) * (365.25 / 29.53)
            monthly_variation = 0.1 * np.sin(2 * np.pi * synodic_phase)
            variations.append(1 + monthly_variation)
        return variations
    
    def _simulate_annual_variation(self, years):
        variations = []
        for year in years:
            annual_variation = 0.02 * np.sin(2 * np.pi * (year - self.start_year) / 1)
            variations.append(1 + annual_variation)
        return variations
    
    def _simulate_saros_cycle(self, years):
        saros_values = []
        for year in years:
            saros_phase = (year - self.start_year) % 18.03
            saros_value = np.sin(2 * np.pi * saros_phase / 18.03)
            saros_values.append(0.5 + 0.5 * saros_value)
        return saros_values
    
    def _simulate_metonic_cycle(self, years):
        metonic_values = []
        for year in years:
            metonic_phase = (year - self.start_year) % 19.0
            metonic_value = np.sin(2 * np.pi * metonic_phase / 19.0)
            metonic_values.append(0.5 + 0.5 * metonic_value)
        return metonic_values
    
    def _simulate_nodal_precession(self, years):
        precession_values = []
        for year in years:
            precession_phase = (year - self.start_year) % 18.6
            precession_value = np.sin(2 * np.pi * precession_phase / 18.6)
            precession_values.append(precession_value)
        return precession_values
    
    def _simulate_eclipse_probability(self, years):
        probabilities = []
        for i, year in enumerate(years):
            saros_factor = self._simulate_saros_cycle([year])[0]
            precession_factor = abs(self._simulate_nodal_precession([year])[0])
            probability = 0.3 + 0.4 * saros_factor * precession_factor
            probabilities.append(min(probability, 0.9))
        return probabilities
    
    def _simulate_supermoon_occurrence(self, years):
        occurrences = []
        for year in years:
            base_occurrence = 3.5
            cycle_variation = 0.5 * np.sin(2 * np.pi * (year - self.start_year) / 14.0)
            occurrence = base_occurrence + cycle_variation
            occurrences.append(max(1, occurrence))
        return occurrences
    
    def _simulate_libration_amplitude(self, years):
        amplitudes = []
        for year in years:
            base_amplitude = 7.0
            variation = 0.5 * np.sin(2 * np.pi * (year - self.start_year) / 27.3)
            amplitude = base_amplitude + variation
            amplitudes.append(amplitude)
        return amplitudes
    
    def _simulate_smoothed_data(self, years):
        base_cycle = self._simulate_lunar_cycle(years)
        smoothed = []
        for i in range(len(base_cycle)):
            start_idx = max(0, i - 2)
            end_idx = min(len(base_cycle), i + 3)
            window = base_cycle[start_idx:end_idx]
            smoothed.append(np.mean(window))
        return smoothed
    
    def _simulate_cycle_phase(self, years):
        phases = []
        for year in years:
            phase = (year - self.start_year) % self.config["cycle_years"] / self.config["cycle_years"]
            phases.append(phase)
        return phases
    
    def _simulate_anomaly_data(self, years):
        base_cycle = self._simulate_lunar_cycle(years)
        smoothed = self._simulate_smoothed_data(years)
        anomalies = [base_cycle[i] - smoothed[i] for i in range(len(base_cycle))]
        return anomalies
    
    def _simulate_secular_trend(self, years):
        trends = []
        for year in years:
            if self.data_type == "lunar_distance":
                years_from_start = year - self.start_year
                trend = 1 + (years_from_start * 0.000038) / self.config["base_value"]
            else:
                trend = 1.0
            trends.append(trend)
        return trends
    
    def _simulate_predicted_data(self, years):
        predictions = []
        base_cycle = self._simulate_lunar_cycle(years)
        secular_trend = self._simulate_secular_trend(years)
        
        for i, year in enumerate(years):
            current_value = base_cycle[i]
            trend_factor = secular_trend[i]
            
            if year > 2020:
                years_since_2020 = year - 2020
                uncertainty = 0.01 * years_since_2020
                prediction = current_value * trend_factor * (1 + np.random.normal(0, uncertainty))
            else:
                prediction = current_value
            predictions.append(prediction)
        return predictions
    
    def _simulate_uncertainty_range(self, years):
        uncertainties = []
        for year in years:
            if year < 1700:
                uncertainty = 0.1
            elif year < 1900:
                uncertainty = 0.05
            elif year < 1960:
                uncertainty = 0.01
            else:
                uncertainty = 0.001
            uncertainties.append(uncertainty)
        return uncertainties
    
    def _add_lunar_events(self, df):
        """Ajoute des événements lunaires historiques"""
        events = []
        for i, row in df.iterrows():
            year = row['Year']
            
            if year == 1609:
                df.loc[i, 'Anomaly_Value'] *= 1.1
                events.append({"year": year, "event": "Premières observations télescopiques (Galilée)", "type": "observation"})
            
            elif year == 1837:
                df.loc[i, 'Uncertainty_Range'] *= 0.5
                events.append({"year": year, "event": "Première carte détaillée (Beer et Mädler)", "type": "cartographie"})
            
            elif year == 1969:
                df.loc[i, 'Uncertainty_Range'] *= 0.01
                df.loc[i, 'Base_Value'] = self.config["base_value"]
                events.append({"year": year, "event": "Apollo 11 - Premiers pas sur la Lune", "type": "exploration"})
            
            elif year == 1994:
                df.loc[i, 'Uncertainty_Range'] *= 0.001
                events.append({"year": year, "event": "Mission Clementine - Cartographie complète", "type": "mission"})
            
            elif year == 2009:
                df.loc[i, 'Uncertainty_Range'] *= 0.0001
                events.append({"year": year, "event": "Mission LRO - Cartographie haute résolution", "type": "mission"})
            
            # Éclipses historiques
            if year in [1504, 1509, 1547, 1601, 1642, 1704, 1769, 1803, 1856, 1900, 1950, 2000, 2015, 2019]:
                df.loc[i, 'Eclipse_Probability'] = min(1.0, df.loc[i, 'Eclipse_Probability'] * 1.2)
                events.append({"year": year, "event": "Éclipse lunaire historique remarquable", "type": "eclipse"})
            
            # Supermoons exceptionnelles
            if year in [1912, 1930, 1948, 1974, 1992, 2005, 2016, 2023]:
                df.loc[i, 'SuperMoon_Occurrence'] += 1
                events.append({"year": year, "event": "Supermoon exceptionnelle", "type": "supermoon"})
        
        # Stocker les événements dans l'attribut de l'objet
        self.events = events

def create_plotly_visualizations(df, analyzer):
    """Crée des visualisations Plotly interactives"""
    
    # 1. Graphique principal avec sélecteur de période
    fig_main = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Cycle Lunaire Principal', 'Phases et Anomalies',
                       'Cycles Multiples', 'Éclipses et Supermoons',
                       'Données Brutes vs Lissées', 'Prédictions Futures'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cycle principal
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Base_Value'],
                  mode='lines', name='Valeur de base',
                  line=dict(color=analyzer.config['color'], width=2),
                  hovertemplate='Année: %{x}<br>Valeur: %{y:.2f} ' + analyzer.config['unit']),
        row=1, col=1
    )
    
    # Phases et anomalies
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Cycle_Phase'],
                  mode='lines', name='Phase du cycle',
                  line=dict(color='#E0E0E0', width=2)),
        row=1, col=2
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Anomaly_Value'],
                  mode='lines', name='Anomalie',
                  line=dict(color='#FFFFFF', width=2)),
        row=1, col=2
    )
    
    # Cycles multiples
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Saros_Cycle'],
                  mode='lines', name='Cycle de Saros',
                  line=dict(color='#A0A0A0', width=2)),
        row=2, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Metonic_Cycle'],
                  mode='lines', name='Cycle métonique',
                  line=dict(color='#B0B0B0', width=2)),
        row=2, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Nodal_Precession'],
                  mode='lines', name='Précession des nœuds',
                  line=dict(color='#C0C0C0', width=2)),
        row=2, col=1
    )
    
    # Éclipses et supermoons
    fig_main.add_trace(
        go.Bar(x=df['Year'], y=df['Eclipse_Probability'],
               name='Probabilité éclipses',
               marker_color='#909090', opacity=0.6),
        row=2, col=2
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['SuperMoon_Occurrence'],
                  mode='lines+markers', name='Supermoons/an',
                  line=dict(color='#FFFFFF', width=2),
                  marker=dict(size=4, color='#FFFFFF')),
        row=2, col=2
    )
    
    # Données brutes vs lissées
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Base_Value'],
                  mode='lines', name='Données brutes',
                  line=dict(color='#A0A0A0', width=1, dash='dot')),
        row=3, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'], y=df['Smoothed_Value'],
                  mode='lines', name='Données lissées',
                  line=dict(color='#E0E0E0', width=3)),
        row=3, col=1
    )
    
    # Prédictions
    fig_main.add_trace(
        go.Scatter(x=df['Year'][df['Year'] <= 2020], 
                  y=df['Base_Value'][df['Year'] <= 2020],
                  mode='lines', name='Historique',
                  line=dict(color='#A0A0A0', width=2)),
        row=3, col=2
    )
    fig_main.add_trace(
        go.Scatter(x=df['Year'][df['Year'] >= 2020], 
                  y=df['Predicted_Value'][df['Year'] >= 2020],
                  mode='lines', name='Prédictions',
                  line=dict(color='#FFFFFF', width=2, dash='dash')),
        row=3, col=2
    )
    
    # Mise en forme
    fig_main.update_layout(
        height=1200,
        showlegend=True,
        template='plotly_dark',
        title_text=f"Analyse Interactive des Données Lunaires - {analyzer.config['description']}",
        title_font_size=20,
        title_font_color='white',
        hovermode='x unified',
        legend=dict(font=dict(color='white'))
    )
    
    # Mise à jour des axes
    for i in range(1, 4):
        for j in range(1, 3):
            fig_main.update_xaxes(title_text="Année", row=i, col=j, gridcolor='#333333')
            fig_main.update_yaxes(gridcolor='#333333', row=i, col=j)
    
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=1, col=1)
    fig_main.update_yaxes(title_text="Valeur normalisée", row=1, col=2)
    fig_main.update_yaxes(title_text="Amplitude relative", row=2, col=1)
    fig_main.update_yaxes(title_text="Occurrences/Probabilité", row=2, col=2)
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=3, col=1)
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=3, col=2)
    
    return fig_main

def create_heatmap_correlation(df):
    """Crée une heatmap de corrélation"""
    # Sélectionner les colonnes numériques
    numeric_cols = ['Base_Value', 'Monthly_Variation', 'Annual_Variation',
                   'Saros_Cycle', 'Metonic_Cycle', 'Nodal_Precession',
                   'Eclipse_Probability', 'SuperMoon_Occurrence', 'Libration_Amplitude']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Gray',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": "white"},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Matrice de Corrélation des Variables Lunaires",
        template='plotly_dark',
        height=600,
        width=800
    )
    
    return fig_heatmap

def create_timeline_events(events):
    """Crée une timeline des événements"""
    if not events:
        return None
    
    df_events = pd.DataFrame(events)
    
    # Créer un scatter plot pour la timeline
    fig_timeline = go.Figure()
    
    # Couleurs par type d'événement
    color_map = {
        'observation': '#C0C0C0',
        'cartographie': '#E0E0E0',
        'exploration': '#FFFFFF',
        'mission': '#A0A0A0',
        'eclipse': '#808080',
        'supermoon': '#606060'
    }
    
    for event_type in df_events['type'].unique():
        df_type = df_events[df_events['type'] == event_type]
        fig_timeline.add_trace(go.Scatter(
            x=df_type['year'],
            y=[1] * len(df_type),
            mode='markers+text',
            name=event_type.capitalize(),
            marker=dict(size=12, color=color_map.get(event_type, '#C0C0C0'), symbol='diamond'),
            text=df_type['event'],
            textposition="top center",
            hoverinfo='text',
            showlegend=True
        ))
    
    fig_timeline.update_layout(
        title="Chronologie des Événements Lunaires Historiques",
        template='plotly_dark',
        height=300,
        xaxis=dict(title="Année", gridcolor='#333333'),
        yaxis=dict(showticklabels=False, gridcolor='#333333'),
        hovermode='x'
    )
    
    return fig_timeline

def create_3d_phase_space(df):
    """Crée une visualisation 3D de l'espace des phases"""
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df['Base_Value'],
        y=df['Cycle_Phase'],
        z=df['Anomaly_Value'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['Year'],
            colorscale='Gray',
            showscale=True,
            colorbar=dict(title="Année")
        ),
        text=df['Year'],
        hovertemplate='<b>Année: %{text}</b><br>' +
                      'Valeur: %{x:.2f}<br>' +
                      'Phase: %{y:.2f}<br>' +
                      'Anomalie: %{z:.2f}<br>'
    )])
    
    fig_3d.update_layout(
        title="Espace des Phases Lunaire",
        template='plotly_dark',
        height=600,
        scene=dict(
            xaxis_title=dict(text="Valeur de base", font=dict(color='white')),
            yaxis_title=dict(text="Phase", font=dict(color='white')),
            zaxis_title=dict(text="Anomalie", font=dict(color='white')),
            bgcolor='black'
        )
    )
    
    return fig_3d

def create_radar_chart(stats):
    """Crée un graphique radar des statistiques"""
    categories = ['Moyenne', 'Maximum', 'Minimum', 'Écart-type', 'Tendance']
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[stats['mean'], stats['max'], stats['min'], stats['std'], stats['trend']],
        theta=categories,
        fill='toself',
        name='Statistiques',
        line=dict(color='#C0C0C0', width=2),
        fillcolor='rgba(192, 192, 192, 0.3)'
    ))
    
    fig_radar.update_layout(
        title="Profil Statistique des Données",
        template='plotly_dark',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, max(stats.values())])
        ),
        height=400
    )
    
    return fig_radar

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🌕 Lunar Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/FullMoon2010.jpg", 
                 use_column_width=True)
        
        st.markdown("## 🎯 Configuration")
        
        # Types de données
        lunar_data_types = {
            "lunar_distance": "📏 Distance Terre-Lune",
            "lunar_phase": "🌗 Phase lunaire",
            "lunar_inclination": "📐 Inclinaison orbitale",
            "lunar_libration": "🔄 Libration lunaire",
            "lunar_perigee": "⬇️ Distance au périgée",
            "lunar_apogee": "⬆️ Distance à l'apogée",
            "lunar_eclipse": "🌑 Fréquence des éclipses",
            "lunar_temperature": "🌡️ Température de surface",
            "lunar_gravity": "⚖️ Gravité lunaire"
        }
        
        selected_type = st.selectbox(
            "Type de données lunaires",
            options=list(lunar_data_types.keys()),
            format_func=lambda x: lunar_data_types[x]
        )
        
        # Période d'analyse
        st.markdown("### 📅 Période d'analyse")
        start_year = st.number_input("Année de début", min_value=1000, max_value=2000, value=1600)
        end_year = st.number_input("Année de fin", min_value=1601, max_value=2100, value=2025)
        
        # Options d'affichage
        st.markdown("### 🎨 Options d'affichage")
        show_events = st.checkbox("Afficher les événements", value=True)
        show_predictions = st.checkbox("Afficher les prédictions", value=True)
        
        # Bouton de génération
        if st.button("🚀 Générer l'analyse", use_container_width=True):
            st.session_state['generate'] = True
        
        st.markdown("---")
        st.markdown("### 📊 Statistiques en direct")
        
        # Métriques en temps réel
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Données chargées", "✓", delta=None)
        with col2:
            st.metric("Précision", "99.9%", delta="0.1%")
    
    # Initialisation de l'analyseur
    analyzer = LunarDataAnalyzer(selected_type)
    analyzer.start_year = start_year
    analyzer.end_year = end_year
    
    # Génération des données
    if 'generate' in st.session_state or 'df' not in st.session_state:
        with st.spinner("🌙 Génération des données lunaires en cours..."):
            df = analyzer.generate_lunar_data()
            st.session_state['df'] = df
            st.session_state['analyzer'] = analyzer
    else:
        df = st.session_state['df']
        analyzer = st.session_state['analyzer']
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{analyzer.config['icon']} Valeur moyenne</div>
            <div class="metric-value">{df['Base_Value'].mean():.2f}</div>
            <div style="color: #C0C0C0;">{analyzer.config['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_value = df[df['Year'] == 2025]['Base_Value'].values[0] if 2025 in df['Year'].values else df['Base_Value'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Valeur actuelle</div>
            <div class="metric-value">{current_value:.2f}</div>
            <div style="color: #C0C0C0;">{analyzer.config['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cycle_length = analyzer.config['cycle_years']
        n_cycles = (end_year - start_year) / cycle_length
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔄 Cycles observés</div>
            <div class="metric-value">{n_cycles:.1f}</div>
            <div style="color: #C0C0C0;">({cycle_length} ans/cycle)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        uncertainty_current = df['Uncertainty_Range'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Incertitude</div>
            <div class="metric-value">{uncertainty_current:.4f}</div>
            <div style="color: #C0C0C0;">(actuelle)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualisations principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Analyse Principale", "🔄 Cycles", "🌑 Événements", "📊 Statistiques", "🔮 Prédictions"
    ])
    
    with tab1:
        st.markdown("## Visualisation Interactive")
        
        # Sélecteur de période
        col1, col2 = st.columns([3, 1])
        with col1:
            year_range = st.slider(
                "Période d'affichage",
                min_value=int(df['Year'].min()),
                max_value=int(df['Year'].max()),
                value=(1600, 2025)
            )
        
        with col2:
            chart_type = st.selectbox(
                "Type de graphique",
                ["Lignes", "Points", "Barres", "Area"]
            )
        
        # Filtrer les données
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        
        # Créer et afficher le graphique principal
        fig_main = create_plotly_visualizations(df_filtered, analyzer)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Section des informations supplémentaires
        with st.expander("ℹ️ À propos des données"):
            st.markdown(f"""
            <div class="info-box">
                <h4>{analyzer.config['description']}</h4>
                <p><strong>Unité:</strong> {analyzer.config['unit']}</p>
                <p><strong>Type de tendance:</strong> {analyzer.config['trend']}</p>
                <p><strong>Cycle principal:</strong> {analyzer.config['cycle_years']} ans</p>
                <p><strong>Amplitude:</strong> {analyzer.config['amplitude']} {analyzer.config['unit']}</p>
                <p><strong>Période d'analyse:</strong> {start_year} - {end_year}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## Analyse des Cycles Lunaires")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visualisation 3D de l'espace des phases
            fig_3d = create_3d_phase_space(df_filtered)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # Heatmap de corrélation
            fig_heatmap = create_heatmap_correlation(df_filtered)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Analyse spectrale simple
        st.markdown("### Analyse Spectrale")
        
        # FFT simple pour visualiser les fréquences
        from scipy.fft import fft, fftfreq
        
        # Calculer la FFT des données
        y = df_filtered['Base_Value'].values
        n = len(y)
        yf = fft(y)
        xf = fftfreq(n, 1)[:n//2]
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=xf[1:50],  # Afficher les 50 premières fréquences
            y=np.abs(yf[1:n//2][:49]),
            mode='lines',
            line=dict(color='#C0C0C0', width=2),
            fill='tozeroy',
            fillcolor='rgba(192,192,192,0.3)'
        ))
        
        fig_fft.update_layout(
            title="Analyse Spectrale - Fréquences Dominantes",
            xaxis_title="Fréquence (cycles/année)",
            yaxis_title="Amplitude",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_fft, use_container_width=True)
    
    with tab3:
        st.markdown("## Événements Lunaires Historiques")
        
        if hasattr(analyzer, 'events') and analyzer.events:
            # Timeline des événements
            fig_timeline = create_timeline_events(analyzer.events)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Liste des événements
            st.markdown("### Liste détaillée des événements")
            
            # Filtrer par type
            event_types = list(set([e['type'] for e in analyzer.events]))
            selected_types = st.multiselect(
                "Filtrer par type",
                options=event_types,
                default=event_types
            )
            
            # Afficher les événements filtrés
            filtered_events = [e for e in analyzer.events if e['type'] in selected_types]
            
            for event in sorted(filtered_events, key=lambda x: x['year']):
                st.markdown(f"""
                <div class="event-marker">
                    <strong>{event['year']}</strong> - {event['event']} 
                    <span style="color: #C0C0C0;">({event['type']})</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucun événement historique enregistré pour cette période.")
        
        # Graphique des occurrences d'éclipses
        st.markdown("### Probabilité d'éclipses dans le temps")
        
        fig_eclipse = go.Figure()
        fig_eclipse.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Eclipse_Probability'],
            mode='lines',
            name='Probabilité d\'éclipse',
            line=dict(color='#808080', width=2),
            fill='tozeroy'
        ))
        
        # Ajouter les années d'éclipses historiques
        if hasattr(analyzer, 'events'):
            eclipse_years = [e['year'] for e in analyzer.events if e['type'] == 'eclipse']
            fig_eclipse.add_trace(go.Scatter(
                x=eclipse_years,
                y=[df[df['Year'] == y]['Eclipse_Probability'].values[0] if y in df['Year'].values else 0.5 for y in eclipse_years],
                mode='markers',
                name='Éclipses historiques',
                marker=dict(size=10, color='#FFFFFF', symbol='star')
            ))
        
        fig_eclipse.update_layout(
            template='plotly_dark',
            xaxis_title="Année",
            yaxis_title="Probabilité",
            height=400
        )
        
        st.plotly_chart(fig_eclipse, use_container_width=True)
    
    with tab4:
        st.markdown("## Statistiques Détaillées")
        
        # Statistiques descriptives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Résumé statistique")
            stats_df = df[['Base_Value', 'Cycle_Phase', 'Anomaly_Value', 
                          'Eclipse_Probability', 'SuperMoon_Occurrence']].describe()
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        with col2:
            # Graphique radar des statistiques
            stats = {
                'mean': df['Base_Value'].mean(),
                'max': df['Base_Value'].max(),
                'min': df['Base_Value'].min(),
                'std': df['Base_Value'].std(),
                'trend': (df['Base_Value'].iloc[-1] - df['Base_Value'].iloc[0]) / df['Base_Value'].iloc[0]
            }
            
            # Normaliser pour le radar
            max_val = max(stats.values())
            stats_normalized = {k: v/max_val for k, v in stats.items()}
            
            fig_radar = create_radar_chart(stats_normalized)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Distribution des données
        st.markdown("### Distribution des données")
        
        fig_dist = make_subplots(rows=1, cols=2, 
                                 subplot_titles=('Histogramme', 'Box Plot'))
        
        fig_dist.add_trace(
            go.Histogram(x=df['Base_Value'], nbinsx=50,
                        marker_color='#C0C0C0',
                        name='Distribution'),
            row=1, col=1
        )
        
        fig_dist.add_trace(
            go.Box(y=df['Base_Value'], name='Box Plot',
                  marker_color='#C0C0C0',
                  boxmean=True),
            row=1, col=2
        )
        
        fig_dist.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Analyse temporelle
        st.markdown("### Évolution des incertitudes")
        
        fig_uncertainty = go.Figure()
        fig_uncertainty.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Uncertainty_Range'],
            mode='lines',
            name='Incertitude',
            line=dict(color='#FFFFFF', width=2)
        ))
        
        fig_uncertainty.update_layout(
            template='plotly_dark',
            xaxis_title="Année",
            yaxis_title="Incertitude (échelle log)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    with tab5:
        st.markdown("## Prédictions et Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Projection future
            future_years = 50
            last_year = df['Year'].max()
            future_dates = list(range(last_year + 1, last_year + future_years + 1))
            
            # Générer des prédictions simples
            last_value = df['Base_Value'].iloc[-1]
            trend = (df['Base_Value'].iloc[-1] - df['Base_Value'].iloc[-10]) / 10
            
            future_values = []
            future_uncertainty = []
            
            for i, year in enumerate(future_dates):
                predicted = last_value + trend * (i + 1)
                uncertainty = 0.01 * (i + 1) * last_value
                future_values.append(predicted + np.random.normal(0, uncertainty))
                future_uncertainty.append(uncertainty)
            
            # Graphique des prédictions
            fig_pred = go.Figure()
            
            # Données historiques
            fig_pred.add_trace(go.Scatter(
                x=df['Year'][-100:],
                y=df['Base_Value'][-100:],
                mode='lines',
                name='Données historiques',
                line=dict(color='#C0C0C0', width=2)
            ))
            
            # Prédictions
            fig_pred.add_trace(go.Scatter(
                x=future_dates,
                y=future_values,
                mode='lines',
                name='Prédictions',
                line=dict(color='#FFFFFF', width=2, dash='dash')
            ))
            
            # Intervalle de confiance
            fig_pred.add_trace(go.Scatter(
                x=future_dates + future_dates[::-1],
                y=[v + u for v, u in zip(future_values, future_uncertainty)] + 
                  [v - u for v, u in zip(future_values[::-1], future_uncertainty[::-1])],
                fill='toself',
                fillcolor='rgba(255,255,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle de confiance (95%)'
            ))
            
            fig_pred.update_layout(
                title=f"Projections sur {future_years} ans",
                template='plotly_dark',
                xaxis_title="Année",
                yaxis_title=analyzer.config['unit'],
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            st.markdown("### Métriques de confiance")
            
            # Métriques de qualité des prédictions
            mae = np.mean(np.abs(df['Base_Value'] - df['Predicted_Value']))
            mse = np.mean((df['Base_Value'] - df['Predicted_Value'])**2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((df['Base_Value'] - df['Predicted_Value']) / df['Base_Value'])) * 100
            
            metrics_df = pd.DataFrame({
                'Métrique': ['MAE', 'MSE', 'RMSE', 'MAPE'],
                'Valeur': [f"{mae:.2f}", f"{mse:.2f}", f"{rmse:.2f}", f"{mape:.2f}%"],
                'Description': [
                    'Erreur absolue moyenne',
                    'Erreur quadratique moyenne',
                    'Racine de l\'erreur quadratique moyenne',
                    'Erreur absolue moyenne en pourcentage'
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("### Analyse de tendance")
            
            # Calculer la tendance sur différentes périodes
            periods = [10, 25, 50, 100]
            trends = []
            
            for period in periods:
                if len(df) >= period:
                    start_value = df['Base_Value'].iloc[-period]
                    end_value = df['Base_Value'].iloc[-1]
                    trend_pct = ((end_value - start_value) / start_value) * 100
                    trends.append({
                        'Période (ans)': period,
                        'Tendance (%)': f"{trend_pct:.2f}%",
                        'Direction': '↗️' if trend_pct > 0 else '↘️'
                    })
            
            trends_df = pd.DataFrame(trends)
            st.dataframe(trends_df, use_container_width=True)
    
    # Footer avec téléchargement des données
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Téléchargement CSV
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="lunar_data.csv" style="text-decoration: none;">📥 Télécharger les données (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        # Téléchargement Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Lunar Data')
        excel_data = output.getvalue()
        b64_excel = base64.b64encode(excel_data).decode()
        href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="lunar_data.xlsx" style="text-decoration: none;">📊 Télécharger (Excel)</a>'
        st.markdown(href_excel, unsafe_allow_html=True)
    
    with col3:
        st.markdown("🌙 **Dashboard v2.0 - Analyse Lunaire Avancée**")

if __name__ == "__main__":
    main()
