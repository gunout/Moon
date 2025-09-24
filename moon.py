import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LunarDataAnalyzer:
    def __init__(self, data_type):
        self.data_type = data_type
        self.colors = ['#C0C0C0', '#F0F0F0', '#E0E0E0', '#D0D0D0', '#B0B0B0', 
                      '#A0A0A0', '#909090', '#808080', '#707070', '#606060']
        
        self.start_year = 1600  # Début des observations lunaires précises
        self.end_year = 2025
        
        # Configuration spécifique pour chaque type de données lunaires
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
                "description": "Distance Terre-Lune"
            },
            "lunar_phase": {
                "base_value": 0.5,
                "cycle_years": 29.53/365.25,  # Cycle synodique en années
                "amplitude": 0.5,
                "trend": "cyclique",
                "unit": "Phase (0-1)",
                "description": "Phase lunaire"
            },
            "lunar_inclination": {
                "base_value": 5.15,
                "cycle_years": 18.6,
                "amplitude": 0.5,
                "trend": "cyclique",
                "unit": "degrés",
                "description": "Inclinaison orbitale lunaire"
            },
            "lunar_libration": {
                "base_value": 0,
                "cycle_years": 27.3,
                "amplitude": 7.5,
                "trend": "cyclique",
                "unit": "degrés",
                "description": "Libration lunaire"
            },
            "lunar_perigee": {
                "base_value": 363000,
                "cycle_years": 8.85,
                "amplitude": 2000,
                "trend": "cyclique",
                "unit": "km",
                "description": "Distance au périgée"
            },
            "lunar_apogee": {
                "base_value": 405000,
                "cycle_years": 8.85,
                "amplitude": 2000,
                "trend": "cyclique",
                "unit": "km",
                "description": "Distance à l'apogée"
            },
            "lunar_eclipse": {
                "base_value": 0,
                "cycle_years": 18.03,
                "amplitude": 1,
                "trend": "cyclique",
                "unit": "occurrences/an",
                "description": "Fréquence des éclipses lunaires"
            },
            "lunar_temperature": {
                "base_value": -20,
                "cycle_years": 29.53/365.25,
                "amplitude": 150,
                "trend": "cyclique",
                "unit": "°C",
                "description": "Température de surface lunaire"
            },
            "lunar_gravity": {
                "base_value": 1.62,
                "cycle_years": 27.3,
                "amplitude": 0.01,
                "trend": "stable",
                "unit": "m/s²",
                "description": "Gravité lunaire"
            },
            # Configuration par défaut
            "default": {
                "base_value": 100,
                "cycle_years": 27.3,
                "amplitude": 50,
                "trend": "cyclique",
                "unit": "Unités",
                "description": "Données lunaires génériques"
            }
        }
        
        return configs.get(self.data_type, configs["default"])
    
    def generate_lunar_data(self):
        """Génère des données lunaires simulées basées sur les cycles lunaires réels"""
        print(f"🌕 Génération des données lunaires pour {self.config['description']}...")
        
        # MODIFICATION : Utiliser une liste d'années au lieu de dates pandas
        years = list(range(self.start_year, self.end_year + 1))
        
        data = {'Year': years}
        
        # Simuler les données pour chaque année
        data['Base_Value'] = self._simulate_lunar_cycle(years)
        data['Monthly_Variation'] = self._simulate_monthly_variation(years)
        data['Annual_Variation'] = self._simulate_annual_variation(years)
        
        # Cycles spécifiques à la Lune
        data['Saros_Cycle'] = self._simulate_saros_cycle(years)
        data['Metonic_Cycle'] = self._simulate_metonic_cycle(years)
        data['Nodal_Precession'] = self._simulate_nodal_precession(years)
        
        # Phénomènes lunaires
        data['Eclipse_Probability'] = self._simulate_eclipse_probability(years)
        data['SuperMoon_Occurrence'] = self._simulate_supermoon_occurrence(years)
        data['Libration_Amplitude'] = self._simulate_libration_amplitude(years)
        
        # Données dérivées
        data['Smoothed_Value'] = self._simulate_smoothed_data(years)
        data['Cycle_Phase'] = self._simulate_cycle_phase(years)
        data['Anomaly_Value'] = self._simulate_anomaly_data(years)
        
        # Prédictions et tendances
        data['Secular_Trend'] = self._simulate_secular_trend(years)
        data['Predicted_Value'] = self._simulate_predicted_data(years)
        data['Uncertainty_Range'] = self._simulate_uncertainty_range(years)
        
        df = pd.DataFrame(data)
        
        # Ajouter des événements lunaires historiques
        self._add_lunar_events(df)
        
        return df
    
    def _simulate_lunar_cycle(self, years):
        """Simule le cycle lunaire principal"""
        base_value = self.config["base_value"]
        cycle_years = self.config["cycle_years"]
        amplitude = self.config["amplitude"]
        
        values = []
        for year in years:
            # Cycle lunaire de base
            phase = (year - self.start_year) % cycle_years
            cycle_value = np.sin(2 * np.pi * phase / cycle_years)
            
            # Ajustement pour différents types de données
            if self.data_type == "lunar_distance":
                # Variation de distance Terre-Lune
                value = base_value + amplitude * cycle_value
            elif self.data_type == "lunar_phase":
                # Phase lunaire (0 = nouvelle lune, 0.5 = pleine lune)
                value = 0.5 + 0.5 * np.sin(2 * np.pi * phase / cycle_years)
            elif self.data_type == "lunar_temperature":
                # Température extrême (-173°C à +127°C)
                value = base_value + amplitude * cycle_value
            else:
                value = base_value + amplitude * cycle_value
            
            # Bruit naturel
            noise = np.random.normal(0, amplitude * 0.05)
            values.append(value + noise)
        
        return values
    
    def _simulate_monthly_variation(self, years):
        """Simule les variations mensuelles (cycle synodique)"""
        variations = []
        for year in years:
            # Variation basée sur le cycle synodique (29.53 jours)
            # Utiliser une valeur fixe pour la journée de l'année (jour 180)
            synodic_phase = (180 / 365.25) * (365.25 / 29.53)
            monthly_variation = 0.1 * np.sin(2 * np.pi * synodic_phase)
            variations.append(1 + monthly_variation)
        
        return variations
    
    def _simulate_annual_variation(self, years):
        """Simule les variations annuelles"""
        variations = []
        for year in years:
            # Variation annuelle basée sur la position orbitale
            annual_variation = 0.02 * np.sin(2 * np.pi * (year - self.start_year) / 1)
            variations.append(1 + annual_variation)
        
        return variations
    
    def _simulate_saros_cycle(self, years):
        """Simule le cycle de Saros (18.03 ans) pour les éclipses"""
        saros_values = []
        for year in years:
            saros_phase = (year - self.start_year) % 18.03
            
            # Maximum d'éclipses au milieu du cycle
            saros_value = np.sin(2 * np.pi * saros_phase / 18.03)
            saros_values.append(0.5 + 0.5 * saros_value)
        
        return saros_values
    
    def _simulate_metonic_cycle(self, years):
        """Simule le cycle métonique (19 ans)"""
        metonic_values = []
        for year in years:
            metonic_phase = (year - self.start_year) % 19.0
            
            # Cycle métonique - alignement des phases lunaires avec l'année solaire
            metonic_value = np.sin(2 * np.pi * metonic_phase / 19.0)
            metonic_values.append(0.5 + 0.5 * metonic_value)
        
        return metonic_values
    
    def _simulate_nodal_precession(self, years):
        """Simule la précession des nœuds orbitaux (18.6 ans)"""
        precession_values = []
        for year in years:
            precession_phase = (year - self.start_year) % 18.6
            
            # Précession qui affecte les éclipses
            precession_value = np.sin(2 * np.pi * precession_phase / 18.6)
            precession_values.append(precession_value)
        
        return precession_values
    
    def _simulate_eclipse_probability(self, years):
        """Simule la probabilité d'éclipses lunaires"""
        probabilities = []
        for i, year in enumerate(years):
            # Combinaison des cycles de Saros et de précession
            saros_factor = self._simulate_saros_cycle([year])[0]
            precession_factor = abs(self._simulate_nodal_precession([year])[0])
            
            # Probabilité maximale quand les cycles s'alignent
            probability = 0.3 + 0.4 * saros_factor * precession_factor
            probabilities.append(min(probability, 0.9))
        
        return probabilities
    
    def _simulate_supermoon_occurrence(self, years):
        """Simule l'occurrence des supermoons"""
        occurrences = []
        for year in years:
            # Supermoon quand la pleine lune coïncide avec le périgée
            # Environ 3-4 fois par an
            base_occurrence = 3.5  # Nombre moyen par an
            
            # Variation cyclique
            cycle_variation = 0.5 * np.sin(2 * np.pi * (year - self.start_year) / 14.0)
            occurrence = base_occurrence + cycle_variation
            
            occurrences.append(max(1, occurrence))  # Au moins 1 par an
        
        return occurrences
    
    def _simulate_libration_amplitude(self, years):
        """Simule l'amplitude de la libration lunaire"""
        amplitudes = []
        for year in years:
            # Libration varie avec la distance et l'inclinaison
            base_amplitude = 7.0  # degrés
            variation = 0.5 * np.sin(2 * np.pi * (year - self.start_year) / 27.3)
            
            amplitude = base_amplitude + variation
            amplitudes.append(amplitude)
        
        return amplitudes
    
    def _simulate_smoothed_data(self, years):
        """Simule des données lissées"""
        base_cycle = self._simulate_lunar_cycle(years)
        
        smoothed = []
        for i in range(len(base_cycle)):
            # Moyenne mobile sur 5 points
            start_idx = max(0, i - 2)
            end_idx = min(len(base_cycle), i + 3)
            window = base_cycle[start_idx:end_idx]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _simulate_cycle_phase(self, years):
        """Simule la phase du cycle principal (0-1)"""
        phases = []
        cycle_years = self.config["cycle_years"]
        
        for year in years:
            phase = (year - self.start_year) % cycle_years / cycle_years
            phases.append(phase)
        
        return phases
    
    def _simulate_anomaly_data(self, years):
        """Simule les anomalies par rapport à la moyenne"""
        base_cycle = self._simulate_lunar_cycle(years)
        smoothed = self._simulate_smoothed_data(years)
        
        anomalies = []
        for i in range(len(base_cycle)):
            anomaly = base_cycle[i] - smoothed[i]
            anomalies.append(anomaly)
        
        return anomalies
    
    def _simulate_secular_trend(self, years):
        """Simule les tendances séculaires à long terme"""
        trends = []
        for year in years:
            # Tendances à très long terme
            if self.data_type == "lunar_distance":
                # La Lune s'éloigne de 3.8 cm/an
                years_from_start = year - self.start_year
                trend = 1 + (years_from_start * 0.000038) / self.config["base_value"]
            else:
                trend = 1.0
            
            trends.append(trend)
        
        return trends
    
    def _simulate_predicted_data(self, years):
        """Simule des données prédites"""
        predictions = []
        base_cycle = self._simulate_lunar_cycle(years)
        secular_trend = self._simulate_secular_trend(years)
        
        for i, year in enumerate(years):
            current_value = base_cycle[i]
            trend_factor = secular_trend[i]
            
            if year > 2020:  # Période de prédiction
                # Ajouter une incertitude croissante
                years_since_2020 = year - 2020
                uncertainty = 0.01 * years_since_2020
                prediction = current_value * trend_factor * (1 + np.random.normal(0, uncertainty))
            else:
                prediction = current_value
            
            predictions.append(prediction)
        
        return predictions
    
    def _simulate_uncertainty_range(self, years):
        """Simule la plage d'incertitude des mesures"""
        uncertainties = []
        for year in years:
            # Incertitude diminue avec le temps (technologies améliorées)
            if year < 1700:
                uncertainty = 0.1  # 10% avant les mesures précises
            elif year < 1900:
                uncertainty = 0.05  # 5% avec les premiers télescopes
            elif year < 1960:
                uncertainty = 0.01  # 1% avec les mesures modernes
            else:
                uncertainty = 0.001  # 0.1% avec la télémétrie laser
            
            uncertainties.append(uncertainty)
        
        return uncertainties
    
    def _add_lunar_events(self, df):
        """Ajoute des événements lunaires historiques significatifs"""
        for i, row in df.iterrows():
            year = row['Year']
            
            # Événements lunaires historiques
            if year == 1609:
                # Premières observations télescopiques de Galilée
                df.loc[i, 'Anomaly_Value'] *= 1.1
            
            elif year == 1837:
                # Première carte détaillée de la Lune (Beer et Mädler)
                df.loc[i, 'Uncertainty_Range'] *= 0.5
            
            elif year == 1969:
                # Apollo 11 - premiers pas sur la Lune
                df.loc[i, 'Uncertainty_Range'] *= 0.01
                df.loc[i, 'Base_Value'] = self.config["base_value"]  # Valeur précise
            
            elif year == 1994:
                # Mission Clementine - cartographie complète
                df.loc[i, 'Uncertainty_Range'] *= 0.001
            
            elif year == 2009:
                # Mission LRO (Lunar Reconnaissance Orbiter)
                df.loc[i, 'Uncertainty_Range'] *= 0.0001
            
            # Éclipses lunaires historiques remarquables
            if year in [1504, 1509, 1547, 1601, 1642, 1704, 1769, 1803, 1856, 1900, 1950, 2000, 2015, 2019]:
                df.loc[i, 'Eclipse_Probability'] = min(1.0, df.loc[i, 'Eclipse_Probability'] * 1.2)
            
            # Supermoons exceptionnelles
            if year in [1912, 1930, 1948, 1974, 1992, 2005, 2016, 2023]:
                df.loc[i, 'SuperMoon_Occurrence'] += 1
    
    def create_lunar_analysis(self, df):
        """Crée une analyse complète des données lunaires"""
        plt.style.use('dark_background')  # Fond sombre pour l'astronomie
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Cycle lunaire principal
        ax1 = plt.subplot(5, 2, 1)
        self._plot_lunar_cycle(df, ax1)
        
        # 2. Données historiques
        ax2 = plt.subplot(5, 2, 2)
        self._plot_historical_data(df, ax2)
        
        # 3. Cycles multiples
        ax3 = plt.subplot(5, 2, 3)
        self._plot_multiple_cycles(df, ax3)
        
        # 4. Phases et anomalies
        ax4 = plt.subplot(5, 2, 4)
        self._plot_phases_anomalies(df, ax4)
        
        # 5. Éclipses et supermoons
        ax5 = plt.subplot(5, 2, 5)
        self._plot_eclipses_supermoons(df, ax5)
        
        # 6. Données lissées
        ax6 = plt.subplot(5, 2, 6)
        self._plot_smoothed_data(df, ax6)
        
        # 7. Incertitudes de mesure
        ax7 = plt.subplot(5, 2, 7)
        self._plot_uncertainties(df, ax7)
        
        # 8. Tendances séculaires
        ax8 = plt.subplot(5, 2, 8)
        self._plot_secular_trends(df, ax8)
        
        # 9. Prédictions futures
        ax9 = plt.subplot(5, 2, 9)
        self._plot_predictions(df, ax9)
        
        # 10. Analyse comparative
        ax10 = plt.subplot(5, 2, 10)
        self._plot_comparative_analysis(df, ax10)
        
        plt.suptitle(f'Analyse des Données Lunaires: {self.config["description"]} ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()
        plt.savefig(f'lunar_{self.data_type}_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.show()
        
        # Générer les insights
        self._generate_lunar_insights(df)
    
    def _plot_lunar_cycle(self, df, ax):
        """Plot du cycle lunaire principal"""
        ax.plot(df['Year'], df['Base_Value'], label='Valeur de base', 
               linewidth=2, color='#C0C0C0', alpha=0.9)
        
        ax.set_title(f'Cycle Lunaire Principal - {self.config["description"]}', 
                    fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel(self.config["unit"], color='#C0C0C0')
        ax.tick_params(axis='y', labelcolor='#C0C0C0')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # CORRECTION : Vérifier que le cycle n'est pas trop petit avant de tracer les lignes verticales
        cycle_years = self.config["cycle_years"]
        if cycle_years >= 1:  # Ne tracer que pour les cycles d'au moins 1 an
            for year in range(int(self.start_year), int(self.end_year), max(1, int(cycle_years))):
                if year in df['Year'].values:
                    ax.axvline(x=year, alpha=0.3, color='silver', linestyle='--')
    
    def _plot_historical_data(self, df, ax):
        """Plot des données historiques"""
        ax.fill_between(df['Year'], df['Base_Value'], alpha=0.7, 
                       color='#E0E0E0', label='Données historiques')
        
        ax.set_title('Évolution Historique des Données Lunaires', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel(self.config["unit"], color='#E0E0E0')
        ax.set_xlabel('Année', color='white')
        ax.tick_params(axis='y', labelcolor='#E0E0E0')
        ax.tick_params(axis='x', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # Marquer les événements importants
        events = {
            1609: 'Galilée\n(télescope)',
            1837: 'Beer & Mädler\n(carte lunaire)',
            1969: 'Apollo 11\n(premiers pas)',
            2009: 'LRO\n(cartographie)'
        }
        
        for year, label in events.items():
            if year in df['Year'].values:
                y_val = df.loc[df['Year'] == year, 'Base_Value'].values[0]
                ax.annotate(label, xy=(year, y_val), xytext=(year, y_val*1.1),
                           arrowprops=dict(arrowstyle='->', color='silver'),
                           color='silver', fontsize=8, ha='center')
    
    def _plot_multiple_cycles(self, df, ax):
        """Plot des cycles multiples de la Lune"""
        ax.plot(df['Year'], df['Saros_Cycle'], label='Cycle de Saros (18.03 ans)', 
               alpha=0.7, color='#A0A0A0')
        ax.plot(df['Year'], df['Metonic_Cycle'], label='Cycle métonique (19 ans)', 
               alpha=0.7, color='#B0B0B0')
        ax.plot(df['Year'], df['Nodal_Precession'], label='Précession des nœuds (18.6 ans)', 
               alpha=0.7, color='#C0C0C0')
        
        ax.set_title('Cycles Multiples de la Lune', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Amplitude relative', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_phases_anomalies(self, df, ax):
        """Plot des phases et anomalies"""
        ax.plot(df['Year'], df['Cycle_Phase'], label='Phase du cycle', 
               color='#D0D0D0', alpha=0.7)
        ax.plot(df['Year'], df['Anomaly_Value'], label='Anomalie', 
               color='#F0F0F0', alpha=0.7)
        
        ax.set_title('Phase du Cycle et Anomalies', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Valeur normalisée', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_eclipses_supermoons(self, df, ax):
        """Plot des éclipses et supermoons"""
        ax.bar(df['Year'], df['Eclipse_Probability'], label='Probabilité d\'éclipses', 
              alpha=0.6, color='#909090')
        ax.plot(df['Year'], df['SuperMoon_Occurrence'], label='Supermoons/an', 
               color='#FFFFFF', linewidth=2)
        
        ax.set_title('Éclipses Lunaires et Supermoons', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Occurrences/Probabilité', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_smoothed_data(self, df, ax):
        """Plot des données lissées"""
        ax.plot(df['Year'], df['Base_Value'], label='Données brutes', 
               alpha=0.5, color='#A0A0A0')
        ax.plot(df['Year'], df['Smoothed_Value'], label='Données lissées', 
               linewidth=2, color='#E0E0E0')
        
        ax.set_title('Données Brutes vs Lissées', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_uncertainties(self, df, ax):
        """Plot des incertitudes de mesure"""
        ax.semilogy(df['Year'], df['Uncertainty_Range'], label='Incertitude de mesure', 
                   color='#C0C0C0', linewidth=2)
        
        ax.set_title('Évolution des Incertitudes de Mesure', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Incertitude (échelle logarithmique)', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_secular_trends(self, df, ax):
        """Plot des tendances séculaires"""
        ax.plot(df['Year'], df['Secular_Trend'], label='Tendance séculaire', 
               linewidth=2, color='#F0F0F0')
        
        ax.set_title('Tendances Séculaires à Long Terme', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Facteur multiplicatif', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_predictions(self, df, ax):
        """Plot des prédictions futures"""
        ax.plot(df['Year'], df['Base_Value'], label='Données historiques', 
               color='#A0A0A0', alpha=0.7)
        ax.plot(df['Year'], df['Predicted_Value'], label='Projections futures', 
               linewidth=2, color='#FFFFFF', linestyle='--')
        
        ax.axvline(x=2020, color='silver', linestyle=':', alpha=0.7, label='Début des prédictions')
        
        ax.set_title('Données Historiques et Projections Futures', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_comparative_analysis(self, df, ax):
        """Plot d'analyse comparative"""
        # Normaliser les données pour la comparaison
        normalized_base = (df['Base_Value'] - df['Base_Value'].min()) / (df['Base_Value'].max() - df['Base_Value'].min())
        normalized_smoothed = (df['Smoothed_Value'] - df['Smoothed_Value'].min()) / (df['Smoothed_Value'].max() - df['Smoothed_Value'].min())
        
        ax.plot(df['Year'], normalized_base, label='Données brutes (normalisées)', 
               alpha=0.7, color='#B0B0B0')
        ax.plot(df['Year'], normalized_smoothed, label='Données lissées (normalisées)', 
               alpha=0.7, color='#D0D0D0')
        ax.plot(df['Year'], df['Cycle_Phase'], label='Phase du cycle', 
               alpha=0.7, color='#F0F0F0')
        
        ax.set_title('Analyse Comparative (Données Normalisées)', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Valeur normalisée', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _generate_lunar_insights(self, df):
        """Génère des insights analytiques sur les données lunaires"""
        print(f"🌕 INSIGHTS ANALYTIQUES - {self.config['description']}")
        print("=" * 70)
        
        # 1. Statistiques de base
        print("\n1. 📊 STATISTIQUES FONDAMENTALES:")
        avg_value = df['Base_Value'].mean()
        max_value = df['Base_Value'].max()
        min_value = df['Base_Value'].min()
        current_value = df['Base_Value'].iloc[-1]
        
        print(f"Valeur moyenne: {avg_value:.2f} {self.config['unit']}")
        print(f"Valeur maximale: {max_value:.2f} {self.config['unit']}")
        print(f"Valeur minimale: {min_value:.2f} {self.config['unit']}")
        print(f"Valeur actuelle: {current_value:.2f} {self.config['unit']}")
        
        # 2. Analyse des cycles
        print("\n2. 🔄 ANALYSE DES CYCLES LUNAIRES:")
        cycle_length = self.config["cycle_years"]
        n_cycles = (self.end_year - self.start_year) / cycle_length
        
        print(f"Durée du cycle: {cycle_length} années")
        print(f"Nombre de cycles observés: {n_cycles:.1f}")
        print(f"Type de tendance: {self.config['trend']}")
        
        # 3. Cycles multiples
        print("\n3. 🌙 CYCLES MULTIPLES IMPORTANTS:")
        print("• Cycle synodique: 29.53 jours (phases lunaires)")
        print("• Cycle draconitique: 27.21 jours (éclipses)")
        print("• Cycle anomalistique: 27.55 jours (distance)")
        print("• Cycle de Saros: 18.03 ans (récurrence éclipses)")
        print("• Cycle métonique: 19 ans (phases ↔ années solaires)")
        print("• Précession des nœuds: 18.6 ans (éclipses limites)")
        
        # 4. Événements majeurs
        print("\n4. 📅 ÉVÉNEMENTS LUNAIRES MARQUANTS:")
        print("• 1609: Premières observations télescopiques (Galilée)")
        print("• 1837: Première carte détaillée (Beer et Mädler)")
        print("• 1969: Apollo 11 - premiers pas sur la Lune")
        print("• 1994: Mission Clementine - cartographie complète")
        print("• 2009: LRO - cartographie haute résolution")
        
        # 5. Caractéristiques actuelles
        print("\n5. 🔭 CARACTÉRISTIQUES ACTUELLES:")
        phase_current = df['Cycle_Phase'].iloc[-1]
        uncertainty_current = df['Uncertainty_Range'].iloc[-1]
        
        print(f"Phase actuelle du cycle: {phase_current:.2f}")
        print(f"Incertitude actuelle: {uncertainty_current:.4f}")
        print(f"Distance actuelle Terre-Lune: ~384,400 km")
        print(f"Éloignement actuel: 3.8 cm/an")
        
        # 6. Projections futures
        print("\n6. 🔮 PROJECTIONS FUTURES:")
        if self.data_type == "lunar_distance":
            years_1000 = 1000
            distance_increase = years_1000 * 0.000038  # km
            print(f"Dans 1000 ans: Éloignement de {distance_increase:.1f} km")
            print(f"Période orbitale: Augmentation de ~0.002 secondes/an")
        
        print("• Stabilisation de la rotation terrestre par effet de freinage")
        print("• Augmentation progressive de la durée du jour terrestre")
        print("• Modification des marées océaniques")
        
        # 7. Implications scientifiques
        print("\n7. 🎯 IMPLICATIONS SCIENTIFIQUES:")
        if self.data_type == "lunar_distance":
            print("• Étude de l'évolution du système Terre-Lune")
            print("• Calibration des constantes astronomiques")
            print("• Tests de la relativité générale")
        
        elif self.data_type == "lunar_phase":
            print("• Calendriers et cycles culturels")
            print("• Effets sur les marées et écosystèmes")
            print("• Influence sur l'observation astronomique")
        
        elif self.data_type == "lunar_eclipse":
            print("• Étude de l'atmosphère terrestre")
            print("• Calibration des modèles orbitaux")
            print("• Observations historiques pour la datation")
        
        print("• Base pour l'exploration spatiale future")
        print("• Étude de la formation du système solaire")
        print("• Tests des théories gravitationnelles")

def main():
    """Fonction principale pour l'analyse des données lunaires"""
    # Types de données lunaires disponibles
    lunar_data_types = [
        "lunar_distance", "lunar_phase", "lunar_inclination", "lunar_libration",
        "lunar_perigee", "lunar_apogee", "lunar_eclipse", "lunar_temperature", "lunar_gravity"
    ]
    
    print("🌕 ANALYSE DES DONNÉES NUMÉRIQUES DE LA LUNE (1600-2025)")
    print("=" * 65)
    
    # Demander à l'utilisateur de choisir un type de données
    print("Types de données lunaires disponibles:")
    for i, data_type in enumerate(lunar_data_types, 1):
        analyzer_temp = LunarDataAnalyzer(data_type)
        print(f"{i}. {analyzer_temp.config['description']}")
    
    try:
        choix = int(input("\nChoisissez le numéro du type de données à analyser: "))
        if choix < 1 or choix > len(lunar_data_types):
            raise ValueError
        selected_type = lunar_data_types[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de la distance Terre-Lune par défaut.")
        selected_type = "lunar_distance"
    
    # Initialiser l'analyseur
    analyzer = LunarDataAnalyzer(selected_type)
    
    # Générer les données
    lunar_data = analyzer.generate_lunar_data()
    
    # Sauvegarder les données
    output_file = f'lunar_{selected_type}_data_1600_2025.csv'
    lunar_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(lunar_data[['Year', 'Base_Value', 'Cycle_Phase', 'Uncertainty_Range']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse des données lunaires...")
    analyzer.create_lunar_analysis(lunar_data)
    
    print(f"\n✅ Analyse des données {analyzer.config['description']} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("🌙 Données: Cycles lunaires, distances, phases, éclipses, prédictions")

if __name__ == "__main__":
    main()