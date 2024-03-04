Voici un exemple de README pour votre projet :

---

# Projet argoPXPCM

Ce projet permet de récupérer et d'analyser les données provenant de la flotte de bouées Argo, puis de les classifier à l'aide du modèle PCM (Probabilistic Counting Model).

## Prérequis

- Python 3.7 ou supérieur
- Conda (pour gérer l'environnement)
- Streamlit
- Xarray
- Argopy
- Gsw (Thermodynamic Equation of Seawater 2010)

## Installation

1. Cloner le dépôt Git :
   ```bash
   git clone https://github.com/votre-utilisateur/argoPXPCM.git
   ```

2. Créer l'environnement Conda à partir du fichier `environment.yml` :
   ```bash
   conda env create -f environment.yml
   ```

3. Activer l'environnement Conda :
   ```bash
   conda activate MPA
   ```

## Lancer l'application

1. Pour démarrer l'application Streamlit, exécutez :
   ```bash
   streamlit run main.py
   ```

2. Accédez à l'application via votre navigateur web à l'adresse indiquée.

## Instructions d'utilisation

- **Récupérer les données Argo** :
  - Utilisez les paramètres dans la zone de sélection pour définir la zone géographique, la profondeur, et la période temporelle.
  - Cliquez sur le bouton "Récupérer les données" pour récupérer les données Argo correspondantes.

- **Classifier les données** :
  - Définissez les paramètres du modèle PCM (nombre de clusters, variables à utiliser, etc.) dans la zone de paramètres PyXPCM.
  - Cliquez sur le bouton "Classifier les données" pour appliquer le modèle PCM aux données récupérées.

- **Visualiser les résultats** :
  - Les résultats seront affichés sur une carte interactive (avec les données Argo) et des graphiques (quantiles, scalers, réduction de dimension, etc.).
  - Utilisez les options de la barre latérale pour personnaliser les graphiques affichés.

## Fonctions principales

### Récupération des données Argo

- `recup_argo_data(llon, rlon, llat, ulat, depthmin, depthmax, intervalle, time_in, time_f)`

### Classification avec le modèle PCM

- `pyxpcm_sal_temp(da, k, quan, varmax)`: Profil de salinité et température
- `pyxpcm_sal(da, k, quan, varmax)`: Profil de salinité uniquement
- `pyxpcm_temp(da, k, quan, varmax)`: Profil de température uniquement

## Variables globales

- `ds`: Dataset principal pour les données Argo.
- `df_points`: DataFrame pour stocker les données à afficher sur la carte.
- `button_fetch_data_pressed`: Indique si le bouton "Récupérer les données" a été pressé.
- `button_class_data_pressed`: Indique si le bouton "Classifier les données" a été pressé.
- `m`: Modèle PCM utilisé pour la classification.
- `ds_pcm`: Dataset pour les résultats de la classification.
- `graphs_updated`: Indique si les graphiques ont été mis à jour.
- `graphs`: Dictionnaire pour stocker les graphiques à afficher.



---

N'oubliez pas de personnaliser les liens vers votre dépôt GitHub et votre profil utilisateur. Vous pouvez également ajouter des captures d'écran ou des exemples d'utilisation dans la section appropriée.