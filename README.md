# 🗺️ Dashboard Géophysique — Ciblage de Gisements Miniers

Pipeline complet de Machine Learning appliqué à la géophysique minière.
Détection automatique de gisements à partir de données magnétiques,
gravimétriques, électromagnétiques et géochimiques.

---

## 📊 Résultats du modèle

| Métrique | Score |
|---|---|
| AUC-ROC | **1.000** |
| Précision gisements | 100% |
| Rappel gisements | 100% |
| Variable clé | Anomalie magnétique (23%) |

---

## 🧪 Variables géophysiques

| Variable | Unité | Rôle |
|---|---|---|
| Anomalie magnétique | nT | Contraste de susceptibilité magnétique |
| Anomalie gravimétrique | mGal | Contraste de densité des roches |
| Conductivité EM | mS/m | Détection des sulfures conducteurs |
| Arsenic (As) | ppm | Pathfinder géochimique de l'or |
| Or (Au) | ppb | Teneur directe en or |

---

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TON_USERNAME/dashboard-geophysique.git
cd dashboard-geophysique

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline complet
python main.py
```

---

## 🎯 Utilisation

```bash
# Lancer avec les données par défaut
python main.py

# Spécifier un fichier de données
python main.py --data mes_donnees/terrain.csv

# Spécifier un dossier de sortie
python main.py --outputs resultats/

# Aide
python main.py --aide
```

---

## 📁 Structure du projet

```
dashboard-geophysique/
│
├── main.py                          ← Point d'entrée principal
├── requirements.txt                 ← Dépendances Python
├── README.md                        ← Ce fichier
├── .gitignore                       ← Fichiers ignorés par Git
│
├── data/
│   └── donnees_geophysiques.csv    ← Données de terrain
│
├── src/
│   ├── modele/
│   │   ├── point_geophysique.py    ← Structure de données
│   │   └── chargement.py           ← Chargement et prétraitement
│   ├── ml/
│   │   └── random_forest.py        ← Modèle Random Forest
│   └── visualisation/
│       └── graphiques.py           ← Tous les graphiques
│
├── outputs/                         ← Graphiques générés (auto)
│   ├── 01_exploration_donnees.png
│   ├── 02_resultats_random_forest.png
│   ├── 03_carte_potentiel_mineral.png
│   └── 04_synthese.png
│
├── notebooks/                       ← Analyses Jupyter (optionnel)
└── docs/                            ← Documentation
```

---

## 🔬 Méthodologie

### 1. Acquisition des données
Mesures géophysiques multi-paramètres sur une grille de terrain.
Chaque point combine 5 variables complémentaires.

### 2. Exploration et visualisation
Analyse des distributions, corrélations et séparation des classes
(gisements connus vs zones stériles).

### 3. Machine Learning — Random Forest
- 100 arbres de décision en parallèle
- Profondeur maximale : 5 niveaux
- Pondération des classes pour compenser le déséquilibre
- Validation croisée spatiale recommandée pour de futurs travaux

### 4. Carte de potentiel minéral
Prédiction de probabilité sur tout le territoire exploré.
Identification des zones prioritaires pour le forage.

---

## ⚠️ Recommandations

1. **Risque d'overfitting** : AUC = 1.0 sur 25 points → augmenter à 100+ points
2. **Validation terrain** : Toujours confirmer par échantillonnage géochimique
3. **Nouvelles variables** : Ajouter Cu, Pb, Zn pour mieux typer le gisement
4. **Sismique** : Intégrer des données de sismique réfraction pour la profondeur

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.5+-150458?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.6+-11557C)

---

## 🤝 Contribuer

1. Forker le dépôt
2. Créer une branche : `git checkout -b feature/nouvelle-fonctionnalite`
3. Committer : `git commit -m "Ajout de XGBoost"`
4. Pousser : `git push origin feature/nouvelle-fonctionnalite`
5. Ouvrir une Pull Request

---

## 📄 Licence

MIT License — libre d'utilisation, modification et distribution.
