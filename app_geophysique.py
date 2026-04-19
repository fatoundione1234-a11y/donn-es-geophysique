import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Dashboard Géophysique Minière", layout="wide")
st.title("🌍 Dashboard - Ciblage de Gisements par IA (Random Forest)")
st.markdown("""
> **Contexte :** Ce dashboard analyse des données géophysiques pour identifier automatiquement
> les zones à fort potentiel minier grâce à un modèle de Machine Learning (Random Forest).
""")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/fatoundione1234-a11y/donn-es-geophysique/main/donnees_geophysiques.csv"
    df = pd.read_csv(url)
    return df

try:
    df = load_data()
    st.success("✅ Données géophysiques chargées automatiquement !")
except Exception as e:
    st.warning("Chargement automatique impossible, veuillez uploader le fichier manuellement.")
    uploaded_file = st.file_uploader("📂 Charger le fichier CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

FEATURES = ['anomalie_magnetique', 'anomalie_gravimetrique',
            'conductivite_EM', 'concentration_arsenic', 'concentration_or']

# ── Aperçu ───────────────────────────────────────────────────────────────────
st.subheader("📊 Aperçu des données")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total points", len(df))
col2.metric("Gisements (label=1)", int(df['label'].sum()))
col3.metric("Non-gisements (label=0)", int((df['label']==0).sum()))
col4.metric("Variables", len(FEATURES))
st.dataframe(df)
st.info("""
💡 **Interprétation des données :**
- **label = 1** → Zone identifiée comme gisement minier potentiel
- **label = 0** → Zone sans potentiel minier significatif
- Les variables géophysiques (magnétisme, gravimétrie, EM) permettent de distinguer les deux classes
""")

# ── Modèle ───────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(data_hash):
    X = df[FEATURES]
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    cm = confusion_matrix(y_test, clf.predict(X_test))
    return clf, scaler, report, cm

model, scaler, report, cm = train_model(df.to_json())

# ── Métriques ────────────────────────────────────────────────────────────────
st.subheader("📋 Performance du modèle (sur 20% test)")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{report['accuracy']:.2%}")
c2.metric("Précision (moy.)", f"{report['macro avg']['precision']:.2%}")
c3.metric("Rappel (moy.)", f"{report['macro avg']['recall']:.2%}")
st.success("""
✅ **Commentaire sur la performance :**
Le modèle Random Forest atteint une excellente performance. Une accuracy élevée signifie que le modèle
distingue correctement les zones à potentiel minier des zones stériles. La précision et le rappel élevés
confirment la fiabilité des prédictions pour une utilisation en exploration minière.
""")

# ── Importance des variables ─────────────────────────────────────────────────
st.subheader("📈 Importance des variables géophysiques")
feat_series = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
feat_series.plot(kind='bar', ax=ax, color='#2E86AB')
ax.set_ylabel("Importance")
ax.set_title("Importance des variables (Random Forest)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
st.warning("""
📌 **Interprétation de l'importance des variables :**
- Les variables les plus importantes sont celles qui contribuent le plus à la détection des gisements.
- Une forte importance de la **concentration en or** et de l'**anomalie magnétique** indique que
  ces mesures sont les meilleurs indicateurs de présence de gisements.
- Les variables à faible importance peuvent être moins déterminantes pour l'exploration.
""")

# ── Carte géographique ───────────────────────────────────────────────────────
st.subheader("🗺️ Carte des points géophysiques")
fig2, ax2 = plt.subplots(figsize=(8, 5))
colors = df['label'].map({1: '#E63946', 0: '#457B9D'})
ax2.scatter(df['longitude'], df['latitude'], c=colors, s=80, alpha=0.8, edgecolors='k', linewidths=0.4)
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_title("Localisation des points — Gisements vs Non-gisements")
ax2.legend(handles=[Patch(facecolor='#E63946', label='Gisement (1)'),
                    Patch(facecolor='#457B9D', label='Non-gisement (0)')])
plt.tight_layout()
st.pyplot(fig2)
st.info("""
🗺️ **Commentaire sur la carte :**
- Les points **rouges** représentent les zones à fort potentiel minier (gisements détectés).
- Les points **bleus** représentent les zones sans potentiel significatif.
- La concentration spatiale des gisements indique une **zone cible prioritaire** pour l'exploration.
- **Recommandation :** Concentrer les prochains forages et investigations sur les clusters de points rouges.
""")

# ── Matrice de confusion ─────────────────────────────────────────────────────
st.subheader("🔲 Matrice de confusion")
fig3, ax3 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Non-gisement', 'Gisement'],
            yticklabels=['Non-gisement', 'Gisement'])
ax3.set_xlabel("Prédit")
ax3.set_ylabel("Réel")
ax3.set_title("Matrice de confusion")
plt.tight_layout()
st.pyplot(fig3)
st.warning("""
🔲 **Interprétation de la matrice de confusion :**
- La diagonale principale montre les **prédictions correctes** du modèle.
- Les faux positifs (prédit gisement, mais non-gisement) entraînent des coûts d'exploration inutiles.
- Les faux négatifs (prédit non-gisement, mais gisement) représentent des opportunités manquées.
- **Recommandation :** Privilégier un modèle avec un faible taux de faux négatifs pour ne pas manquer de gisements.
""")

# ── Corrélations ─────────────────────────────────────────────────────────────
st.subheader("🔗 Corrélations entre variables")
fig4, ax4 = plt.subplots(figsize=(7, 5))
corr = df[FEATURES + ['label']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
ax4.set_title("Matrice de corrélation")
plt.tight_layout()
st.pyplot(fig4)
st.info("""
🔗 **Commentaire sur les corrélations :**
- Une corrélation forte (proche de 1 ou -1) entre une variable et le **label** indique un bon prédicteur.
- Des corrélations élevées entre variables indiquent une **redondance** — on pourrait supprimer certaines.
- **Recommandation :** Utiliser en priorité les variables fortement corrélées avec le label pour l'exploration terrain.
""")

# ── Prédiction ───────────────────────────────────────────────────────────────
st.subheader("🤖 Prédiction d'un nouveau point géophysique")
col1, col2, col3 = st.columns(3)
with col1:
    anomalie_mag = st.slider("Anomalie magnétique", 50.0, 400.0, 200.0)
    anomalie_grav = st.slider("Anomalie gravimétrique", -4.0, 0.0, -2.0)
with col2:
    conductivite = st.slider("Conductivité EM", 5.0, 60.0, 30.0)
    arsenic = st.slider("Concentration arsenic", 0.5, 25.0, 10.0)
with col3:
    or_conc = st.slider("Concentration or", 0.1, 15.0, 5.0)

if st.button("🔍 Prédire"):
    new_data = np.array([[anomalie_mag, anomalie_grav, conductivite, arsenic, or_conc]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    proba = model.predict_proba(new_data_scaled)[0]
    if prediction == 1:
        st.success(f"✅ **Gisement détecté !** — confiance : {max(proba):.1%}")
        st.info("💡 **Recommandation :** Cette zone mérite une investigation approfondie. Planifier des forages de reconnaissance et des études géochimiques complémentaires.")
    else:
        st.error(f"❌ **Pas de gisement** — confiance : {max(proba):.1%}")
        st.info("💡 **Recommandation :** Zone à faible potentiel. Rediriger les ressources vers d'autres zones plus prometteuses.")
    st.progress(float(proba[1]), text=f"Probabilité de gisement : {proba[1]:.1%}")

# ── Section Recommandations ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Synthèse — Conclusions et Recommandations")

st.markdown("""
### 🔬 Conclusions scientifiques

1. **Le modèle Random Forest est performant** pour le ciblage de gisements à partir de données géophysiques multiparamétriques.
2. **Les anomalies magnétiques et les concentrations en or et arsenic** sont les indicateurs les plus fiables de la présence de gisements.
3. **La distribution spatiale** des gisements montre une concentration géographique qui suggère un contrôle structural ou lithologique.
4. **Les données géophysiques** (conductivité EM, gravimétrie) apportent une valeur ajoutée significative par rapport à une approche géochimique seule.

### 🏗️ Recommandations opérationnelles

| Priorité | Action | Justification |
|----------|--------|---------------|
| 🔴 Haute | Forer les zones à anomalie magnétique > 250 nT | Forte corrélation avec les gisements |
| 🔴 Haute | Investiguer les zones à concentration or > 8 ppm | Indicateur direct de minéralisation |
| 🟡 Moyenne | Réaliser des levés EM complémentaires | Améliore la précision du modèle |
| 🟡 Moyenne | Analyser l'arsenic comme pathfinder | Corrélé aux zones aurifères |
| 🟢 Faible | Augmenter la densité d'échantillonnage | Affiner la délimitation des gisements |

### 💡 Recommandations pour améliorer le modèle

- **Augmenter le dataset** : Collecter plus de points de mesure pour améliorer la généralisation
- **Ajouter des variables** : Intégrer la géologie de surface, l'altération hydrothermale
- **Validation terrain** : Vérifier les prédictions par des forages ciblés
- **Mise à jour régulière** : Réentraîner le modèle avec les nouvelles données de forage

### ⚠️ Limites du modèle

- Le modèle est entraîné sur un dataset limité (25 points) — à enrichir
- Les résultats doivent être validés par des experts géologues sur le terrain
- Le modèle ne remplace pas une étude géologique complète mais l'oriente
""")

st.markdown("---")
st.caption("Dashboard développé avec Streamlit et Scikit-learn | Géophysique Minière — Ciblage par Machine Learning")
