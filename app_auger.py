import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

np.random.seed(42)

# ── Génération des données synthétiques ─────────────────────────────────────
# Zone Afrique de l'Ouest (style Sénégal/Mali)
BASE_LON = -11.250
BASE_LAT = 12.800
ESPACEMENT_LIGNE = 0.002   # ~220m entre lignes
ESPACEMENT_POINT = 0.0015  # ~165m entre points

N_LIGNES = 10
N_POINTS = 7

statuts_possibles = ['Foré', 'Stoppé', 'En cours']
proba_statuts = [0.55, 0.30, 0.15]

records = []
for i in range(N_LIGNES):
    for j in range(N_POINTS):
        lon = BASE_LON + j * ESPACEMENT_POINT + np.random.normal(0, 0.00005)
        lat = BASE_LAT + i * ESPACEMENT_LIGNE + np.random.normal(0, 0.00005)
        statut = np.random.choice(statuts_possibles, p=proba_statuts)
        profondeur = np.random.uniform(3, 25) if statut != 'Stoppé' else np.random.uniform(1, 8)
        au = np.random.lognormal(-1, 1.2) if statut == 'Foré' else np.nan
        cu = np.random.uniform(5, 150) if statut == 'Foré' else np.nan
        as_ = np.random.uniform(2, 80) if statut == 'Foré' else np.nan
        fe = np.random.uniform(1, 35) if statut == 'Foré' else np.nan
        mn = np.random.uniform(0.1, 15) if statut == 'Foré' else np.nan
        records.append({
            'ligne': f'L{i+1:02d}',
            'trou': f'L{i+1:02d}T{j+1:02d}',
            'longitude': round(lon, 6),
            'latitude': round(lat, 6),
            'statut': statut,
            'profondeur_m': round(profondeur, 1),
            'Au_ppb': round(au, 2) if not np.isnan(au) else np.nan,
            'Cu_ppm': round(cu, 1) if not np.isnan(cu) else np.nan,
            'As_ppm': round(as_, 1) if not np.isnan(as_) else np.nan,
            'Fe_pct': round(fe, 2) if not np.isnan(fe) else np.nan,
            'Mn_ppm': round(mn, 1) if not np.isnan(mn) else np.nan,
        })

df = pd.DataFrame(records)

COLOR_MAP = {'Foré': '#2196F3', 'Stoppé': '#F44336', 'En cours': '#4CAF50'}

# ── App ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Auger - Forage Minier", layout="wide")
st.title("⛏️ Dashboard Auger — Campagne de Forage Géochimique")
st.markdown("> **Projet :** Exploration minière en Afrique de l'Ouest | 10 lignes × 7 trous de forage")

# ── Métriques ────────────────────────────────────────────────────────────────
st.subheader("📊 Résumé de la campagne")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total trous", len(df))
c2.metric("Forés ✅", int((df['statut']=='Foré').sum()))
c3.metric("Stoppés 🔴", int((df['statut']=='Stoppé').sum()))
c4.metric("En cours 🟢", int((df['statut']=='En cours').sum()))
c5.metric("Lignes", N_LIGNES)

# ── Tableau des données ───────────────────────────────────────────────────────
with st.expander("📋 Voir le tableau complet des trous"):
    st.dataframe(df.style.applymap(
        lambda v: f'background-color: {"#ffcccc" if v=="Stoppé" else "#ccffcc" if v=="En cours" else "#cce5ff" if v=="Foré" else ""}',
        subset=['statut']
    ), use_container_width=True)

# ── CARTE AUGER ───────────────────────────────────────────────────────────────
st.subheader("🗺️ Carte Auger — Plan de forage")

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#F5F5F0')
ax.set_facecolor('#EAE6DA')

# Grille de fond
for i in range(N_LIGNES):
    lons = df[df['ligne']==f'L{i+1:02d}']['longitude'].values
    lats = df[df['ligne']==f'L{i+1:02d}']['latitude'].values
    ax.plot(lons, lats, color='#AAAAAA', linewidth=0.8, linestyle='--', zorder=1)

# Points avec couleurs selon statut
for statut, color in COLOR_MAP.items():
    sub = df[df['statut']==statut]
    marker = 'o' if statut == 'Foré' else 'X' if statut == 'Stoppé' else 's'
    ax.scatter(sub['longitude'], sub['latitude'],
               c=color, s=120, marker=marker,
               edgecolors='black', linewidths=0.6,
               zorder=3, label=statut)

# Labels des trous
for _, row in df.iterrows():
    ax.annotate(row['trou'], (row['longitude'], row['latitude']),
                textcoords="offset points", xytext=(4, 4),
                fontsize=5.5, color='#333333', zorder=4)

# Labels des lignes
for i in range(N_LIGNES):
    sub = df[df['ligne']==f'L{i+1:02d}']
    ax.text(sub['longitude'].min() - 0.0003, sub['latitude'].mean(),
            f'L{i+1:02d}', fontsize=8, fontweight='bold',
            color='#1A237E', va='center', ha='right')

# ── Flèche Nord ──────────────────────────────────────────────────────────────
ax_lon_max = df['longitude'].max()
ax_lat_max = df['latitude'].max()
nord_x = ax_lon_max + 0.0005
nord_y = ax_lat_max - 0.002
ax.annotate('', xy=(nord_x, nord_y + 0.0015), xytext=(nord_x, nord_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(nord_x, nord_y + 0.0017, 'N', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='black')

# ── Échelle ───────────────────────────────────────────────────────────────────
scale_lon = df['longitude'].min()
scale_lat = df['latitude'].min() - 0.0005
scale_len = 0.003  # ~330m
ax.plot([scale_lon, scale_lon + scale_len], [scale_lat, scale_lat],
        color='black', linewidth=2.5)
ax.plot([scale_lon, scale_lon], [scale_lat-0.0001, scale_lat+0.0001], color='black', lw=2)
ax.plot([scale_lon+scale_len, scale_lon+scale_len], [scale_lat-0.0001, scale_lat+0.0001], color='black', lw=2)
ax.text(scale_lon + scale_len/2, scale_lat - 0.0003, '~330 m',
        ha='center', fontsize=8, fontweight='bold')

# ── Légende ───────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor='#2196F3', edgecolor='black', label='Trou foré'),
    mpatches.Patch(facecolor='#F44336', edgecolor='black', label='Trou stoppé'),
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Trou en cours'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          title='Statut des trous', title_fontsize=9,
          framealpha=0.9, edgecolor='black')

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Carte Auger — Plan de forage géochimique\nAfrique de l'Ouest", fontsize=13, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.4, color='gray')
plt.tight_layout()
st.pyplot(fig)

st.info("""
🗺️ **Lecture de la carte :**
- Chaque ligne (L01 à L10) contient 7 trous de forage espacés régulièrement (~165 m)
- Les lignes sont espacées d'environ 220 m
- 🔵 **Trous forés** : forage complété avec échantillons géochimiques
- 🔴 **Trous stoppés** : forage interrompu (obstacle, refus de terrain)
- 🟢 **Trous en cours** : forage en progression
""")

# ── CARTE ANOMALIE AU ─────────────────────────────────────────────────────────
st.subheader("🌡️ Carte d'anomalie — Or (Au ppb)")

df_fore = df[df['statut']=='Foré'].copy()

fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor('#F5F5F0')

# Interpolation simple pour la carte de chaleur
from scipy.interpolate import griddata

xi = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)
yi = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
Xi, Yi = np.meshgrid(xi, yi)

if len(df_fore) > 3:
    Zi = griddata(
        (df_fore['longitude'], df_fore['latitude']),
        df_fore['Au_ppb'],
        (Xi, Yi), method='linear'
    )
    contour = ax2.contourf(Xi, Yi, Zi, levels=15, cmap='YlOrRd', alpha=0.75)
    plt.colorbar(contour, ax=ax2, label='Au (ppb)')

# Points
for statut, color in COLOR_MAP.items():
    sub = df[df['statut']==statut]
    marker = 'o' if statut == 'Foré' else 'X' if statut == 'Stoppé' else 's'
    ax2.scatter(sub['longitude'], sub['latitude'],
                c=color, s=100, marker=marker,
                edgecolors='black', linewidths=0.6, zorder=3)

# Nord
ax2.annotate('', xy=(df['longitude'].max()+0.0005, df['latitude'].max()-0.001),
             xytext=(df['longitude'].max()+0.0005, df['latitude'].max()-0.0025),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax2.text(df['longitude'].max()+0.0005, df['latitude'].max()-0.0008,
         'N', ha='center', fontsize=12, fontweight='bold')

# Échelle
ax2.plot([df['longitude'].min(), df['longitude'].min()+0.003],
         [df['latitude'].min()-0.0005]*2, color='black', lw=2.5)
ax2.text(df['longitude'].min()+0.0015, df['latitude'].min()-0.0008,
         '~330 m', ha='center', fontsize=8, fontweight='bold')

legend_elements2 = [
    mpatches.Patch(facecolor='#2196F3', edgecolor='black', label='Trou foré'),
    mpatches.Patch(facecolor='#F44336', edgecolor='black', label='Trou stoppé'),
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Trou en cours'),
]
ax2.legend(handles=legend_elements2, loc='lower right', fontsize=9,
           title='Statut', framealpha=0.9, edgecolor='black')

ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_title("Carte d'anomalie géochimique — Or (Au ppb)", fontsize=13, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
st.pyplot(fig2)

st.warning("""
🌡️ **Interprétation de la carte d'anomalie Au :**
- Les zones en **rouge/orange** indiquent de fortes teneurs en or (anomalies positives)
- Les zones en **jaune clair** indiquent des teneurs faibles ou de fond géochimique
- **Recommandation :** Concentrer les prochains forages sur les zones à forte anomalie
""")

# ── Graphiques géochimiques ───────────────────────────────────────────────────
st.subheader("📈 Profils géochimiques par ligne")

element = st.selectbox("Choisir l'élément à visualiser :",
                       ['Au_ppb', 'Cu_ppm', 'As_ppm', 'Fe_pct', 'Mn_ppm'])

fig3, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)
axes = axes.flatten()

for i, ligne in enumerate([f'L{j+1:02d}' for j in range(N_LIGNES)]):
    sub = df[(df['ligne']==ligne) & (df['statut']=='Foré')].sort_values('longitude')
    if len(sub) > 0:
        axes[i].bar(range(len(sub)), sub[element], color='#2E86AB', edgecolor='black', linewidth=0.5)
        axes[i].set_title(ligne, fontsize=9, fontweight='bold')
        axes[i].set_xlabel("Trou", fontsize=7)
        axes[i].set_ylabel(element, fontsize=7)
        axes[i].tick_params(labelsize=6)
    else:
        axes[i].text(0.5, 0.5, 'Aucune donnée', ha='center', va='center', fontsize=8)
        axes[i].set_title(ligne, fontsize=9)

plt.suptitle(f"Profils géochimiques — {element} par ligne de forage", fontsize=12, fontweight='bold')
plt.tight_layout()
st.pyplot(fig3)

# ── Statistiques ─────────────────────────────────────────────────────────────
st.subheader("📊 Statistiques géochimiques")
cols_geo = ['Au_ppb', 'Cu_ppm', 'As_ppm', 'Fe_pct', 'Mn_ppm']
st.dataframe(df[cols_geo].describe().round(2), use_container_width=True)

# ── Recommandations ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Synthèse et Recommandations")
st.markdown(f"""
### 🔬 État de la campagne Auger

| Indicateur | Valeur |
|-----------|--------|
| Total trous planifiés | {len(df)} |
| Trous forés | {int((df['statut']=='Foré').sum())} ({int((df['statut']=='Foré').sum())/len(df)*100:.0f}%) |
| Trous stoppés | {int((df['statut']=='Stoppé').sum())} ({int((df['statut']=='Stoppé').sum())/len(df)*100:.0f}%) |
| Taux d'avancement | {int((df['statut']!='En cours').sum())/len(df)*100:.0f}% |
| Au max détecté | {df['Au_ppb'].max():.1f} ppb |
| Au moyen | {df['Au_ppb'].mean():.1f} ppb |

### 🏗️ Recommandations opérationnelles

1. **Zones prioritaires :** Concentrer les prochains forages sur les lignes à forte anomalie Au
2. **Trous stoppés :** Investiguer les causes (latérite dure, roche mère) et planifier des re-forages
3. **Densification :** Réduire l'espacement à 100m dans les zones anomaliques confirmées
4. **Analyses complémentaires :** Réaliser des analyses multi-éléments (Au, Cu, As, Fe, Mn) sur tous les trous
5. **Extension :** Prévoir des lignes supplémentaires au nord et sud des anomalies détectées

### ⚠️ Observations terrain

- Les trous stoppés peuvent indiquer la présence de **cuirasse latéritique** à faible profondeur
- Les anomalies en **As et Au** sont souvent spatialement corrélées dans les contextes aurifères d'Afrique de l'Ouest
- Privilégier le forage en **saison sèche** pour optimiser la progression
""")

st.caption("Dashboard Auger — Exploration Géochimique | Afrique de l'Ouest")
