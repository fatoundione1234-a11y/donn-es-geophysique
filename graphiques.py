"""
Visualisation des données géophysiques et des résultats ML.
Génère des graphiques de haute qualité pour l'analyse et les rapports.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional

from modele.chargement import FEATURES, NOMS_COURTS

# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#444466',
    'text.color':       'white',
    'axes.labelcolor':  '#ccccdd',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'grid.color':       '#2a2a4a',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans'
})

COULEUR_GISEMENT = '#FF6B6B'
COULEUR_STERILE  = '#4ECDC4'
COULEUR_ACCENT   = '#FFE66D'


def sauvegarder(fig: plt.Figure, nom: str, dossier: str = 'outputs') -> None:
    Path(dossier).mkdir(exist_ok=True)
    chemin = Path(dossier) / nom
    fig.savefig(chemin, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  ✅ Sauvegardé → {chemin}")
    plt.close(fig)


# ── 1. Exploration des données ────────────────────────────────────────────────

def graphique_exploration(df: pd.DataFrame, dossier: str = 'outputs') -> None:
    """6 graphiques d'exploration des données géophysiques."""
    print("\n🎨 Graphique 1 : Exploration des données...")

    gis = df[df['label'] == 1]
    ste = df[df['label'] == 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Exploration des Données Géophysiques\n"
                 "Rouge★ = Gisements  |  Cyan● = Stériles",
                 fontsize=15, fontweight='bold', color='white', y=1.01)

    kw_g = dict(c=COULEUR_GISEMENT, s=130, alpha=0.9, marker='*', zorder=3, label='Gisements')
    kw_s = dict(c=COULEUR_STERILE,  s=80,  alpha=0.8, zorder=2,   label='Stériles')

    # --- Carte de position
    ax = axes[0, 0]
    ax.scatter(ste['longitude'], ste['latitude'], **kw_s)
    ax.scatter(gis['longitude'], gis['latitude'], **kw_g)
    ax.set_title('📍 Carte de Localisation', fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(facecolor='#222244', labelcolor='white')
    ax.grid(True)

    # --- Magnétique vs EM
    ax = axes[0, 1]
    ax.scatter(ste['anomalie_magnetique'], ste['conductivite_EM'], **kw_s)
    ax.scatter(gis['anomalie_magnetique'], gis['conductivite_EM'], **kw_g)
    ax.set_title('🧲 Magnétique vs EM', fontweight='bold')
    ax.set_xlabel('Anomalie magnétique (nT)'); ax.set_ylabel('Conductivité EM (mS/m)')
    ax.legend(facecolor='#222244', labelcolor='white'); ax.grid(True)

    # --- Arsenic vs Or
    ax = axes[0, 2]
    ax.scatter(ste['concentration_arsenic'], ste['concentration_or'], **kw_s)
    ax.scatter(gis['concentration_arsenic'], gis['concentration_or'], **kw_g)
    ax.set_title('⚗️  Géochimie : As vs Au', fontweight='bold')
    ax.set_xlabel('Arsenic (ppm)'); ax.set_ylabel('Or (ppb)')
    ax.legend(facecolor='#222244', labelcolor='white'); ax.grid(True)

    # --- Magnétique vs Gravimétrique
    ax = axes[1, 0]
    ax.scatter(ste['anomalie_magnetique'], ste['anomalie_gravimetrique'], **kw_s)
    ax.scatter(gis['anomalie_magnetique'], gis['anomalie_gravimetrique'], **kw_g)
    ax.set_title('🌍 Magnétique vs Gravimétrique', fontweight='bold')
    ax.set_xlabel('Anomalie magnétique (nT)'); ax.set_ylabel('Anomalie gravimétrique (mGal)')
    ax.legend(facecolor='#222244', labelcolor='white'); ax.grid(True)

    # --- Histogramme magnétique
    ax = axes[1, 1]
    ax.hist(ste['anomalie_magnetique'], bins=8, color=COULEUR_STERILE,
            alpha=0.75, label='Stériles', edgecolor='white', linewidth=0.5)
    ax.hist(gis['anomalie_magnetique'], bins=5, color=COULEUR_GISEMENT,
            alpha=0.75, label='Gisements', edgecolor='white', linewidth=0.5)
    ax.set_title('📊 Distribution — Anomalie Magnétique', fontweight='bold')
    ax.set_xlabel('Anomalie magnétique (nT)'); ax.set_ylabel('Fréquence')
    ax.legend(facecolor='#222244', labelcolor='white'); ax.grid(True)

    # --- Matrice de corrélation
    ax = axes[1, 2]
    corr = df[FEATURES].corr()
    corr.index   = ['Mag', 'Grav', 'EM', 'As', 'Au']
    corr.columns = ['Mag', 'Grav', 'EM', 'As', 'Au']
    sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, linecolor='#1a1a2e',
                annot_kws={'size': 9, 'color': 'white'})
    ax.set_title('🔗 Matrice de Corrélation', fontweight='bold')

    plt.tight_layout()
    sauvegarder(fig, '01_exploration_donnees.png', dossier)


# ── 2. Résultats du Random Forest ────────────────────────────────────────────

def graphique_ml(resultats: dict, importances: pd.Series,
                 dossier: str = 'outputs') -> None:
    """Matrice de confusion, importance des variables, courbe ROC."""
    print("\n🎨 Graphique 2 : Résultats ML...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Résultats du Random Forest — Ciblage des Gisements",
                 fontsize=14, fontweight='bold', color='white')

    # --- Matrice de confusion
    ax = axes[0]
    cm = resultats['matrice_confusion']
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Stérile', 'Gisement'],
                yticklabels=['Stérile', 'Gisement'],
                ax=ax, linewidths=2, linecolor='#1a1a2e',
                annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'})
    ax.set_title('🎯 Matrice de Confusion', fontweight='bold')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')

    # --- Importance des variables
    ax = axes[1]
    imp_sorted = importances.sort_values()
    colors = [COULEUR_GISEMENT if i == len(imp_sorted)-1 else COULEUR_STERILE
              for i in range(len(imp_sorted))]
    bars = ax.barh(range(len(imp_sorted)), imp_sorted.values,
                   color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(['Mag.', 'Grav.', 'EM', 'Arsenic', 'Or'], fontsize=10)
    ax.set_title('⭐ Importance des Variables', fontweight='bold')
    ax.set_xlabel('Score d\'importance')
    for bar, val in zip(bars, imp_sorted.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='white')
    ax.grid(True, axis='x')

    # --- Courbe ROC
    ax = axes[2]
    fpr, tpr, auc = resultats['fpr'], resultats['tpr'], resultats['auc']
    ax.plot(fpr, tpr, color=COULEUR_GISEMENT, lw=2.5,
            label=f'Random Forest (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Aléatoire')
    ax.fill_between(fpr, tpr, alpha=0.15, color=COULEUR_GISEMENT)
    ax.set_title('📈 Courbe ROC', fontweight='bold')
    ax.set_xlabel('Taux de Faux Positifs')
    ax.set_ylabel('Taux de Vrais Positifs')
    ax.legend(facecolor='#222244', labelcolor='white')
    ax.grid(True)

    plt.tight_layout()
    sauvegarder(fig, '02_resultats_random_forest.png', dossier)


# ── 3. Carte de potentiel minéral ─────────────────────────────────────────────

def carte_potentiel(df: pd.DataFrame, dossier: str = 'outputs') -> None:
    """Carte de chaleur du potentiel minéral avec probabilités."""
    print("\n🎨 Graphique 3 : Carte de potentiel minéral...")

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d0d1e')

    sc = ax.scatter(
        df['longitude'], df['latitude'],
        c=df['probabilite_gisement'],
        cmap='RdYlGn', s=250,
        edgecolors='white', linewidth=0.5,
        alpha=0.95, vmin=0, vmax=1, zorder=2
    )

    # Entourer les vrais gisements
    gis = df[df['label'] == 1]
    ax.scatter(gis['longitude'], gis['latitude'],
               c='none', edgecolors='white', linewidth=2.5,
               s=380, marker='*', zorder=3, label='Gisements confirmés')

    # Étiquettes de probabilité
    for _, row in df.iterrows():
        prob = row['probabilite_gisement']
        if prob > 0.5:
            ax.annotate(f"{prob:.0%}",
                        (row['longitude'], row['latitude']),
                        textcoords='offset points', xytext=(8, 5),
                        fontsize=8, color='white', alpha=0.85)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Probabilité de gisement', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title('🗺️  Carte de Potentiel Minéral\n'
                 'Vert = forte probabilité · Rouge = faible probabilité',
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.legend(facecolor='#222244', labelcolor='white', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    sauvegarder(fig, '03_carte_potentiel_mineral.png', dossier)


# ── 4. Rapport de synthèse ───────────────────────────────────────────────────

def graphique_synthese(df: pd.DataFrame, auc: float,
                       dossier: str = 'outputs') -> None:
    """Graphique de synthèse avec statistiques clés."""
    print("\n🎨 Graphique 4 : Synthèse...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Synthèse — Comparaison Gisements vs Stériles",
                 fontsize=14, fontweight='bold', color='white')

    gis = df[df['label'] == 1]
    ste = df[df['label'] == 0]

    # --- Comparaison des moyennes normalisées
    ax = axes[0]
    noms = ['Mag.', 'Grav.', 'EM', 'Arsenic', 'Or']
    moy_g = [abs(gis[f].mean()) for f in FEATURES]
    moy_s = [abs(ste[f].mean()) for f in FEATURES]
    total = [g + s for g, s in zip(moy_g, moy_s)]
    norm_g = [g/t*100 for g, t in zip(moy_g, total)]
    norm_s = [s/t*100 for s, t in zip(moy_s, total)]

    x = np.arange(len(noms))
    width = 0.35
    ax.bar(x - width/2, norm_g, width, label='Gisements',
           color=COULEUR_GISEMENT, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, norm_s, width, label='Stériles',
           color=COULEUR_STERILE, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(noms)
    ax.set_ylabel('Contribution relative (%)')
    ax.set_title('Moyennes normalisées par classe', fontweight='bold')
    ax.legend(facecolor='#222244', labelcolor='white')
    ax.grid(True, axis='y')

    # --- Boxplots anomalie magnétique
    ax = axes[1]
    data_box = [gis['anomalie_magnetique'].values, ste['anomalie_magnetique'].values]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(COULEUR_GISEMENT + '99')
    bp['boxes'][1].set_facecolor(COULEUR_STERILE  + '99')
    bp['boxes'][0].set_edgecolor(COULEUR_GISEMENT)
    bp['boxes'][1].set_edgecolor(COULEUR_STERILE)
    for w in bp['whiskers'] + bp['caps'] + bp['fliers']:
        w.set_color('#888899')
    ax.set_xticklabels(['Gisements', 'Stériles'])
    ax.set_ylabel('Anomalie magnétique (nT)')
    ax.set_title('Distribution de l\'anomalie magnétique', fontweight='bold')
    ax.grid(True, axis='y')

    # AUC en annotation
    fig.text(0.5, -0.03,
             f"Modèle Random Forest  |  AUC-ROC = {auc:.3f}  |  "
             f"25 points  |  {len(gis)} gisements / {len(ste)} stériles",
             ha='center', fontsize=10, color='#aaaacc',
             style='italic')

    plt.tight_layout()
    sauvegarder(fig, '04_synthese.png', dossier)
