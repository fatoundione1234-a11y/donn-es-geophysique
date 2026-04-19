"""
============================================================
  PROJET : Ciblage de Gisements par Machine Learning
  Géophysique Minière — Pipeline complet

  Usage :
      python main.py
      python main.py --data data/donnees_geophysiques.csv
      python main.py --aide
============================================================
"""

import sys
import argparse
from pathlib import Path

# Ajout du dossier src au path Python
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modele.chargement import charger_donnees, statistiques, preparer_donnees, FEATURES
from ml.random_forest import ModeleGisement
from visualisation.graphiques import (
    graphique_exploration,
    graphique_ml,
    carte_potentiel,
    graphique_synthese
)


def afficher_banniere():
    print("""
╔══════════════════════════════════════════════════════╗
║   CIBLAGE DE GISEMENTS PAR MACHINE LEARNING          ║
║   Application de l'IA à la Géophysique Minière       ║
╠══════════════════════════════════════════════════════╣
║   Méthode   : Random Forest                          ║
║   Données   : Magnétique, Gravimétrique, EM, Géochim ║
║   Objectif  : Carte de potentiel minéral             ║
╚══════════════════════════════════════════════════════╝
""")


def pipeline(chemin_data: str = 'data/donnees_geophysiques.csv',
             dossier_outputs: str = 'outputs'):
    """
    Pipeline complet :
    1. Chargement des données
    2. Visualisation exploratoire
    3. Entraînement Random Forest
    4. Évaluation
    5. Carte de potentiel minéral
    6. Rapport de synthèse
    """

    afficher_banniere()

    # ── 1. Chargement ──────────────────────────────────────────
    print("─" * 55)
    print("  ÉTAPE 1/6 — Chargement des données")
    print("─" * 55)
    df = charger_donnees(chemin_data)
    statistiques(df)

    # ── 2. Visualisation exploratoire ──────────────────────────
    print("\n" + "─" * 55)
    print("  ÉTAPE 2/6 — Visualisation exploratoire")
    print("─" * 55)
    graphique_exploration(df, dossier_outputs)

    # ── 3. Préparation ML ──────────────────────────────────────
    print("\n" + "─" * 55)
    print("  ÉTAPE 3/6 — Préparation des données ML")
    print("─" * 55)
    X_train, X_test, y_train, y_test, scaler = preparer_donnees(df)

    # ── 4. Entraînement ────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  ÉTAPE 4/6 — Entraînement du modèle")
    print("─" * 55)
    modele = ModeleGisement(n_estimators=100, max_depth=5)
    modele.entrainer(X_train, y_train)

    # ── 5. Évaluation ──────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  ÉTAPE 5/6 — Évaluation du modèle")
    print("─" * 55)
    resultats = modele.evaluer(X_test, y_test)
    graphique_ml(resultats, modele.importances, dossier_outputs)

    # ── 6. Carte de potentiel ──────────────────────────────────
    print("\n" + "─" * 55)
    print("  ÉTAPE 6/6 — Carte de potentiel minéral")
    print("─" * 55)

    import numpy as np
    df_labeled = df[df['label'].isin([0, 1])].copy()
    X_all = scaler.transform(df_labeled[FEATURES].values)
    labels_pred, probas = modele.predire(X_all)
    df_labeled = df_labeled.copy()
    df_labeled['probabilite_gisement'] = probas
    df_labeled['prediction'] = labels_pred

    carte_potentiel(df_labeled, dossier_outputs)
    graphique_synthese(df_labeled, resultats['auc'], dossier_outputs)

    # Sauvegarde du modèle
    Path(dossier_outputs).mkdir(exist_ok=True)
    modele.sauvegarder(f'{dossier_outputs}/modele_rf.pkl')

    # ── Résumé final ───────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════╗
║   ✅ PIPELINE TERMINÉ — Fichiers générés :           ║
║                                                      ║
║   outputs/01_exploration_donnees.png                 ║
║   outputs/02_resultats_random_forest.png             ║
║   outputs/03_carte_potentiel_mineral.png             ║
║   outputs/04_synthese.png                            ║
║   outputs/modele_rf.pkl                              ║
╠══════════════════════════════════════════════════════╣
║   AUC-ROC final : {resultats['auc']:.4f}                        ║
╚══════════════════════════════════════════════════════╝
""")

    return modele, df_labeled, resultats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline ML pour le ciblage de gisements miniers'
    )
    parser.add_argument(
        '--data', default='data/donnees_geophysiques.csv',
        help='Chemin vers le fichier CSV de données'
    )
    parser.add_argument(
        '--outputs', default='outputs',
        help='Dossier de sortie pour les graphiques'
    )
    args = parser.parse_args()

    pipeline(
        chemin_data=args.data,
        dossier_outputs=args.outputs
    )
