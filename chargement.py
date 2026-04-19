"""
Chargement, validation et prétraitement des données géophysiques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


# Variables utilisées par le modèle ML
FEATURES = [
    'anomalie_magnetique',
    'anomalie_gravimetrique',
    'conductivite_EM',
    'concentration_arsenic',
    'concentration_or'
]

NOMS_COURTS = {
    'anomalie_magnetique': 'Magnétique (nT)',
    'anomalie_gravimetrique': 'Gravimétrique (mGal)',
    'conductivite_EM': 'EM (mS/m)',
    'concentration_arsenic': 'Arsenic (ppm)',
    'concentration_or': 'Or (ppb)'
}


def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Charge et valide les données géophysiques depuis un CSV.

    Parameters
    ----------
    chemin : str
        Chemin vers le fichier CSV

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé et validé
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)

    # Vérification des colonnes requises
    colonnes_requises = FEATURES + ['longitude', 'latitude', 'label']
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")

    # Suppression des lignes avec valeurs manquantes
    avant = len(df)
    df = df.dropna()
    apres = len(df)
    if avant != apres:
        print(f"⚠️  {avant - apres} lignes supprimées (valeurs manquantes)")

    print(f"✅ {len(df)} points chargés depuis {chemin.name}")
    return df


def statistiques(df: pd.DataFrame) -> None:
    """Affiche un résumé statistique des données."""
    nb_gis = (df['label'] == 1).sum()
    nb_ste = (df['label'] == 0).sum()
    nb_inc = (df['label'] == -1).sum()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║         RÉSUMÉ DES DONNÉES GÉOPHYSIQUES          ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Total points          : {len(df):<24} ║")
    print(f"║  Gisements (label=1)   : {nb_gis:<24} ║")
    print(f"║  Stériles  (label=0)   : {nb_ste:<24} ║")
    print(f"║  Inconnus  (label=-1)  : {nb_inc:<24} ║")
    print("╠══════════════════════════════════════════════════╣")

    for feat in FEATURES:
        moy = df[feat].mean()
        mini = df[feat].min()
        maxi = df[feat].max()
        print(f"║  {NOMS_COURTS[feat]:<22}: moy={moy:>8.2f} [{mini:.1f} – {maxi:.1f}] ║")

    print("╚══════════════════════════════════════════════════╝")


def preparer_donnees(df: pd.DataFrame,
                     test_size: float = 0.2,
                     random_state: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, StandardScaler]:
    """
    Prépare les données pour l'entraînement ML.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    from sklearn.model_selection import train_test_split

    df_labeled = df[df['label'].isin([0, 1])].copy()
    X = df_labeled[FEATURES].values
    y = df_labeled['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n📂 Entraînement : {len(X_train)} points")
    print(f"📂 Test         : {len(X_test)} points")

    return X_train, X_test, y_train, y_test, scaler
