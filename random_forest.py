"""
Modèle Random Forest pour le ciblage de gisements miniers.
Entraînement, évaluation et prédiction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Tuple, Optional

from modele.chargement import FEATURES


class ModeleGisement:
    """
    Encapsule le pipeline ML complet pour le ciblage de gisements.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'   # compense le déséquilibre gisements/stériles
        )
        self.entraine = False
        self.importances = None

    def entrainer(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entraîne le modèle Random Forest."""
        print("\n🌲 Entraînement du Random Forest...")
        self.rf.fit(X_train, y_train)
        self.entraine = True
        self.importances = pd.Series(
            self.rf.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)
        print("✅ Modèle entraîné !")
        self._afficher_importances()

    def evaluer(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Évalue les performances sur les données de test.

        Returns
        -------
        dict avec métriques : auc, rapport, matrice_confusion
        """
        if not self.entraine:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")

        y_pred  = self.rf.predict(X_test)
        y_proba = self.rf.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)

        print("\n" + "=" * 50)
        print("  ÉVALUATION DU MODÈLE")
        print("=" * 50)
        print(classification_report(
            y_test, y_pred,
            target_names=['Stérile (0)', 'Gisement (1)']
        ))
        print(f"🏆 AUC-ROC : {auc:.4f}  "
              f"{'(Excellent !)' if auc > 0.85 else '(À améliorer)'}")

        return {
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'rapport': classification_report(y_test, y_pred, output_dict=True),
            'matrice_confusion': confusion_matrix(y_test, y_pred),
            'fpr': roc_curve(y_test, y_proba)[0],
            'tpr': roc_curve(y_test, y_proba)[1]
        }

    def predire(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit les labels et probabilités pour de nouveaux points.

        Returns
        -------
        (labels_predits, probabilites_gisement)
        """
        if not self.entraine:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        labels = self.rf.predict(X)
        probas = self.rf.predict_proba(X)[:, 1]
        return labels, probas

    def validation_croisee(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
        """Évalue le modèle par validation croisée (k-fold)."""
        scores = cross_val_score(self.rf, X, y, cv=cv, scoring='roc_auc')
        print(f"\n📊 Validation croisée ({cv}-fold) :")
        print(f"   AUC moyen  : {scores.mean():.4f}")
        print(f"   Écart-type : {scores.std():.4f}")
        print(f"   Min / Max  : {scores.min():.4f} / {scores.max():.4f}")

    def sauvegarder(self, chemin: str) -> None:
        """Sauvegarde le modèle entraîné sur disque."""
        joblib.dump(self.rf, chemin)
        print(f"💾 Modèle sauvegardé → {chemin}")

    def charger(self, chemin: str) -> None:
        """Charge un modèle sauvegardé."""
        self.rf = joblib.load(chemin)
        self.entraine = True
        print(f"📂 Modèle chargé depuis {chemin}")

    def _afficher_importances(self) -> None:
        print("\n⭐ Importance des variables :")
        for feat, imp in self.importances.items():
            barre = '█' * int(imp * 40)
            print(f"   {feat:<30} {barre} {imp:.3f}")
