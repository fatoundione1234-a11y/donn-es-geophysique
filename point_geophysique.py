"""
Modèle de données géophysiques.
Définit la structure d'un point de mesure sur le terrain.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PointGeophysique:
    """
    Représente un point de mesure géophysique sur le terrain.

    Attributs
    ---------
    longitude : float
        Coordonnée est-ouest (degrés décimaux)
    latitude : float
        Coordonnée nord-sud (degrés décimaux)
    anomalie_magnetique : float
        Anomalie du champ magnétique en nanoTesla (nT)
    anomalie_gravimetrique : float
        Anomalie de Bouguer en milliGal (mGal)
    conductivite_EM : float
        Conductivité électromagnétique en mS/m
    concentration_arsenic : float
        Teneur en arsenic (pathfinder de l'or) en ppm
    concentration_or : float
        Teneur en or en ppb
    label : int
        1 = gisement confirmé, 0 = zone stérile, -1 = inconnu
    """
    longitude: float
    latitude: float
    anomalie_magnetique: float
    anomalie_gravimetrique: float
    conductivite_EM: float
    concentration_arsenic: float
    concentration_or: float
    label: int = -1

    def est_gisement(self) -> bool:
        return self.label == 1

    def est_sterile(self) -> bool:
        return self.label == 0

    def est_inconnu(self) -> bool:
        return self.label == -1

    def to_dict(self) -> dict:
        return {
            'longitude': self.longitude,
            'latitude': self.latitude,
            'anomalie_magnetique': self.anomalie_magnetique,
            'anomalie_gravimetrique': self.anomalie_gravimetrique,
            'conductivite_EM': self.conductivite_EM,
            'concentration_arsenic': self.concentration_arsenic,
            'concentration_or': self.concentration_or,
            'label': self.label
        }

    def __str__(self):
        statut = {1: 'GISEMENT', 0: 'STERILE', -1: 'INCONNU'}.get(self.label, '?')
        return (f"[{statut}] ({self.longitude:.4f}, {self.latitude:.4f}) | "
                f"Mag={self.anomalie_magnetique:.1f}nT | "
                f"Grav={self.anomalie_gravimetrique:.2f}mGal | "
                f"EM={self.conductivite_EM:.1f}mS/m | "
                f"As={self.concentration_arsenic:.1f}ppm | "
                f"Au={self.concentration_or:.1f}ppb")
