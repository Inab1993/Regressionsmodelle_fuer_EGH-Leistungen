from __future__ import annotations

from pathlib import Path

import pandas as pd

NAME = "Name"
HILFEN_35A = "35a Hilfen pro 10000"
AUSLAENDER = "Ausländeranteil"
ABITUR = "Abiturquote"
SGBII = "SGB II-Quote"
KINDERANTEIL = "Kinderanteil"
KINDERARZTDICHTE = "Kinderarztdichte"
TRAEGER = "Überörtlicher Träger"
KREISSTRUKTUR = "Kreisstrukturtyp"
GEBIETSKOERPERSCHAFT = "Gebietskörperschaft"
ERZ_HILFEN = "erz. Hilfen pro 10000"
BEV_DICHTE = "Bevölkerungsdichte"
KJP = "KJP-Dichte"


BASE_RESULT_PATH = Path("results")


def result_path(theme: str, base: Path = BASE_RESULT_PATH) -> Path:
    path = base /theme
    path.mkdir(parents=True, exist_ok=True)
    return path


def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return float(lower), float(upper)
