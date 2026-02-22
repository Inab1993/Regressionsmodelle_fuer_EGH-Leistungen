from dataclasses import dataclass

import pandas as pd
import re

import unicodedata

from utils.commons import iqr_bounds


def preprocess(
    df,
    parse_cols: dict,
    name_col="Name",
    name_filter=r"Kreis|krfr\. Stadt|Städteregion",
    drop_aachen=True,
):
    """
    parse_cols: dict
        Mapping von Spaltenname -> Zieltyp
        z.B. {
            "Anzahl 35a Hilfen": "int",
            "35a Hilfen pro 10000": "float"
        }
    """

    # komplett leere Zeilen/Spalten entfernen
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # nur relevante Gebietstypen behalten
    df = df[df[name_col].str.contains(name_filter, case=False, na=False)]
    df[name_col] = df[name_col].str.strip()

    # Aachen optional entfernen
    if drop_aachen:
        df = df[~df[name_col].isin([
            "Aachen, Kreis",
            "Aachen, krfr. Stadt",
            "Aachen, krfr. Stadt (ab 21.10.2009)",
        ])]

    # Spalten typisieren
    for col, target_type in parse_cols.items():

        # erst String normalisieren: Komma → Punkt
        series = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        # dann numerisch parsen
        series = pd.to_numeric(series, errors="coerce")

        if target_type == "int":
            df[col] = series.astype("Int64")

        elif target_type == "float":
            df[col] = series.astype("float64")

        else:
            raise ValueError(f"Unbekannter Zieltyp für {col}: {target_type}")


    return df



def clean_and_sort(
        df: pd.DataFrame,
        *value_cols: str,
        source_col: str = "Name",
) -> pd.DataFrame:
    df = df.copy()

    def clean_name(raw: str) -> str:
        raw = unicodedata.normalize("NFKC", str(raw)).replace("\xa0", " ").strip()

        # Suffixe (am Ende)
        raw = re.sub(r"\s*,\s*krfr\.?\s*Stadt\s*$", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*,\s*kreisfreie\s*Stadt\s*$", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*,\s*Stadt\s*$", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*,\s*Kreis\s*$", "", raw, flags=re.IGNORECASE)

        # Präfixe (am Anfang)
        raw = re.sub(r"^(Stadt|Kreis)\s+", "", raw, flags=re.IGNORECASE)

        return raw.strip()

    df["Name"] = df[source_col].apply(clean_name)

    # Städteregion Aachen -> Aachen
    mask = df["Name"].str.contains("aachen", case=False, na=False)
    df.loc[mask, "Name"] = "Aachen"

    # Spaltenauswahl: Name + value_cols oder alles außer Rohspalte
    base_cols = ["Name"]
    if value_cols:
        cols = base_cols + list(value_cols)
    else:
        cols = base_cols + [c for c in df.columns if c not in base_cols + [source_col]]

    cols = [c for c in cols if c in df.columns]
    cleaned_df = df[cols].reset_index(drop=True)

    return cleaned_df

def validate_df(
    df: pd.DataFrame,
    *,
    not_null: list[str] = None,
    positive: list[str] = None,
    non_negative: list[str] = None,
    numeric: list[str] = None,
    bounds: dict[str, tuple] = None,
    key_cols: list[str] = None,
    df_name: str = "DataFrame",
    length: int = 53
):
    report = []

    not_null = not_null or []
    positive = positive or []
    non_negative = non_negative or []
    bounds = bounds or {}
    key_cols = key_cols or []
    numeric = numeric or []

    # Not-null Checks
    for col in not_null:
        n = df[col].isna().sum()
        if n > 0:
            report.append(f"{df_name}: {col} → {n} NaN")

    # > 0 Checks
    for col in positive:
        n = (df[col] <= 0).sum()
        if n > 0:
            report.append(f"{df_name}: {col} → {n} Werte ≤ 0")

    # ≥ 0 Checks
    for col in non_negative:
        n = (df[col] < 0).sum()
        if n > 0:
            report.append(f"{df_name}: {col} → {n} negative Werte")

    # Bounds
    for col, (lo, hi) in bounds.items():
        n = (~df[col].between(lo, hi) & df[col].notna()).sum()
        if n > 0:
            report.append(f"{df_name}: {col} → {n} Werte außerhalb [{lo}, {hi}]")

    # Uniqueness
    if key_cols:
        n = df.duplicated(subset=key_cols).sum()
        if n > 0:
            report.append(f"{df_name}: {n} Duplikate in {key_cols}")

    # Typen-Prüfung
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            n_coerce = pd.to_numeric(df[col], errors="coerce").isna().sum()
            report.append(
                f"{df_name}: {col} → kein numerischer Typ "
                f"(coerce→NaN: {n_coerce})"
            )

    # korrekte Länge
    if len(df) != length:
        raise ValueError(f"Unerwartete Anzahl Zeilen: {len(df)} (erwartet: {length})")

    if report:
        print("Validierungsbericht")
        for r in report:
            print(" -", r)
    else:
        print(f"{df_name}: alle Checks bestanden")

    return report


def winsorize_iqr(x: pd.Series, k: float = 1.5) -> pd.Series:
    lower, upper = iqr_bounds(x, k=k)
    return x.clip(lower=lower, upper=upper)



CLEAN_SUFFIX = "__clean"

@dataclass
class DataView:
    df: pd.DataFrame

    def add_clean_column(self, col: str, cleaner_func) -> None:
        self.df[col + CLEAN_SUFFIX] = cleaner_func(self.df[col])

    def use(self, cleaned: bool, cols: list[str]) -> pd.DataFrame:
        out = self.df.copy()
        for c in cols:
            if cleaned:
                cc = c + CLEAN_SUFFIX
                if cc not in out.columns:
                    raise KeyError(f"Clean-Spalte fehlt: {cc}. Erst add_clean_column() ausführen.")
                out[c] = out[cc]
        return out