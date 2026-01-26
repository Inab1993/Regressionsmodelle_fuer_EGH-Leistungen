import pandas as pd
import re

import unicodedata


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
