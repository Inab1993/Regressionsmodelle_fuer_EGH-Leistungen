import geopandas as gpd
import pandas as pd
import re, unicodedata

def find_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def summarize(s):
    desc = s.describe()
    iqr = desc["75%"] - desc["25%"]
    span = desc["max"] - desc["min"]
    cv = desc["std"] / desc["mean"]

    summary = {
        "mean": desc["mean"],
        "median": desc["50%"],
        "std": desc["std"],
        "max": desc["max"],
        "min": desc["min"],
        "IQR": iqr,
        "span": span,
        "IQR/span": iqr / span,
        "CV": cv
    }

    return pd.Series(summary).round(3)


def top_bottom(df, spalte, n=3, kreis_spalte="Name"):
    # Sortieren nach der gewünschten Spalte
    sortiert = df.sort_values(by=spalte, ascending=True)

    print(f"\nTop {n} (höchste Werte):")
    print(sortiert[[kreis_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))
    print(f"\nBottom {n} (niedrigste Werte):")
    print(sortiert[[kreis_spalte, spalte]].head(n))

def top_bottom_split_double(df, spalte, n=3, name_spalte="Name"):
    # Filter
    df_kreise = df[df["Gebietskörperschaft"] == "Kreis"]
    df_staedte = df[df["Gebietskörperschaft"] == "Kreisfreie Stadt"]

    # Sortieren
    sort_kreise = df_kreise.sort_values(by=spalte, ascending=True)
    sort_staedte = df_staedte.sort_values(by=spalte, ascending=True)

    print("=== Kreise ===")
    print(f"\nTop {n} (höchste+ ++ Werte):")
    print(sort_kreise[[name_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))

    print(f"\nBottom {n} (niedrigste Werte):")
    print(sort_kreise[[name_spalte, spalte]].head(n))

    print("\n\n=== Kreisfreie Städte ===")
    print(f"\nTop {n} (höchste Werte):")
    print(sort_staedte[[name_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))

    print(f"\nBottom {n} (niedrigste Werte):")
    print(sort_staedte[[name_spalte, spalte]].head(n))


def top_bottom_split_triple(df, spalte, n=3, name_spalte="Name"):
    # Filter
    df_gr_kreise = df[df["Kreisstrukturtyp"] == "Großer Kreis"]
    df_kl_kreise = df[df["Kreisstrukturtyp"] == "Kleiner Kreis"]
    df_staedte = df[df["Kreisstrukturtyp"] == "Kreisfreie Stadt"]

    # Sortieren
    sort_gr_kreise = df_gr_kreise.sort_values(by=spalte, ascending=True)
    sort_kl_kreise = df_kl_kreise.sort_values(by=spalte, ascending=True)
    sort_staedte = df_staedte.sort_values(by=spalte, ascending=True)

    print("=== kleine Kreise ===")
    print(f"\nTop {n} (höchste Werte):")
    print(sort_kl_kreise[[name_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))

    print(f"\nBottom {n} (niedrigste Werte):")
    print(sort_kl_kreise[[name_spalte, spalte]].head(n))

    print("=== große Kreise ===")
    print(f"\nTop {n} (höchste Werte):")
    print(sort_gr_kreise[[name_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))

    print(f"\nBottom {n} (niedrigste Werte):")
    print(sort_gr_kreise[[name_spalte, spalte]].head(n))

    print("\n\n=== Kreisfreie Städte ===")
    print(f"\nTop {n} (höchste Werte):")
    print(sort_staedte[[name_spalte, spalte]].tail(n).sort_values(by=spalte, ascending=False))

    print(f"\nBottom {n} (niedrigste Werte):")
    print(sort_staedte[[name_spalte, spalte]].head(n))


def read_nrw_map(df):
    nrw = gpd.read_file("../../data/NRW-map/dvg1krs_nw.shp")

    mapping = {
        "Mülheim a.d. Ruhr": "Mülheim an der Ruhr",
        'Städteregion Aachen': 'Aachen'
    }

    nrw.loc[:, "GN"] = (
        nrw["GN"].replace(mapping)
    )

    nrw = nrw.merge(df, left_on="GN", right_on="Name", how="left")
    return nrw



def validate_df(
    df: pd.DataFrame,
    *,
    not_null: list[str] = None,
    positive: list[str] = None,
    non_negative: list[str] = None,
    numeric: list[str] = None,
    bounds: dict[str, tuple] = None,
    key_cols: list[str] = None,
    df_name: str = "DataFrame"
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
    if len(df) != 53:
        raise ValueError(f"Unerwartete Anzahl Zeilen: {len(df)} (erwartet: 53)")

    if report:
        print("Validierungsbericht")
        for r in report:
            print(" -", r)
    else:
        print(f"{df_name}: alle Checks bestanden")

    return report



