import pandas as pd
import re
from helper.preprocessing import clean_and_sort, preprocess
from helper.functions import validate_df

df = pd.read_csv("data/raw/2024/22517-02i.csv", sep=";", encoding="latin1", skiprows=6)

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024). Kinder- und Jugendhilfe:
Hilfen nach § 35a SGB VIII. Landesdatenbank NRW, abgerufen am 28.01.2026. URL:
https://www.landesdatenbank.nrw.de/ldbnrw/online?operation=table&code=22517-02i.
"""

df = df.rename(columns={"Unnamed: 2": "Name", "Eingliederungshilfe für seelisch behinderte junge Menschen § 35a SGB VIII": "Anzahl 35a Hilfen"})

grosse_kommunen = ("Arnsberg, Bergheim, Bergisch Gladbach, Bocholt, Castrop-Rauxel, "
                   "Detmold, Dinslaken, Dormagen, Dorsten, Düren, Gladbeck, Grevenbroich, "
                   "Gütersloh, Herford, Herten, Iserlohn, Kerpen, Lippstadt, Lüdenscheid, "
                   "Lünen, Marl, Minden, Moers, Neuss, Paderborn, Ratingen, Recklinghausen, "
                   "Rheine, Siegen, Troisdorf, Unna, Velbert, Viersen, Wesel, Witten")

kommunen = {k.strip().lower() for k in grosse_kommunen.split(",")}
pattern = "|".join(map(re.escape, kommunen))
name_l = df["Name"].astype(str).str.lower()
mask_special = name_l.str.contains("jugendamt", na=False) & name_l.str.contains(pattern, na=False)
df.loc[mask_special, "kreis_code_tmp"] = df.loc[mask_special, "Unnamed: 1"].astype(str).str[2:7]
large_codes = set(df.loc[mask_special, "kreis_code_tmp"].dropna())

df = preprocess(df, {"Insgesamt" :  "float", "Anzahl 35a Hilfen": "float"})

df = df[~df["Name"].str.contains("Kreisjugendamt", case=False, na=False)]


# Flag nur bei exakt passender Kreis-Zeile setzen
df["Kreis mit großer Gemeinde"] = (
    df["Unnamed: 1"].astype(str).isin(large_codes)
)


# Typ (Kreis/kreisfreie Stadt/großer Kreis) extrahieren
kreisfreie_keywords = ["krfr. Stadt", "kreisfreie Stadt", "Stadt"]
kreis_keywords = ["Kreis", "kreis"]

def extract(raw: str, grosser_kreis: bool):
    raw = str(raw).strip()
    grosser_kreis = bool(grosser_kreis)

    if any(k in raw for k in kreisfreie_keywords):
        return raw, "Kreisfreie Stadt"

    if any(k in raw for k in kreis_keywords):
        return raw, "Großer Kreis" if grosser_kreis else "Kleiner Kreis"

    return raw, "Unbekannt"

df[["Name", "Typ 1"]] = df.apply(
    lambda r: pd.Series(extract(r["Name"], r["Kreis mit großer Gemeinde"])),
    axis=1
)

df["Typ 2"] = df["Typ 1"].replace(
    {"Großer Kreis": "Kreis", "Kleiner Kreis": "Kreis"})

df = clean_and_sort(df, "Typ 1", "Typ 2", "Insgesamt", "Anzahl 35a Hilfen")

df.loc[df["Name"] == "Aachen", "Typ 1"] ="Städteregion"
df.loc[df["Name"] == "Aachen", "Typ 2"] = "Städteregion"
df.loc[df["Name"] == "Essen", "Anzahl 35a Hilfen"] = 0 #dummy

validate_df(
    df,
    not_null=["Insgesamt", "Anzahl 35a Hilfen"],
    non_negative=["Insgesamt", "Anzahl 35a Hilfen"],
    numeric=["Insgesamt", "Anzahl 35a Hilfen"],
    key_cols=["Name"],
)

# saubere Tabelle abspeichern
df.to_csv("data/processed/hilfen_2024.csv", index=False)