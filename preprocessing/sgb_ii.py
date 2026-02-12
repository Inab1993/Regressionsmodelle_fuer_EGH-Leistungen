import pandas as pd
from utils.preprocessing_utils import preprocess, clean_and_sort

# CSV-Datei einlesen
"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024e). Sozialberichterstattung
in der amtlichen Statistik: Empfänger/innen von sozialen Mindestsicherungsleistungen nach
Art der Leistung - Gemeinden (Wohnortprinzip) - Jahr. Landesdatenbank NRW, abgeru-
fen am 28.01.2026. URL: https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=22811-05i"""
df = pd.read_csv("data/raw/2024/22811-05i.csv", sep=";", encoding="latin1", skiprows=4)


df = df[['Unnamed: 2','Unnamed: 3']]
df = df.rename(columns={"Unnamed: 2": "Name", "Unnamed: 3": "SGB II-Bezug"})

df = preprocess(df, {"SGB II-Bezug": "int"})
df = clean_and_sort(df, "SGB II-Bezug")

# speichern
df.to_csv("data/processed/sgb2_2024.csv", index=False)