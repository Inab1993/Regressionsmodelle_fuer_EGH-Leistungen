import pandas as pd
from utils.preprocessing_utils import clean_and_sort, preprocess

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024d). Sozialversicherungs-
pflichtige Beschäftigte (Arbeitsort) nach Wirtschaftsabschnitte sowie Schul- und Berufsab-
schluss - kreisfreie Städte und Kreise - Stichtag (ab 2021). Landesdatenbank NRW, abge-
rufen am 28.01.2026. URL: https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=13111-54i"""
df = pd.read_csv("data/raw/2024/13111-54i.csv", sep=";", encoding="latin1", skiprows=6)

df = df[df["Unnamed: 3"] == "Insgesamt"]

df = df.rename(columns={"Unnamed: 1": "Name"})

type_dict = {
    'Insgesamt': "int",
    'mit Abitur/Fachabitur':"int",
}

df = preprocess(df, type_dict)

df['Abiturquote'] = (df['mit Abitur/Fachabitur'].div(df["Insgesamt"], axis=0))*100

df = clean_and_sort(df, 'Abiturquote')

df.to_csv("data/processed/abiturquote_2024.csv", index=False)
