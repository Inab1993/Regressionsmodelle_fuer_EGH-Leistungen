import pandas as pd
from utils.preprocessing_utils import clean_and_sort, preprocess

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024f). Sozialversicherungs-
pflichtige Beschäftigte (Arbeitsort) nach Wirtschaftsabschnitte sowie Schul- und Berufsab-
schluss - kreisfreie Städte und Kreise - Stichtag (ab 2021). Landesdatenbank NRW, abge-
rufen am 28.01.2026. URL: https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=13111-54i"""
df = pd.read_csv("data/raw/2024/13111-54i.csv", sep=";", encoding="latin1", skiprows=6)


df = df[df["Unnamed: 3"] == "Insgesamt"]
df = df.drop(df.columns[[0, 2, 3, -5, -4, -3, -2,-1]], axis=1)

# Spalten umbenennen
df = df.rename(columns={"Unnamed: 1": "Name"})

type_dict = {
    'Insgesamt': "int",
    'ohne Schulabschluss':"int",
    'mit Volks-/Haupschulabschluss':"int",
    'mit mittlerer Reife/gleichwertigem Abschluss':"int",
    'mit Abitur/Fachabitur':"int",
}

df = preprocess(df, type_dict)

df = clean_and_sort(df, 'Insgesamt', 'ohne Schulabschluss',
       'mit Volks-/Haupschulabschluss',
       'mit mittlerer Reife/gleichwertigem Abschluss', 'mit Abitur/Fachabitur')




abschluss_cols = df.columns.drop(["Name", "Insgesamt"])

df[abschluss_cols] = df[abschluss_cols].div(df["Insgesamt"], axis=0)


weights = {
    'ohne Schulabschluss':0,
    'mit Volks-/Haupschulabschluss':1,
    'mit mittlerer Reife/gleichwertigem Abschluss':2,
    'mit Abitur/Fachabitur':3
}

df["Bildungsindex"] = sum(
    df[col] * weight
    for col, weight in weights.items()
)

df['Abiturquote'] = df['mit Abitur/Fachabitur']*100

df = clean_and_sort(df, 'Abiturquote', "Bildungsindex")
df.to_csv("data/processed/abiturquote_2024.csv", index=False)
