import pandas as pd
from helper.preprocessing import clean_and_sort, preprocess

# Quelle: https://www.landesdatenbank.nrw.de/
df = pd.read_csv("data/raw/2024/13111-54i.csv", sep=";", encoding="latin1", skiprows=6)


df = df[df["Unnamed: 3"] == "Insgesamt"]
df = df.drop(df.columns[[0, 2, 3, -5, -4, -3, -1]], axis=1)

# Spalten umbenennen
df = df.rename(columns={"Unnamed: 1": "Name"})

type_dict = {
    'Insgesamt': "int",
    'ohne Schulabschluss':"int",
    'mit Volks-/Haupschulabschluss':"int",
    'mit mittlerer Reife/gleichwertigem Abschluss':"int",
    'mit Abitur/Fachabitur':"int",
    'mit akademischem Abschluss':"int"
}

df = preprocess(df, type_dict)

df = clean_and_sort(df, 'Insgesamt', 'ohne Schulabschluss',
       'mit Volks-/Haupschulabschluss',
       'mit mittlerer Reife/gleichwertigem Abschluss', 'mit Abitur/Fachabitur',
       'mit akademischem Abschluss')

#Anteil des jew. Abschlusses an Gesamtbevölkerung des Kreises berechnen
#df_bevoelkerung = pd.read_csv("../data/processed/bevoelkerung_2024.csv", sep=",")

#df = df.merge(df_bevoelkerung[["Name", "Gesamtbevölkerung"]], on="Name", how="left", validate="one_to_one")

#print(df.columns)


abschluss_cols = df.columns.drop(["Name", "Insgesamt"])

df[abschluss_cols] = df[abschluss_cols].div(df["Insgesamt"], axis=0)

weights = {
    'ohne Schulabschluss':0,
    'mit Volks-/Haupschulabschluss':1,
    'mit mittlerer Reife/gleichwertigem Abschluss':2,
    'mit Abitur/Fachabitur':3,
    'mit akademischem Abschluss':4
}

df["Bildungsindex"] = sum(
    df[col] * weight
    for col, weight in weights.items()
)


df = df[["Name", "Bildungsindex"]]

df.to_csv("data/processed/bildungsabschluss_2024.csv", index=False)
