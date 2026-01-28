from helper.preprocessing import clean_and_sort, preprocess
import pandas as pd

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024a). Bevölkerungsstand nach Altersjahren (90) - Gemeinden - Stichtag. Landesdatenbank NRW, abgerufen am
28.01.2026. URL:https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=12411-09"""

df = pd.read_csv("data/raw/2024/12411-09i.csv", sep=";", encoding="latin1", skiprows=6)

df = df[['Unnamed: 2', 'Insgesamt', '6 bis unter 10 Jahre', '10 bis unter 15 Jahre', '15 bis unter 18 Jahre', '18 bis unter 20 Jahre']]
df = df.rename(columns={"Unnamed: 2": "Name", "Insgesamt": "Gesamtbevölkerung"})

type_dict = {"Gesamtbevölkerung": "int",
        '6 bis unter 10 Jahre': "int",
        '10 bis unter 15 Jahre': "int",
        '15 bis unter 18 Jahre': "int",
        '18 bis unter 20 Jahre': "int"
             }

df = preprocess(df, type_dict)


df["Bevölkerung 6 bis 20"] = df[[ '6 bis unter 10 Jahre', '10 bis unter 15 Jahre', '15 bis unter 18 Jahre', '18 bis unter 20 Jahre']].sum(axis=1)
df = df.drop(columns = ['6 bis unter 10 Jahre', '10 bis unter 15 Jahre', '15 bis unter 18 Jahre', '18 bis unter 20 Jahre'])

df["Anteil gesamt"] = (
    df["Bevölkerung 6 bis 20"] / df["Gesamtbevölkerung"] *100
)

df = clean_and_sort(df, "Gesamtbevölkerung", "Bevölkerung 6 bis 20", "Anteil gesamt")

# speichern
df.to_csv("data/processed/bevoelkerung_2024.csv", index=False)