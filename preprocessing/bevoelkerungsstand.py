from utils.preprocessing_utils import clean_and_sort, preprocess
import pandas as pd

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024b). Bevölkerungsstand nach Altersjahren (90) - Gemeinden - Stichtag. Landesdatenbank NRW, abgerufen am
28.01.2026. URL:https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=12411-09i 
 """

df = pd.read_csv("data/raw/2024/12411-09i.csv", sep=";", encoding="latin1", skiprows=6)


jahre_6_bis_21 = df.columns[10:25] #unter 7 bis unter 21
jahre_u6 = df.columns[5:10]

df = df[['Unnamed: 2', 'Insgesamt', *jahre_6_bis_21, *jahre_u6]]
df = df.rename(columns={"Unnamed: 2": "Name", "Insgesamt": "Gesamtbevölkerung"})

type_dict = {"Gesamtbevölkerung": "int",
    **{jahr: "int" for jahr in jahre_6_bis_21},
    **{jahr: "int" for jahr in jahre_u6},
}

df = preprocess(df, type_dict)


df["Bevölkerung u6"] = df[jahre_u6].sum(axis=1)
df = df.drop(columns =jahre_u6)

df["Bevölkerung 6 bis 21"] = df[jahre_6_bis_21].sum(axis=1)

jahre_19_bis_21 = df.columns[22:25]

df["Bevölkerung 6 bis 18"] = (
    df["Bevölkerung 6 bis 21"]
    - df[jahre_19_bis_21].sum(axis=1)
)

df = df.drop(columns =jahre_6_bis_21)


df["Anteil 6 bis 21jähriger"] = (
    df["Bevölkerung 6 bis 21"] / df["Gesamtbevölkerung"] *100
)

df = clean_and_sort(df, "Gesamtbevölkerung", "Bevölkerung u6", "Bevölkerung 6 bis 21","Bevölkerung 6 bis 18", "Anteil 6 bis 21jähriger")

# speichern
df.to_csv("data/processed/bevoelkerung_2024.csv", index=False)