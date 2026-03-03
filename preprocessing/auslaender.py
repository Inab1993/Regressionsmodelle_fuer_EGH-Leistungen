
from utils.preprocessing_utils import clean_and_sort, preprocess
import pandas as pd

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024a). Ausländische Bevölkerung nach Geschlecht und
Staatsangehörigkeiten (215) - kreisfreie Städte und Kreise - Stichtag. Landesdatenbank NRW, abgerufen am
14.02.2026. URL:https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=12521-03i"""

df = pd.read_csv("data/raw/2024/12521-03i.csv", sep=";", encoding="latin1", skiprows=6)

df = df.rename(columns={"Unnamed: 2": "Name", "Insgesamt": "Anzahl Ausländer"})

type_dict = {"Unnamed: 3": "int",
             "Anzahl Ausländer": "int",
             }

df = preprocess(df, type_dict)

df = df[df["Unnamed: 3"].isin([1, 2, 3, 4, 5, 9])]
df = (
    df.groupby("Name", as_index=False)["Anzahl Ausländer"]
      .sum()
)

df = clean_and_sort(df, "Anzahl Ausländer")

df.to_csv("data/processed/auslaender_2024.csv", index=False)