import pandas as pd
import re

from helper.preprocessing_utils import clean_and_sort

""" Quelle: Bundesamt für Bauwesen und Raumordnung (2024). Referenztabellen zu Raumgliede-
rungen des BBSR. Bundesinstitut für Bau-, Stadt- und Raumforschung (BBSR), ab-
gerufen am 28.01.2026. URL: https://www.bbsr.bund.de/BBSR/DE/forschung/raumbeobachtung/Raumabgrenzungen/downloads/download-referenzen.html"""
df = pd.read_excel("data/raw/2024/raumgliederungen-referenzen-2024.xlsx", skiprows=1, sheet_name="Kreisreferenz")

df=df[["Kreise (2024) Name", "Raumordnungsregionen (2024) Name"]]
df=df.rename(columns={"Kreise (2024) Name": "Name", "Raumordnungsregionen (2024) Name": "ROR"})

df2 = pd.read_csv("data/processed/hilfen_2024.csv")


names = (
    df2["Name"]
    .dropna()
    .astype(str)
    .str.lower()
    .apply(re.escape)  
)

pattern = r"\b(" + "|".join(names) + r")\b"

df = df.loc[
    df["Name"]
      .astype(str)
      .str.lower()
      .str.contains(pattern, na=False, regex=True)
]

df = clean_and_sort(df, "ROR")

mapping = {'Hagen, Stadt der FernUniversität':'Hagen', 'Solingen, Klingenstadt': 'Solingen'}
df["Name"] = df["Name"].replace(mapping)

df.to_csv("data/processed/ror.csv", index=False)
