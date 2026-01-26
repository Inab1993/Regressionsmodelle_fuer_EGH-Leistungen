import pandas as pd
import re

from helper.preprocessing import clean_and_sort

# Quelle: https://www.bbsr.bund.de/BBSR/DE/forschung/raumbeobachtung/Raumabgrenzungen/downloads/download-referenzen.html
df = pd.read_excel("data/raw/raumgliederungen-referenzen-2023.xlsx", skiprows=1, sheet_name="Kreisreferenz")

df=df[["Kreise (2023) Name", "Raumordnungsregionen (2023) Name"]]
df=df.rename(columns={"Kreise (2023) Name": "Name", "Raumordnungsregionen (2023) Name": "ROR"})

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
