import pandas as pd
import re

""" Quelle: Kassenärztliche Bundesvereinigung (2024). Regionale Verteilung von Ärzten. Kassenärzt-
liche Bundesvereinigung KdöR, abgerufen am 28.01.2026. URL: https://www.kbv.de/infothek/zahlen-und-fakten/gesundheitsdaten/aerzte-regionale-verteilung"""
df = pd.read_excel("data/raw/I.1.1.8.xls", skiprows=3, sheet_name="2024")

df = df[
    [
        "Name",
        "Regionstyp",
        'Kinderärzte.6',
        'Kind.Jug.Psychiater.6'
    ]
]

df = df.rename(columns={'Kinderärzte.6': 'Kinderarztdichte','Kind.Jug.Psychiater.6': 'KJP-Dichte'})

df = df.dropna(axis=1, how="all")
df = df.dropna(axis=0, how="all")

df_ka = df.copy()

df_ka = df_ka.loc[df["Regionstyp"].astype(str).str.contains("Kreis", case=False, na=False)]
df_ka = df_ka.drop(columns=["Regionstyp"])
df_ka = df_ka.drop(columns=["KJP-Dichte"])

df_kjp = df.copy()

df_kjp = df_kjp.loc[df["Regionstyp"].astype(str).str.contains("Raumordnungsregion", case=False, na=False)]
df_kjp = df_kjp.drop(columns=["Regionstyp"])
df_kjp = df_kjp.drop(columns=["Kinderarztdichte"])

# Filtern der Kreise/Städte NRW für die Kinderarztdichte
df2 = pd.read_csv("data/processed/hilfen_2024.csv")

names = (
    df2["Name"]
    .dropna()
    .astype(str)
    .str.lower()
    .apply(re.escape)  
)

pattern = r"\b(" + "|".join(names) + r")\b"

df_ka = df_ka.loc[
    df["Name"]
      .astype(str)
      .str.lower()
      .str.contains(pattern, na=False, regex=True)
]

df_ka["Name"] = df_ka["Name"].str.replace(", Stadt", "", regex=False).str.strip()

df_ka.loc[df_ka["Name"] == "Städteregion Aachen", "Name"] = "Aachen"


df_ka["Kinderarztdichte"] = pd.to_numeric(df["Kinderarztdichte"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df_kjp["KJP-Dichte"] = pd.to_numeric(df["KJP-Dichte"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

# Tabelle mit Raumordnungsregionen einlesen
df_ror = pd.read_csv("data/processed/ror.csv")
df_kjp = df_kjp.rename(columns={"Name":"ROR"})  

# mit KJP-Dichte-Tabelle matchen, um Kreise/Städte den Raumordnungsregionen zuzuorten
df_kjp = df_kjp.rename(columns={"Name":"ROR"}) 
df_merged = df_ror.merge(df_kjp, on=["ROR"], how="left")

df = df_ka.merge(df_merged, on=["Name"], how="left")
df.to_csv("data/processed/arztdichte_2024.csv", index=False)