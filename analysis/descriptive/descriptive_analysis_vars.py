import pandas as pd
from utils.descriptive_utils import basic_descriptive_analysis, descriptive_analysis_by_group

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

kinderanteil = "Kinderanteil"
abi = "Abiturquote"
dichte ="Bevölkerungsdichte"
sgbii ="SGB II-Quote"
kinderarzt="Kinderarztdichte"
kjp="KJP-Dichte"

variables= (kinderanteil, abi, dichte, sgbii, kinderarzt, kjp)

df_cut = df[df["Name"] != "Aachen"]

for var in variables:
    basic_descriptive_analysis(df, var, shapiro=False)
    descriptive_analysis_by_group(df_cut, var, "Kreisstrukturtyp")
    descriptive_analysis_by_group(df_cut, var, "Überörtlicher Träger")
