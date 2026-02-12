import pandas as pd
from utils.descriptive_utils import basic_descriptive_analysis, descriptive_analysis_by_group, clean_outlier

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")


abi = "Abiturquote"
sgbii ="SGB II-Quote"
kinderarzt="Kinderarztdichte"
kinderanteil = "Kinderanteil"
dichte ="Bevölkerungsdichte"
migration = "Migrationsanteil"
erz_Hilfen = "erz. Hilfen pro 10000"


variables= (kinderanteil, abi, dichte, sgbii, kinderarzt, migration, erz_Hilfen)

df_cut = df[df["Name"] != "Aachen"]
df_clean = df.copy()

for var in variables:
    basic_descriptive_analysis(df, var, shapiro=False)
    descriptive_analysis_by_group(df_cut, var, "Kreisstrukturtyp", shapiro=False)
    descriptive_analysis_by_group(df_cut, var, "Gebietskörperschaft", shapiro=False)


df_cleaned = clean_outlier(df, kinderarzt)
df_cleaned.rename(columns={kinderarzt: f"{kinderarzt}_cleaned"}, inplace=True)

basic_descriptive_analysis(df_cleaned, f"{kinderarzt}_cleaned", shapiro=False)
