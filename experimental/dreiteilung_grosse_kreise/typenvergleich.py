import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.commons import result_path

df_dichte = pd.read_csv("../../data/processed/bevoelkerungsdichte_2024.csv")
df_typen = pd.read_csv("typen.csv")

df = df_dichte.merge(df_typen, how="left", on="Name")
df.groupby("Kreisstrukturtyp")["Bevölkerungsdichte"].describe()

sns.boxplot(data=df, x="Kreisstrukturtyp", y="Bevölkerungsdichte")
path = result_path("experimental")
plt.savefig(
     path / "Dichtevergleich_Staedteregion.png",
    dpi=300,
    bbox_inches="tight"
)

aachen = df.loc[df["Name"] == "Aachen", "Bevölkerungsdichte"].iloc[0]
median_summary = df.groupby("Kreisstrukturtyp")["Bevölkerungsdichte"].median()

dist_to_city = abs(aachen - median_summary["Kreisfreie Stadt"])
dist_to_large = abs(aachen - median_summary["Großer Kreis"])

print(dist_to_city, dist_to_large)
