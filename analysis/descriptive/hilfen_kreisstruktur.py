import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helper.functions import summarize, read_nrw_map, find_outliers_iqr

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")


df = df[df["Name"] != "Aachen"]

col_x = "35a Hilfen pro 10000"
col_y= "Kreisstrukturtyp"

y_gr_kreis = df.loc[df[col_y] == "Großer Kreis", col_x]
y_kl_kreis = df.loc[df[col_y] == "Kleiner Kreis", col_x]
y_stadt = df.loc[df[col_y] == "Kreisfreie Stadt", col_x]

print("n großer Kreis:", y_gr_kreis.size, " | n kleiner Kreis:", y_kl_kreis.size, " | n kreisfreie Stadt:", y_stadt.size)

desc_table = pd.DataFrame({
    "große Kreise": summarize(y_gr_kreis),
    "kleine Kreise": summarize(y_kl_kreis),
    "kreisfreie Städte": summarize(y_stadt),
})


print(desc_table.round(3))


nrw = read_nrw_map(df)

### KI-generiert
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column="35a Hilfen pro 10000", ax=ax, legend=True, cmap="OrRd", edgecolor="black")

nrw[nrw["Kreisstrukturtyp"] == "Großer Kreis"].plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    hatch='o',
    linewidth=0
)

nrw[nrw["Kreisstrukturtyp"] == "Kleiner Kreis"].plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    hatch='.',
    linewidth=0
)

for idx, row in nrw.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title("In Anspruch genommene 35a Hilfen in NRW auf je 10.000 junge Menschen")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Kreisstrukturtyp",
    y="35a Hilfen pro 10000",
    showfliers=False
)
sns.stripplot(
    data=df,
    x="Kreisstrukturtyp",
    y="35a Hilfen pro 10000",
    color="black",
    alpha=0.5,
    jitter=True
)
plt.title("§ 35a-Quote nach Gebietstyp (Verteilung und Einzelwerte)")
plt.ylabel("35a Hilfen pro 10.000 Kinder")
plt.xlabel("")
plt.show()


mask_by_type = (
    df
    .groupby("Kreisstrukturtyp")["35a Hilfen pro 10000"]
    .transform(find_outliers_iqr)
)


print("Ausreißer: ",df.loc[mask_by_type, ["Name", "Kreisstrukturtyp", "35a Hilfen pro 10000"]])

