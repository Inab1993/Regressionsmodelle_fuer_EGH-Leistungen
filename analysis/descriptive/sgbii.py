import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from PIL.ImageColor import colormap
from scipy import stats

from helper.descriptive_utils import summarize,read_nrw_map, grouped_summary, outlier_table

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

col ="SGB II-Quote"

df_cut = df[df["Name"] != "Aachen"]

print(summarize(df, col, shapiro=False), "\n")

print(grouped_summary(df_cut, col, ["Kreisstrukturtyp"], shapiro=False), "\n")
print(grouped_summary(df_cut, col, ["Gebietskörperschaft"], shapiro=False), "\n")
print(grouped_summary(df_cut, col, ["Überörtlicher Träger"], shapiro=False), "\n")

print(outlier_table(df, col))

"""
sgbii = df[col]

summarize(sgbii)

x = df[col].to_numpy()

plt.figure()
plt.hist(x, bins=20, density=True, alpha=0.6)

xx = np.linspace(x.min(), x.max(), 200)

mu = x.mean()
sigma = x.std(ddof=1) # Degrees fo Freedom n-1 da Strichprobe, nicht Grundgesamtheit
plt.plot(xx, stats.norm.pdf(xx, mu, sigma))

# bandwith bei stardard gelassen, o.5 liefert ähnlcihe ergebnisse, höhere werte drücken die
kde = stats.gaussian_kde(x)
plt.plot(xx, kde(xx), color="red")


plt.xlabel("SGB II-Quote")
plt.ylabel("Dichte")
plt.title("Histogramm mit Kernel Density Estimation und Normalverteilung")
plt.show()

mask = find_outliers_iqr(sgbii)

print("Ausreißer: ",df.loc[mask, ["Name", col]])


# Visualisierung
nrw = read_nrw_map(df)
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column=col, ax=ax, legend=True, cmap="OrRd", edgecolor="black")

for idx, row in nrw.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title("SGB II Quote in NRW")
plt.show()

df = df[df["Name"] != "Aachen"]

col_x = sgbii
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


plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Kreisstrukturtyp",
    y=col,
    showfliers=False
)
sns.stripplot(
    data=df,
    x="Kreisstrukturtyp",
    y=col,
    color="black",
    alpha=0.5,
    jitter=True
)
plt.title("SGB II Quote nach Gebietstyp (Verteilung und Einzelwerte)")
plt.ylabel("SGB II Quote")
plt.xlabel("")
plt.show()



mask_by_type = (
    df
    .groupby("Kreisstrukturtyp")[col]
    .transform(find_outliers_iqr)
)

print(df.loc[mask_by_type, ["Name", "Kreisstrukturtyp", col]])
"""

