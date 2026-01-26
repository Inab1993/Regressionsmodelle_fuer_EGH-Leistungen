import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from helper.functions import find_outliers_iqr, summarize, read_nrw_map

df = pd.read_csv("../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

col ="Bildungsindex"
br = df[col]

summarize(br)
x = br.to_numpy()

plt.figure()
plt.hist(x, bins=20, density=True, alpha=0.6)

xx = np.linspace(x.min(), x.max(), 200)

mu = x.mean()
sigma = x.std(ddof=1) # Degrees fo Freedom n-1 da Strichprobe, nicht Grundgesamtheit
plt.plot(xx, stats.norm.pdf(xx, mu, sigma))

kde = stats.gaussian_kde(x)
plt.plot(xx, kde(xx), color="red")


plt.xlabel("Bildungsindex")
plt.ylabel("Dichte")
plt.title("Histogramm mit Kernel Density Estimation und Normalverteilung")
plt.show()

mask = find_outliers_iqr(br)
print(df.loc[mask, ["Name", col]])

# Visualisierung
nrw=read_nrw_map(df)
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column=col, ax=ax, legend=True, cmap="OrRd", edgecolor="black")

for idx, row in nrw.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    else:  # MultiPolygon
        x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title("Bildungsindex in NRW")
plt.show()

df = df[df["Name"] != "Aachen"]

summary_by_type2 = (
    df
    .groupby("Typ 2")[col]
    .apply(summarize)
    .round(3)
)

summary_by_type1 = (
    df
    .groupby("Typ 1")[col]
    .apply(summarize)
    .round(3)
)

print(summary_by_type1, "\n", summary_by_type2)

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Typ 1",
    y=col,
    showfliers=False
)
sns.stripplot(
    data=df,
    x="Typ 1",
    y=col,
    color="black",
    alpha=0.5,
    jitter=True
)
plt.title("Bildungsindex nach Gebietstyp (Verteilung und Einzelwerte)")
plt.ylabel("Bildungsindex")
plt.xlabel("")
plt.show()

plt.show()
mask_by_type = (
    df
    .groupby("Typ 1")[col]
    .transform(find_outliers_iqr)
)

print(df.loc[mask_by_type, ["Name", "Typ 1", col]])