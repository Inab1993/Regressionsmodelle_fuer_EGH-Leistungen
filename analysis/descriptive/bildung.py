import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from helper.descriptive_utils import summarize, read_nrw_map, grouped_summary, outlier_table

df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

col="Abiturquote"

df = df[df["Name"] != "Aachen"]

print(summarize(df, col, shapiro=False), "\n")

print(grouped_summary(df, col, ["Kreisstrukturtyp"], shapiro=False), "\n")
print(grouped_summary(df, col, ["Gebietskörperschaft"], shapiro=False), "\n")
print(grouped_summary(df, col, ["Überörtlicher Träger"], shapiro=False), "\n")

print(outlier_table(df, col))

"""
bildung = df[col]

summarize(bildung)

x = bildung.to_numpy()

plt.figure()
plt.hist(x, bins=20, density=True, alpha=0.6)

xx = np.linspace(x.min(), x.max(), 200)

mu = x.mean()
sigma = x.std(ddof=1) # Degrees fo Freedom n-1 da Strichprobe, nicht Grundgesamtheit
plt.plot(xx, stats.norm.pdf(xx, mu, sigma))

kde = stats.gaussian_kde(x)
plt.plot(xx, kde(xx), color="red")


plt.xlabel(col)
plt.ylabel("Dichte")
plt.title("Histogramm mit Kernel Density Estimation und Normalverteilung")
plt.show()

mask = find_outliers_iqr(bildung)
print(df.loc[mask, ["Name", col]])

# Visualisierung
nrw=read_nrw_map(df)
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column=col, ax=ax, legend=True, cmap="OrRd", edgecolor="black")

for idx, row in nrw.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title(col)
plt.show()

df = df[df["Name"] != "Aachen"]

summary_by_type2 = (
    df
    .groupby("Gebietskörperschaft")[col]
    .apply(summarize)
    .round(3)
)

summary_by_type1 = (
    df
    .groupby("Kreisstrukturtyp")[col]
    .apply(summarize)
    .round(3)
)

print(summary_by_type1, "\n", summary_by_type2)

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Gebietskörperschaft",
    y=col,
    showfliers=False
)
sns.stripplot(
    data=df,
    x="Gebietskörperschaft",
    y=col,
    color="black",
    alpha=0.5,
    jitter=True
)

plt.ylabel(col)
plt.xlabel("")
plt.show()

plt.show()
mask_by_type = (
    df
    .groupby("Kreisstrukturtyp")[col]
    .transform(find_outliers_iqr)
)

print(df.loc[mask_by_type, ["Name", "Kreisstrukturtyp", col]])"""