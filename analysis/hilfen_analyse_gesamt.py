import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from helper.functions import summarize, read_nrw_map, find_outliers_iqr, top_bottom


# Einlesen
df = pd.read_csv("../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

x = df["35a Hilfen pro 10000"]
print(summarize(x))

#top_bottom(df, "35a Hilfen pro 10000", n=5)


x_np = x.to_numpy()

plt.figure()
plt.hist(x_np, bins=20, density=True, alpha=0.6)

xx = np.linspace(x_np.min(), x_np.max(), 200)

mu = x_np.mean()
sigma = x_np.std(ddof=1) # Degrees fo Freedom n-1 da Strichprobe, nicht Grundgesamtheit
plt.plot(xx, stats.norm.pdf(xx, mu, sigma))

# bandwith bei stardard gelassen, o.5 liefert ähnlcihe ergebnisse, höhere werte drücken die
kde = stats.gaussian_kde(x_np)
plt.plot(xx, kde(xx), color="red")


plt.xlabel("35a Hilfen pro 10.000")
plt.ylabel("Dichte")
plt.title("Histogramm mit Kernel Density Estimation und Normalverteilung")
plt.show()


plt.figure()
stats.probplot(x_np, dist="norm", plot=plt)
plt.title("Q–Q-Plot der 35a-Daten")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(data=df.sort_values('35a Hilfen pro 10000', ascending=False),
            x='Name', y='35a Hilfen pro 10000')
plt.xticks(rotation=90)
plt.title("In Anspruch genommene 35a Hilfen nach Kreis/Kreisfreier Stadt, auf je 10.000 Kinder")
plt.show()

mask = find_outliers_iqr(x)
print(df.loc[mask, ["Name", "35a Hilfen pro 10000"]])

# Visualisierung
nrw = read_nrw_map(df)
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column="35a Hilfen pro 10000", ax=ax, legend=True, cmap="OrRd", edgecolor="black")

for idx, row in nrw.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    else:  # MultiPolygon
        x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title("In Anspruch genommene 35a Hilfen in NRW auf je 10.000 junge Menschen")
plt.show()