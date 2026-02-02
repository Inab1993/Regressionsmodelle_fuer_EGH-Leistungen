import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from scipy import stats

from helper.descriptive_utils import summarize, show_map

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

x = df["35a Hilfen pro 10000"]
print(summarize(x))

#Normalverteilungstest mit Shapiro-Wilk
stat, p = shapiro(x)

print(f"H0: Die 35a-Quote ist normalverteilt.")
print(f"Shapiro-Wilk-Statistik: {stat:.4f}")
print(f"p-Wert: {p:.4f}")

if p < 0.05:
    print("Keine Normalverteilung (H0 wird verworfen).")
else:
    print("Normalverteilung kann nicht verworfen werden (H0 bleibt bestehen).")


#visuelle Überprüfung mit KDE und QQ-Plot
x_np = x.to_numpy()

plt.figure()
plt.hist(x_np, bins=20, density=True, alpha=0.6)

xx = np.linspace(x_np.min(), x_np.max(), 200)

mu = x_np.mean()
sigma = x_np.std(ddof=1) # Degrees fo Freedom n-1 da Strichprobe, nicht Grundgesamtheit
plt.plot(xx, stats.norm.pdf(xx, mu, sigma))

# Bandwith bei standard gelassen, 0.5 liefert ähnliche Ergebnisse, höhere Werte drücken die Kurve
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
            y='Name', x='35a Hilfen pro 10000')
plt.title("In Anspruch genommene 35a Hilfen nach Kreis/Kreisfreier Stadt, auf je 10.000 Kinder")
plt.show()



nrw = show_map(df)
