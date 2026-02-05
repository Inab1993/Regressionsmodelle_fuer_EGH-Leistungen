import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from utils.preprocessing_utils import clean_and_sort, preprocess

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024c). Sozialversicherungs-
pflichtige Beschäftigte (Arbeitsort) nach Wirtschaftsabschnitte sowie Schul- und Berufsab-
schluss - kreisfreie Städte und Kreise - Stichtag (ab 2021). Landesdatenbank NRW, abge-
rufen am 28.01.2026. URL: https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=13111-54i"""
df = pd.read_csv("../data/raw/2024/13111-54i.csv", sep=";", encoding="latin1", skiprows=6)


df = df[df["Unnamed: 3"] == "Insgesamt"]
df = df.drop(df.columns[[0, 2, 3, -5, -4, -3, -2,-1]], axis=1)

# Spalten umbenennen
df = df.rename(columns={"Unnamed: 1": "Name"})

type_dict = {
    'Insgesamt': "int",
    'ohne Schulabschluss':"int",
    'mit Volks-/Haupschulabschluss':"int",
    'mit mittlerer Reife/gleichwertigem Abschluss':"int",
    'mit Abitur/Fachabitur':"int",
}

df = preprocess(df, type_dict)

df = clean_and_sort(df, 'Insgesamt', 'ohne Schulabschluss',
       'mit Volks-/Haupschulabschluss',
       'mit mittlerer Reife/gleichwertigem Abschluss', 'mit Abitur/Fachabitur')


abschluss_cols = df.columns.drop(["Name", "Insgesamt"])

df[abschluss_cols] = df[abschluss_cols].div(df["Insgesamt"], axis=0)


# Gewichte mit gleichen Abständen
weights = {
    'ohne Schulabschluss':0,
    'mit Volks-/Haupschulabschluss':1,
    'mit mittlerer Reife/gleichwertigem Abschluss':2,
    'mit Abitur/Fachabitur':3
}

df["Bildungsindex_equal"] = sum(
    df[col] * weight
    for col, weight in weights.items()
)

weights = {
    'ohne Schulabschluss':0,
    'mit Volks-/Haupschulabschluss':0,
    'mit mittlerer Reife/gleichwertigem Abschluss':10,
    'mit Abitur/Fachabitur':20
}

df["Bildungsindex_diff"] = sum(
    df[col] * weight
    for col, weight in weights.items()
)

df = df[["Name", 'mit Abitur/Fachabitur', "Bildungsindex_equal","Bildungsindex_diff"]]

# Kolmogorov-Smirnov-Test zur Messung des maximalen Abstands beider Verteilungen und Kernel-Density-Plots zu visuellen Bewertung
# KS Test, da ich zwei Stichproben habe und deren Verteilung vergleichen will, mu und sigma sind nicht geschätzt
sns.kdeplot(df["Bildungsindex_equal"], label="Bildungsindex gleich", fill=True)
sns.kdeplot(df["mit Abitur/Fachabitur"], label="mit Abitur/Fachabitur", fill=True)
sns.kdeplot(df["Bildungsindex_diff"], label="Bildungsindex ungleich", fill=True)

plt.legend()
plt.show()

z = StandardScaler()
x1 = z.fit_transform(df[["Bildungsindex_equal"]]).ravel()
x2 = z.fit_transform(df[["mit Abitur/Fachabitur"]]).ravel()
x3 = z.fit_transform(df[["Bildungsindex_diff"]]).ravel()

print("gleich vs abi: ",ks_2samp(x1, x2))
print("gleich vs ungleich: ",ks_2samp(x1, x3))
print("ungleich vs abi: ",ks_2samp(x2, x3))
