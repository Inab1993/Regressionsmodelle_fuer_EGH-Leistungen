import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from helper.functions import top_bottom_split_double, find_outliers_iqr, summarize
# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

df = df[df["Name"] != "Aachen"]


summary_by_type = (
    df
    .groupby("Gebietskörperschaft")["35a Hilfen pro 10000"]
    .apply(summarize)
)

print(summary_by_type)

print(top_bottom_split_double(df, "35a Hilfen pro 10000", n=5))

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Gebietskörperschaft",
    y="35a Hilfen pro 10000",
    showfliers=False
)
sns.stripplot(
    data=df,
    x="Gebietskörperschaft",
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
    .groupby("Gebietskörperschaft")["35a Hilfen pro 10000"]
    .transform(find_outliers_iqr)
)

print("Ausreißer: ",df.loc[mask_by_type, ["Name", "Gebietskörperschaft", "35a Hilfen pro 10000"]])
