import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper.descriptive_utils import summarize

df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")


col_x = "35a Hilfen pro 10000"
col_y= "Überörtlicher Träger"

y_lwl = df.loc[df[col_y] == "LWL", col_x]
y_lvr = df.loc[df[col_y] == "LVR", col_x]

print("n LWL:", y_lwl.size, " | n LVR:", y_lvr.size)

desc_table = pd.DataFrame({
    "LWL": summarize(y_lwl),
    "LVR": summarize(y_lvr),
})

desc_table["diff (LVR-LWL)"] = desc_table["LVR"] - desc_table["LWL"]

print(desc_table.round(3))

plt.figure(figsize=(7, 4))
plt.boxplot([y_lwl, y_lvr], tick_labels=["LWL", "LVR"], showmeans=True)
plt.title("§35a Hilfen pro 10.000 nach überörtlichem Träger")
plt.tight_layout()
plt.show()


nrw = read_nrw_map(df)

### KI-generiert
fig, ax = plt.subplots(1, 1, figsize=(18, 21))
nrw.plot(column="35a Hilfen pro 10000", ax=ax, legend=True, cmap="OrRd", edgecolor="black")

nrw[nrw["Überörtlicher Träger"] == "LWL"].plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    hatch="///",
    linewidth=0
)

for idx, row in nrw.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['GN'], fontsize=8, ha='center', va='center')

ax.set_title("In Anspruch genommene 35a Hilfen in NRW auf je 10.000 junge Menschen")
plt.show()
