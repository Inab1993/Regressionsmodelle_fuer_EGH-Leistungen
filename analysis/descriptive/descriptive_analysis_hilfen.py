from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from utils.descriptive_utils import basic_descriptive_analysis, descriptive_analysis_by_group, show_map, outlier_table

df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df["Name"] != "Aachen"]

var = "35a Hilfen pro 10000"
group_col_x= "Kreisstrukturtyp"
group_col_y= "Gebietskörperschaft"
group_col_z= "Überörtlicher Träger"

basic_descriptive_analysis(df, var)

for col in (group_col_x, group_col_y, group_col_z):
    descriptive_analysis_by_group(df_cut, var, col)

path = Path(f"../../figures/{var}")
path.mkdir(parents=True, exist_ok=True)

plt.figure()
stats.probplot(df[var].to_numpy(), dist="norm", plot=plt)
plt.title("Q–Q-Plot der 35a-Daten")
plt.savefig(
    path / f"{var}_q_q_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(12,6))
sns.barplot(data=df.sort_values('35a Hilfen pro 10000', ascending=False),
            y='Name', x='35a Hilfen pro 10000')
plt.title("In Anspruch genommene 35a Hilfen nach Kreis/Kreisfreier Stadt, auf je 10.000 Kinder")
plt.savefig(
    path / f"{var}_.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

show_map(df, var, group_col_x)
show_map(df, var, group_col_z)
