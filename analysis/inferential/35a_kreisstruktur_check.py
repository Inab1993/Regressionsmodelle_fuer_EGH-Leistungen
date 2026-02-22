import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from utils.commons import *
from utils.descriptive_utils import qq_plot

df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df["Name"] != "Aachen"]

path = result_path("inferential/Kreisstruktur-Effekt")

# Prüfung Gradient Stadt -> große Kreise -> kleine Kreise via KS Test und Plots
df_gr_kreis = df[df[KREISSTRUKTUR] == "Großer Kreis"].copy()
df_stadt = df[df[KREISSTRUKTUR] == "Kreisfreie Stadt"].copy()
df_kl_kreis = df[df[KREISSTRUKTUR] == "Kleiner Kreis"].copy()

groups = [df_gr_kreis[HILFEN_35A],df_kl_kreis[HILFEN_35A],df_stadt[HILFEN_35A]]

for group_name, group_df in groups.items():
    qq_plot(group_df, HILFEN_35A, title=f"inferential/Kreisstruktur-Check{group_name} – {HILFEN_35A}")

# Überprüfung von Varianzhomogenität
lev_stat, lev_p = stats.levene(*groups)

# Einfaktorielle Anova zur Überprüfung der Varianzgleichheit
f_stat, p_value = stats.f_oneway(*groups)

result = {
    "Levene_Stat": round(lev_stat, 3),
    "Levene_p": round(lev_p, 4),
    "ANOVA_F": round(f_stat, 3),
    "ANOVA_p": round(p_value, 4)
}

#  Kernel-Density-Plots zu visuellen Bewertung
sns.kdeplot(df_gr_kreis[HILFEN_35A], label="große Kreise", fill=True)
sns.kdeplot(df_kl_kreis[HILFEN_35A], label="kleine Kreise", fill=True)
sns.kdeplot(df_stadt[HILFEN_35A], label="Städte", fill=True)

path = result_path("inferential/Kreisstruktur-Effekt")

plt.legend()
plt.savefig(
     path / f"kernel_density_plots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

result_df = pd.Series(result, name=HILFEN_35A)
result_df.index.name = "Variable"
result_df.to_csv(path / f"levene_anova.csv")




