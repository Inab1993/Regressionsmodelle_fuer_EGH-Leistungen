import sys

from matplotlib import pyplot as plt
from scipy import stats
from utils.commons import *
from utils.descriptive_utils import plot
from utils.inferential_utils import shapiro_wilk


df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df["Name"] != "Aachen"]


# Überprüfung von Varianzhomogenität und Normalverteilung
df_gr_kreis = df[df[KREISSTRUKTUR] == "Großer Kreis"].copy()
df_stadt = df[df[KREISSTRUKTUR] == "Kreisfreie Stadt"].copy()
df_kl_kreis = df[df[KREISSTRUKTUR] == "Kleiner Kreis"].copy()

groups = [df_gr_kreis[HILFEN_35A],df_kl_kreis[HILFEN_35A],df_stadt[HILFEN_35A]]

fig = plot(df, HILFEN_35A, group_col=KREISSTRUKTUR, type="qq")
fig.savefig(result_path("inferential/kreisstruktur_effekt")/"qqplots.png", dpi=300, bbox_inches="tight")
plt.close(fig)

sh = shapiro_wilk(df_cut, HILFEN_35A, group_col=KREISSTRUKTUR)
center="mean"
lev_stat, lev_p = stats.levene(*groups, center="mean")

with open("results/inferential/kreisstruktur_effekt/levene_shapiro.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(sh)
    print("___________")
    print(f"Levene-Statistik mit center={center}: ", round(lev_stat, 3))
    print("Levene p-Wert: ", round(lev_p, 4))

# Einfaktorielle Anova zur Überprüfung der Varianzgleichheit
f_stat, p_value = stats.f_oneway(*groups)

with open("results/inferential/kreisstruktur_effekt/ANOVA.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print("ANOVA-Testergebnis:")
    print("F-Statistik ", round(f_stat, 3))
    print("p-Wert ", round(p_value, 4))




