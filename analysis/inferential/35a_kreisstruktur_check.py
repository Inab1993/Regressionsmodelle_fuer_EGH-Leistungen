import sys

from scipy import stats
from utils.commons import *
from utils.descriptive_utils import plot
import matplotlib.pyplot as plt

from utils.inferential_utils import shapiro_wilk

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df["Name"] != "Aachen"]


# Überprüfung der Normalverteilung
df_gr_kreis = df[df[KREISSTRUKTUR] == "Großer Kreis"].copy()
df_stadt = df[df[KREISSTRUKTUR] == "Kreisfreie Stadt"].copy()
df_kl_kreis = df[df[KREISSTRUKTUR] == "Kleiner Kreis"].copy()

groups = [df_gr_kreis[HILFEN_35A],df_kl_kreis[HILFEN_35A],df_stadt[HILFEN_35A]]

fig = plot(df, HILFEN_35A, group_col=KREISSTRUKTUR, type="qq")
fig.savefig(result_path("inferential/kreisstruktur_effekt")/"qqplots.png", dpi=300, bbox_inches="tight")
plt.close(fig)

sh = shapiro_wilk(df_cut, HILFEN_35A, group_col=KREISSTRUKTUR)

with open(result_path("inferential/kreisstruktur_effekt")/"shapiro.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(sh)

f_stat, p_value = stats.f_oneway(*groups, equal_var=False)

with open("results/inferential/kreisstruktur_effekt/ANOVA.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print("ANOVA-Testergebnis:")
    print("F-Statistik ", round(f_stat, 3))
    print("p-Wert ", round(p_value, 4))




