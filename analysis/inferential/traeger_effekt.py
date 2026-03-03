import sys

from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

from utils.descriptive_utils import find_outlier, boxplot_by_group, plot
from utils.inferential_utils import t_test, shapiro_wilk
from utils.preprocessing_utils import DataView, winsorize_iqr
from utils.commons import *

variables = (
    AUSLAENDER,
    ABITUR,
    SGBII,
    KINDERANTEIL,
    KINDERARZTDICHTE,
    ERZ_HILFEN,
)

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")

vars_to_clean = []

for var in variables:
    out = find_outlier(df, var, to_print=False)
    if len(out) >0:
        vars_to_clean.append(var)

view = DataView(df)

for v in vars_to_clean:
    view.add_clean_column(v, cleaner_func=winsorize_iqr)

# Modell ohne Winsorisierung
df = view.use(cleaned=False, cols=vars_to_clean)

df_lwl = df[df[TRAEGER] == "LWL"].copy()
df_lvr = df[df[TRAEGER] == "LVR"].copy()

groups = [df_lwl[HILFEN_35A],df_lvr[HILFEN_35A]]

fig=plot(df, HILFEN_35A, type="qq", group_col=TRAEGER)
fig.savefig(result_path("inferential/traegereffekt")/f"qqplots_{HILFEN_35A}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
sh = shapiro_wilk(df, HILFEN_35A, group_col=TRAEGER)

# Überprüfung von Varianzhomogenität
center="mean"
lev_stat, lev_p = stats.levene(*groups, center="mean")

with open("results/inferential/traeger_effekt/levene_shapiro.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(sh)
    print("___________")
    print(f"Levene-Statistik mit center={center}: ", round(lev_stat, 3))
    print("Levene p-Wert: ", round(lev_p, 4))

# erster Test
# Nullhypothese (H₀): Die mittlere Inanspruchnahme von §35a-Hilfen pro 10.000 ist bei LWL und LVR gleich.
t_test=t_test(df, TRAEGER, HILFEN_35A)

concat_list=[]

for var in variables:
    concat_list.append(t_test(df, TRAEGER, var))
    fig = boxplot_by_group(df, var, TRAEGER)
    fig.savefig(result_path("inferential/traegereffekt")/f"boxplot_by_{var}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


ttests = pd.concat(concat_list, axis=1)
ttests.to_csv(result_path("inferential/traegereffekt")/"ttest_results.csv", index=False)

df = df[df["Name"] != "Aachen"]

df[GEBIETSKOERPERSCHAFT] = df[GEBIETSKOERPERSCHAFT].astype("category")
df[TRAEGER] = df[TRAEGER].astype("category")

m1 = smf.ols(f'Q("{AUSLAENDER}") ~ C(Q("{TRAEGER}"))', data=df).fit(cov_type="HC3")

print(m1.summary())

m2 = smf.ols(f'Q("{AUSLAENDER}") ~ C(Q("{TRAEGER}")) + C({GEBIETSKOERPERSCHAFT})', data=df).fit(cov_type="HC3")

print(m2.summary())


