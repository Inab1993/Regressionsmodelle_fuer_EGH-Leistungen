from scipy import stats
import statsmodels.formula.api as smf

from utils.descriptive_utils import find_outlier, boxplot_by_group
from utils.inferential_utils import t_test
from utils.preprocessing_utils import DataView, winsorize_iqr
from utils.commons import *

variables = (
    MIGRATION,
    ABITUR,
    SGBII,
    KINDERANTEIL,
    KINDERARZTDICHTE,
    ERZ_HILFEN,
)


# p ≥ 0,05 → Die Nullhypothese kann nicht verworfen werden.
#   → Kein empirischer Hinweis auf einen Trägereffekt.
# 
# p < 0,05 → Die Nullhypothese wird verworfen.
#   → Die Daten sprechen gegen die Annahme eines fehlenden Trägereffekts.

df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

vars_to_clean = []

for var in variables:
    out = find_outlier(df, var, to_print=False)
    if len(out) >0:
        vars_to_clean.append(var)

view = DataView(df)

for v in vars_to_clean:
    view.add_clean_column(v, cleaner_func=winsorize_iqr)

df = view.use(cleaned=True, cols=vars_to_clean)

df_lwl = df[df[TRAEGER] == "LWL"].copy()
df_lvr = df[df[TRAEGER] == "LVR"].copy()


groups = [df_lwl[HILFEN_35A],df_lvr[HILFEN_35A]]

# Überprüfung von Varianzhomogenität
lev_stat, lev_p = stats.levene(*groups)
print(lev_stat)
print(lev_p)

# erster Test
# Nullhypothese (H₀): Die mittlere Inanspruchnahme von §35a-Hilfen pro 10.000 ist bei LWL und LVR gleich.
t_test(df, TRAEGER, HILFEN_35A, folder="inferential/Trägereffekt")

# Outlier-Bereinigung, um eventuellen Bias auszuschließen



for var in variables:
    t_test(df, TRAEGER, var, folder="inferential/Trägereffekt")
    boxplot_by_group(df, var, TRAEGER)

df = df[df["Name"] != "Aachen"]

df[GEBIETSKOERPERSCHAFT] = df[GEBIETSKOERPERSCHAFT].astype("category")
df[TRAEGER] = df[TRAEGER].astype("category")

m1 = smf.ols(f'Q("{MIGRATION}") ~ C(Q("{TRAEGER}"))', data=df).fit(cov_type="HC3")

print(m1.summary())

m2 = smf.ols(f'Q("{Abitur}") ~ C(Q("{TRAEGER}")) + C({GEBIETSKOERPERSCHAFT})', data=df).fit(cov_type="HC3")
print(m2.summary())
m3 = smf.ols(
    f'Q("{ABITUR}") ~ C(Q("{TRAEGER}")) + C(Q("{GEBIETSKOERPERSCHAFT}")) + Q("{ABITUR}")',
    data=df
).fit(cov_type="HC3")

print(m3.summary())

