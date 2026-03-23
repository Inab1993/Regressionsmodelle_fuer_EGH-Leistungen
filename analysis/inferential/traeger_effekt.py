import sys

from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns

from utils.inferential_utils import t_test, shapiro_wilk, mann_whitney_u, hetero_diagnostics
from utils.preprocessing_utils import DataView, winsorize_iqr
from utils.commons import *

variables = (
    AUSLAENDER,
    ABITUR,
    SGBII,
    KINDERANTEIL,
    KINDERARZTDICHTE,
    ERZ_HILFEN,
    HILFEN_35A,
)

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")

vars_to_clean = []

view = DataView(df)

view.add_clean_column(KINDERARZTDICHTE, cleaner_func=winsorize_iqr)
view.add_clean_column(ERZ_HILFEN, cleaner_func=winsorize_iqr)

# Modell mit Winsorisierung
df = view.use(cleaned=True, cols=[KINDERARZTDICHTE, ERZ_HILFEN])

df_lwl = df[df[TRAEGER] == "LWL"].copy()
df_lvr = df[df[TRAEGER] == "LVR"].copy()


for var in variables:
    groups = [df_lwl[var], df_lvr[var]]
    sh = shapiro_wilk(df, var, group_col=TRAEGER)
    with open(result_path(f"inferential/traeger_effekt/{var}")/"shapiro.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(sh)

# Nullhypothese (H₀): Die mittlere Inanspruchnahme ist bei LWL und LVR gleich.
ttest_list=[]

for var in [HILFEN_35A, KINDERARZTDICHTE,  ERZ_HILFEN, KINDERANTEIL]:
    res = t_test(df, TRAEGER, var, equal_var=False)
    ttest_list.append(res)

ttests = pd.concat(ttest_list, axis=1)
ttests.to_csv(result_path("inferential/traeger_effekt")/"welchttest_results.csv", index=False)



mwu_list=[]
for var in [AUSLAENDER, ABITUR, SGBII]:
    res = mann_whitney_u(df, TRAEGER, var)
    mwu_list.append(res)
mwutests = pd.concat(mwu_list, axis=1)
mwutests.to_csv(result_path("inferential/traeger_effekt")/"mwu_test_results.csv", index=False)

df = df[df["Name"] != "Aachen"]

df[GEBIETSKOERPERSCHAFT] = df[GEBIETSKOERPERSCHAFT].astype("category")
df[TRAEGER] = df[TRAEGER].astype("category")

for var in [HILFEN_35A, AUSLAENDER, KINDERARZTDICHTE]:
    plt.figure()
    fig = sns.regplot(x=var, y=HILFEN_35A, data=df)
    plt.savefig(result_path(f"inferential/traeger_effekt/{var}") / "reg_plot.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    m1 = smf.ols(f'Q("{var}") ~ C(Q("{TRAEGER}"))', data=df).fit()
    sh_stat, sh_p = stats.shapiro(m1.resid)

    fig = sm.qqplot(m1.resid, line="45", fit=True)
    fig.savefig(result_path(f"inferential/traeger_effekt/{var}") / "qq_plot_resids.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(result_path(f"inferential/traeger_effekt/{var}")/"shapiro_het_tests_resids.txt", "w", encoding="utf-8") as f:
            sys.stdout = f
            print(sh_stat)
            print(sh_p)
            print(hetero_diagnostics(m1))


    with open(result_path(f"inferential/traeger_effekt/{var}")/"OLS_model1.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(m1.summary())

    m2 = smf.ols(f'Q("{var}") ~ C(Q("{TRAEGER}")) + C({GEBIETSKOERPERSCHAFT})', data=df).fit()

    with open(result_path(f"inferential/traeger_effekt/{var}")/"OLS_model2.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(m2.summary())

for var in [ABITUR]:
    plt.figure()
    fig = sns.regplot(x=var, y=HILFEN_35A, data=df)
    plt.savefig(result_path(f"inferential/traeger_effekt/{var}") / "reg_plot.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    m1 = smf.ols(f'Q("{var}") ~ C(Q("{TRAEGER}"))', data=df).fit(cov_type="HC3")
    sh_stat, sh_p = stats.shapiro(m1.resid)

    fig = sm.qqplot(m1.resid, line="45", fit=True)
    fig.savefig(result_path(f"inferential/traeger_effekt/{var}") / "qq_plot_resids.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(result_path(f"inferential/traeger_effekt/{var}")/"shapiro_het_tests_resids.txt", "w", encoding="utf-8") as f:
            sys.stdout = f
            print(sh_stat)
            print(sh_p)
            print(hetero_diagnostics(m1))


    with open(result_path(f"inferential/traeger_effekt/{var}")/"OLS_model1.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(m1.summary())

    m2 = smf.ols(f'Q("{var}") ~ C(Q("{TRAEGER}")) + C({GEBIETSKOERPERSCHAFT})', data=df).fit()

    with open(result_path(f"inferential/traeger_effekt/{var}")/"OLS_model2.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(m2.summary())



