import sys

from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

from utils.inferential_utils import hetero_diagnostics
from utils.commons import *


df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")

scaler = StandardScaler()

df[[ABITUR, BILDUNG]] = scaler.fit_transform(
    df[[ABITUR, BILDUNG]]
)

for var in [BILDUNG, ABITUR]:
    m1 = smf.ols(f'Q("{HILFEN_35A}") ~ Q("{var}")', data=df).fit()
    sh_stat, sh_p = stats.shapiro(m1.resid)

    fig = sm.qqplot(m1.resid, line="45", fit=True)
    fig.savefig(result_path(f"inferential/bildung_sensitivitaetsanalyse/{var}") / "qq_plot_resids.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    fig = sns.regplot(x=var, y=HILFEN_35A, data=df)
    plt.savefig(result_path(f"inferential/bildung_sensitivitaetsanalyse/{var}") / "reg_plot.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    with open(result_path(f"inferential/bildung_sensitivitaetsanalyse/{var}")/"shapiro_het_tests_resids.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(sh_stat)
        print(sh_p)
        print(hetero_diagnostics(m1))

    m2 = smf.ols(f'Q("{HILFEN_35A}") ~ Q("{var}")', data=df).fit(cov_type="HC3")

    with open(result_path(f"inferential/bildung_sensitivitaetsanalyse/{var}")/"OLS.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print(m2.summary())

