import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.descriptive_utils import basic_descriptive_analysis, descriptive_analysis_by_group, find_outlier
from utils.commons import *

# Einlesen
df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")

variables = (
    AUSLAENDER,
    ABITUR,
    SGBII,
    KINDERANTEIL,
    KINDERARZTDICHTE,
    ERZ_HILFEN,
    BEV_DICHTE
)


def vif(dataframe, vars: tuple):
    # Varianzinflationfaktor
    X = dataframe[[*vars]].dropna()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data = vif_data[vif_data["Variable"] != "const"]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    vif_data.to_csv(result_path("multicollinearity") / "VIF.csv", index=False)


def correlations(df: pd.DataFrame, vars: tuple):
    corr = df[[*vars]].corr(method="spearman")

    # Nur obere Dreiecksmatrix (ohne Diagonale)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_pairs = corr.where(mask)

    corr_pairs = (
        corr_pairs
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Variable_1", "Variable_2", "r"]

    filtered = corr_pairs[abs(corr_pairs["r"]) > 0.5]

    filtered.sort_values("r", ascending=False)
    filtered.to_csv(result_path("multicollinearity") / "Korrelationen.csv", index=False)



# Multikollinearität und Korrelationen prüfen
vif(df, variables)
# Bevölkerungsdichte mit Faktor 5,9 wird entfernt

vars_new = list(variables)
vars_new.remove(BEV_DICHTE)
variables = tuple(vars_new)

correlations(df, variables)

# Hohe Korrelation bei SGB II Quote und Migrationsanteil, später in der Regression bedenken.

# Deskriptive Analyse
for var in variables:
    basic_descriptive_analysis(df, var)
    descriptive_analysis_by_group(df[df["Name"] != "Aachen"], var, GEBIETSKOERPERSCHAFT)
    out = find_outlier(df, var)





