import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from utils.descriptive_utils import (vif, correlations, describe_and_save_grouped,
                                     describe_and_save, plot_scatter, find_outlier, corr_pair, corr_pair_by_type)
from utils.commons import *
from utils.preprocessing_utils import DataView, winsorize_iqr


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

# Multikollinearität und Korrelationen prüfen
vif_data=vif(df, variables)
vif_data.to_csv(result_path("multicollinearity") / "VIF.csv", index=False)

contingency_table = pd.crosstab(df[TRAEGER], df[GEBIETSKOERPERSCHAFT])
chi2, p, dof, expected = chi2_contingency(contingency_table)

n = contingency_table.to_numpy().sum()
r, k = contingency_table.shape
cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

with open("results/multicollinearity/chi2.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(f"Chi-Quadrat-Statistik: {chi2:.4f}")
    print(f"p-Wert: {p:.4f}")
    print(f"Cramér's V: {cramers_v:.4f}")


vars_new = list(variables)
vars_new.remove(BEV_DICHTE)
variables = tuple(vars_new)

corrs = correlations(df, variables, method="pearson")
corrs.to_csv(result_path("multicollinearity") / "Pearson_Korrelationen.csv", index=False)
fig = plot_scatter(df, SGBII, AUSLAENDER)
fig.savefig(result_path("multicollinearity") / "SGBII_Auslaender_Scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 1. Modell: Deskriptive Statistik inkl. Korrelationen

for var in variables:
    describe_and_save(df, var)
    describe_and_save_grouped(df[df["Name"] != "Aachen"], var, GEBIETSKOERPERSCHAFT)
    out = find_outlier(df, var)
    if len(out) > 0:
        out.to_csv(result_path(f"descriptives/{var}") / "outlier.csv", index=False)


# 2. Modell: Korrelationen mit winsorisierten Daten
vars_to_clean = [ABITUR, KINDERARZTDICHTE, SGBII, ERZ_HILFEN, KINDERANTEIL]
view = DataView(df)

for v in vars_to_clean:
    view.add_clean_column(v, cleaner_func=winsorize_iqr)


df = view.use(cleaned=True, cols=vars_to_clean)
df=df[df["Name"] != "Aachen"]
for var in vars_to_clean:
    corr_pair(df, var).to_csv(result_path(f"descriptives_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}.csv", index=False)
    corr_pair_by_type(df, var, group=GEBIETSKOERPERSCHAFT).to_csv(result_path(f"descriptives_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}.csv", index=False)
    fig = plot_scatter(df, var, hue=GEBIETSKOERPERSCHAFT)
    fig.savefig(result_path(f"descriptives_cleaned/{var}") / f"scatter_{var}_to_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

corr_pair(df, KINDERANTEIL, method="pearson").to_csv(result_path(f"descriptives_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}_pearson.csv", index=False)
corr_pair_by_type(df, KINDERANTEIL, group=GEBIETSKOERPERSCHAFT, method="pearson").to_csv(result_path(f"descriptives_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}_pearson.csv", index=False)

