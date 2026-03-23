import matplotlib.pyplot as plt
from utils.descriptive_utils import (vif, correlations, describe_and_save_grouped,
                                     describe_and_save, plot_scatter, find_outlier, corr_pair, corr_pair_by_type,
                                     chi2_test_report)
from utils.commons import *
from utils.preprocessing_utils import DataView, winsorize_iqr


df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut=df[df["Name"] != "Aachen"]
variables = (
    AUSLAENDER,
    ABITUR,
    SGBII,
    KINDERANTEIL,
    KINDERARZTDICHTE,
    ERZ_HILFEN,
    BEV_DICHTE,
    KJP,
)

# Multikollinearität und Korrelationen prüfen
vif_data=vif(df, variables)
vif_data.to_csv(result_path("multicollinearity") / "VIF.csv", index=False)

corrs = correlations(df, variables, method="pearson")
corrs.to_csv(result_path("multicollinearity") / "Pearson_Korrelationen.csv", index=False)

vars_new = list(variables)
vars_new.remove(BEV_DICHTE)
variables = tuple(vars_new)

vif_data=vif(df, variables)
vif_data.to_csv(result_path("multicollinearity") / "VIF2.csv", index=False)

fig = plot_scatter(df, SGBII, AUSLAENDER)
fig.savefig(result_path("multicollinearity") / "SGBII_Auslaender_Scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)


report_gebietskoerperschaft = chi2_test_report(df_cut, TRAEGER, GEBIETSKOERPERSCHAFT)
with open("results/multicollinearity/chi2_gebietskoerperschaft.txt", "w", encoding="utf-8") as f:
    f.write(report_gebietskoerperschaft)

report_kreisstruktur = chi2_test_report(df_cut, TRAEGER, KREISSTRUKTUR)
with open("results/multicollinearity/chi2_kreisstrukturtyp.txt", "w", encoding="utf-8") as f:
    f.write(report_kreisstruktur)

# 1. Modell: Deskriptive Statistik inkl. Korrelationen
for var in variables:
    describe_and_save(df, var, map="NRW")
    describe_and_save_grouped(df_cut, var, GEBIETSKOERPERSCHAFT)
    out = find_outlier(df, var)
    if len(out) > 0:
        out.to_csv(result_path(f"descriptive/{var}") / "outlier.csv", index=False)

# 2. Modell: Korrelationen mit winsorisierten Daten
vars_to_clean = [ABITUR, KINDERARZTDICHTE, SGBII, ERZ_HILFEN, KINDERANTEIL, KJP]
view = DataView(df)

for v in vars_to_clean:
    view.add_clean_column(v, cleaner_func=winsorize_iqr)

df = view.use(cleaned=True, cols=vars_to_clean)

for var in vars_to_clean:
    corr_pair(df, var).to_csv(result_path(f"descriptive_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}.csv", index=False)
    corr_pair_by_type(df_cut, var, group=GEBIETSKOERPERSCHAFT).to_csv(result_path(f"descriptive_cleaned/{var}") / f"corr_{var}_to_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}.csv", index=False)
    fig = plot_scatter(df_cut, var, hue=GEBIETSKOERPERSCHAFT)
    fig.savefig(result_path(f"descriptive_cleaned/{var}") / f"scatter_{var}_to_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


