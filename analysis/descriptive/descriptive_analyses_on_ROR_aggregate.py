import pandas as pd
from utils.descriptive_utils import summarize, plot_distribution, plot_scatter, corr_pair
# Einlesen
df = pd.read_csv("../../data/processed/aggregate_on_ROR_master.csv", sep=",", encoding="UTF-8")

hilfequote = "Hilfequote in ROR"
kjp = "KJP-Dichte"

summarize(df, hilfequote)
plot_distribution(df, hilfequote)

summarize(df, kjp)
plot_distribution(df, kjp)

corr_pair(df, kjp, hilfequote)
plot_scatter(df, kjp, hilfequote)



