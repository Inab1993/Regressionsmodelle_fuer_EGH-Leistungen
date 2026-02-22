import pandas as pd
from utils.descriptive_utils import summarize, plot_distribution, plot_scatter, corr_pair

# Einlesen
df = pd.read_csv("data/processed/aggregate_on_ROR_master.csv", sep=",", encoding="UTF-8")

kjp = "KJP-Dichte"
hilfen_ror = "Hilfequote in ROR"

summarize(df, hilfen_ror)
plot_distribution(df, hilfen_ror)

summarize(df, kjp)
plot_distribution(df, kjp)

corr_pair(df, kjp, hilfen_ror)
plot_scatter(df, kjp, hilfen_ror)



