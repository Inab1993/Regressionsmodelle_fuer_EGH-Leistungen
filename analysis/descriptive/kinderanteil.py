import pandas as pd
from helper.descriptive_utils import (summarize,
                                      boxplot_by_group,
                                      grouped_summary,
                                      outlier_table,
                                      plot_distribution,
                                      plot_scatter,
                                      show_map,
                                      corr_pair)

# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

col ="Kinderanteil"

df_cut = df[df["Name"] != "Aachen"]

print(summarize(df, col, shapiro=False), "\n")

print(grouped_summary(df_cut, col, "Kreisstrukturtyp", shapiro=False), "\n")
print(grouped_summary(df_cut, col, "Gebietskörperschaft", shapiro=False), "\n")
print(grouped_summary(df, col, "Überörtlicher Träger", shapiro=False), "\n")

print(corr_pair(df, col))

print(outlier_table(df, col))

print(plot_distribution(df, col))

print(boxplot_by_group(df, col, "Kreisstrukturtyp"))

print(plot_scatter(df, col, hue="Überörtlicher Träger"))

print(show_map(df, col, "Kreisstrukturtyp"))