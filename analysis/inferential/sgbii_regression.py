import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.inferential_utils import hetero_diagnostics, run_regression, scatter_with_ols_line


# Einlesen
df = pd.read_csv("../../data/processed/master_2024.csv", sep=",", encoding="UTF-8")

y = "35a Hilfen pro 10000"
x = "SGB II-Quote"
group_col = "Kreisstrukturtyp"

df_kreis = df[df[group_col] == "Großer Kreis"].copy()

scatter_with_ols_line(df, x, y)

model, used = run_regression(df, x, y)

hetero_diagnostics(model, x)

