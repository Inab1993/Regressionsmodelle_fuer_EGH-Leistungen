import pandas as pd
from matplotlib import pyplot as plt

from utils.commons import result_path
from utils.descriptive_utils import describe_and_save, plot_scatter

# Einlesen
df = pd.read_csv("data/processed/aggregate_on_ROR_master.csv", sep=",", encoding="UTF-8")

kjp = "KJP-Dichte"
hilfen_ror = "Hilfequote in ROR"

describe_and_save(df, hilfen_ror, y=hilfen_ror, folder=f"descriptive/ror_aggregate/{hilfen_ror}")
describe_and_save(df, kjp, y=hilfen_ror, folder=f"descriptive/ror_aggregate/{kjp}")
fig = plot_scatter(df, hilfen_ror, kjp)
fig.savefig(result_path(f"descriptive/ror_aggregate/{kjp}") / "scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)



