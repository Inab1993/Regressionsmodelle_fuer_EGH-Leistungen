import seaborn as sns
import matplotlib.pyplot as plt

from utils.descriptive_utils import describe_and_save, describe_and_save_grouped, plot, show_map
from utils.commons import *

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df[NAME] != "Aachen"]


describe_and_save(df, HILFEN_35A)
fig = plot(df, HILFEN_35A, type="qq")
fig.savefig(result_path(f"descriptives/{HILFEN_35A}") / f"qq_plot_{HILFEN_35A}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

describe_and_save_grouped(df_cut, HILFEN_35A, GEBIETSKOERPERSCHAFT)
describe_and_save_grouped(df_cut, HILFEN_35A, TRAEGER)
describe_and_save_grouped(df_cut, HILFEN_35A, KREISSTRUKTUR)

fig=show_map(df, HILFEN_35A, group_col=TRAEGER)
fig.savefig(result_path(f"descriptives/{HILFEN_35A}") / f"map_{HILFEN_35A}_by_{TRAEGER}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

path = result_path(f"descriptives/{HILFEN_35A}")
plt.figure(figsize=(12,6))
sns.barplot(data=df.sort_values(HILFEN_35A, ascending=False),
            y=NAME, x=HILFEN_35A)
plt.title("In Anspruch genommene 35a Hilfen nach Kreis/Kreisfreier Stadt, auf je 10.000 Kinder")
plt.savefig(
    path / f"barplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
