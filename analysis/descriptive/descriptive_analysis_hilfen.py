import seaborn as sns
import matplotlib.pyplot as plt

from utils.descriptive_utils import describe_and_save, describe_and_save_grouped, plot, show_map
from utils.commons import *

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df[NAME] != "Aachen"]


describe_and_save(df, HILFEN_35A, map="NRW")
fig = plot(df, HILFEN_35A, type="qq")
fig.savefig(result_path(f"descriptive/{HILFEN_35A}") / f"qq_plot_{HILFEN_35A}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

describe_and_save_grouped(df_cut, HILFEN_35A, GEBIETSKOERPERSCHAFT)
describe_and_save_grouped(df_cut, HILFEN_35A, TRAEGER)
describe_and_save_grouped(df_cut, HILFEN_35A, KREISSTRUKTUR)

fig=show_map(df, HILFEN_35A, group_col=TRAEGER)
fig.savefig(result_path(f"descriptive/{HILFEN_35A}") / f"map_{HILFEN_35A}_by_{TRAEGER}.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig=plot(df, HILFEN_35A, type="dist_kde", group_col=GEBIETSKOERPERSCHAFT, bw_method=1.0, bins=30)
fig.savefig(result_path(f"descriptive/{HILFEN_35A}") / f"distplot_{HILFEN_35A}_by_{GEBIETSKOERPERSCHAFT}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
