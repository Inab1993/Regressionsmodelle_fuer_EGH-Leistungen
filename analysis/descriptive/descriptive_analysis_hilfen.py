import seaborn as sns
import matplotlib.pyplot as plt


from utils.descriptive_utils import basic_descriptive_analysis, descriptive_analysis_by_group, qq_plot
from utils.commons import *

df = pd.read_csv("data/processed/master_2024.csv", sep=",", encoding="UTF-8")
df_cut = df[df[NAME] != "Aachen"]

basic_descriptive_analysis(df, HILFEN_35A)
qq_plot(df, HILFEN_35A)

descriptive_analysis_by_group(df_cut, HILFEN_35A, GEBIETSKOERPERSCHAFT)
descriptive_analysis_by_group(df_cut, HILFEN_35A, TRAEGER)
descriptive_analysis_by_group(df_cut, HILFEN_35A, KREISSTRUKTUR)

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
