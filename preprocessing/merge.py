import pandas as pd
from utils.preprocessing_utils import validate_df


df_hilfen = pd.read_csv("data/processed/hilfen_2024.csv", sep=",", encoding="UTF-8")
df_bevoelkerung = pd.read_csv("data/processed/bevoelkerung_2024.csv", sep=",", encoding="UTF-8")
df_sgbii = pd.read_csv("data/processed/sgb2_2024.csv", sep=",", encoding="UTF-8")
df_arztdichte = pd.read_csv("data/processed/arztdichte_2024.csv", sep=",", encoding="UTF-8")
df_traeger = pd.read_csv("data/processed/traeger.csv", sep=",", encoding="UTF-8")
df_bildung = pd.read_csv("data/processed/abiturquote_2024.csv", sep=",", encoding="UTF-8")
df_dichte = pd.read_csv("data/processed/bevoelkerungsdichte_2024.csv", sep=",", encoding="UTF-8")
df_migration = pd.read_csv("data/processed/migration_2024.csv", sep=",", encoding="UTF-8")

dfs_to_merge = [df_bevoelkerung,df_sgbii, df_arztdichte, df_traeger, df_bildung, df_dichte, df_migration]

df_merged = df_hilfen.copy()
for df_other in dfs_to_merge:
    df_merged = df_merged.merge(df_other, on="Name", how="left")

print(len(df_merged))

# relative Werte berechnen
df_merged["35a Hilfen pro 10000"] = (df_merged["Anzahl 35a Hilfen"] / df_merged["Bevölkerung 6 bis 21"] * 10000).round(2)
df_merged["erz. Hilfen pro 10000"] = (df_merged["Insgesamt"] / df_merged["Bevölkerung 6 bis 21"] * 10000).round(2)
df_merged["SGB II-Quote"] = (df_merged["SGB II-Bezug"] / df_merged["Gesamtbevölkerung"]*100).round(2)
df_merged["Anteil 6 bis 21jähriger"] = df_merged["Anteil 6 bis 21jähriger"].round(2)
df_merged["Abiturquote"] = df_merged["Abiturquote"].round(2)
df_merged["Ausländeranteil"] = (df_merged["Anzahl Ausländer"]/ df_merged["Gesamtbevölkerung"]*100).round(2)

# Essen imputieren
RUHR = ["Bochum", "Dortmund", "Duisburg", "Mülheim an der Ruhr", "Oberhausen", "Gelsenkirchen"]
subset = df_merged.loc[df_merged["Name"].isin(RUHR), "35a Hilfen pro 10000"]

mean = subset.apply(pd.to_numeric, errors="coerce").mean()

df_merged.loc[df_merged["Name"] == "Essen", "35a Hilfen pro 10000"] = round(mean, 0)

# Spaltennamen anpassen
df_merged=df_merged.rename(columns={"Anteil 6 bis 21jähriger": "Kinderanteil"})
df_merged = df_merged[["Name",
                        "Kreisstrukturtyp",
                        "Gebietskörperschaft",
                        "erz. Hilfen pro 10000",
                        "35a Hilfen pro 10000",
                        "Überörtlicher Träger",
                        "Kinderanteil",
                        "SGB II-Quote",
                        "Kinderarztdichte",
                        "KJP-Dichte",
                        "Abiturquote",
                        "Bevölkerungsdichte",
                        "Ausländeranteil"
                        ]]


typ_order = ["Kreisfreie Stadt", "Großer Kreis", "Städteregion", "Kleiner Kreis"]
df_merged["Kreisstrukturtyp"] = pd.Categorical(df_merged["Kreisstrukturtyp"], categories=typ_order, ordered=True)
df_merged = df_merged.sort_values(by=["Kreisstrukturtyp", "Name"], ascending=[True, True])

numerical_cols = ['erz. Hilfen pro 10000','35a Hilfen pro 10000', 'Kinderanteil','SGB II-Quote', 'Kinderarztdichte',  'KJP-Dichte','Bevölkerungsdichte', 'Abiturquote', 'Ausländeranteil']

validate_df(df_merged,
            not_null = numerical_cols,
            positive= numerical_cols,
            numeric= numerical_cols,
            bounds={'Kinderanteil':(0,100),'SGB II-Quote':(0,100), 'Kinderarztdichte':(0,100), 'Abiturquote': (0,100), 'Ausländeranteil': (0,100)},
            key_cols=["Name"],
            df_name="Masterframe")
df_merged.to_csv("data/processed/master_2024.csv", index=False)



base = df_hilfen[["Name", "Anzahl 35a Hilfen"]].merge(df_bevoelkerung[["Name", "Bevölkerung 6 bis 21"]], on="Name", how="inner")
base = base.merge(df_arztdichte[["Name", "ROR"]], on="Name", how="inner")

df_ror = (
    base.groupby("ROR", as_index=False)
        .agg(
            hilfen_sum=("Anzahl 35a Hilfen", "sum"),
            pop_sum=("Bevölkerung 6 bis 21", "sum")
        )
)
df_ror["Hilfequote in ROR"] = (df_ror["hilfen_sum"] / df_ror["pop_sum"]) * 10000

kjp_ror = df_arztdichte[["ROR", "KJP-Dichte"]].drop_duplicates(subset=["ROR"])

df_ror = df_ror.merge(kjp_ror, on="ROR", how="left")
df_ror= df_ror[["ROR", "Hilfequote in ROR", "KJP-Dichte"]]
df_ror.to_csv("data/processed/aggregate_on_ROR_master.csv", index=False)
