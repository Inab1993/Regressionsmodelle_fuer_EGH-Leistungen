import pandas as pd
from helper.functions import validate_df


df_hilfen = pd.read_csv("data/processed/hilfen_2024.csv", sep=",", encoding="UTF-8")
df_bevoelkerung = pd.read_csv("data/processed/bevoelkerung_2024.csv", sep=",", encoding="UTF-8")
df_sgbii = pd.read_csv("data/processed/sgb2_2024.csv", sep=",", encoding="UTF-8")
df_arztdichte = pd.read_csv("data/processed/arztdichte_2024.csv", sep=",", encoding="UTF-8")
df_traeger = pd.read_csv("data/processed/traeger.csv", sep=",", encoding="UTF-8")
df_bildung = pd.read_csv("data/processed/abiturquote_2024.csv", sep=",", encoding="UTF-8")
df_dichte = pd.read_csv("data/processed/bevoelkerungsdichte_2024.csv", sep=",", encoding="UTF-8")

dfs_to_merge = [df_bevoelkerung,df_sgbii, df_arztdichte, df_traeger, df_bildung, df_dichte]

df_merged = df_hilfen.copy()
for df_other in dfs_to_merge:
    df_merged = df_merged.merge(df_other, on="Name", how="left")

print(len(df_merged))

# relative Werte berechnen
df_merged["35a Hilfen pro 10000"] = (df_merged["Anzahl 35a Hilfen"] / df_merged["Bevölkerung 6 bis 20"] * 10000).round(0).astype(int)
df_merged["erz. Hilfen pro 10000"] = (df_merged["Insgesamt"] / df_merged["Bevölkerung 6 bis 20"] * 10000).round(0).astype(int)
df_merged["SGB II-Quote"] = (df_merged["SGB II-Bezug"] / df_merged["Gesamtbevölkerung"]*100).round(2)
df_merged["Anteil gesamt"] = df_merged["Anteil gesamt"].round(1)

# Essen imputieren
RUHR = ["Bochum", "Dortmund", "Duisburg", "Mülheim an der Ruhr", "Oberhausen", "Gelsenkirchen"]
subset = df_merged.loc[df_merged["Name"].isin(RUHR), "35a Hilfen pro 10000"]

mean = subset.apply(pd.to_numeric, errors="coerce").mean()

df_merged.loc[df_merged["Name"] == "Essen", "35a Hilfen pro 10000"] = round(mean, 0)

# Spaltennamen anpassen
df_merged=df_merged.rename(columns={"Anteil gesamt": "Anteil Kinder a.d. Gesamtbev.",
                     "Insgesamt": "erz. Hilfen absolut",
                     "Anzahl 35a Hilfen": "35a Hilfen absolut"})
df_merged = df_merged[["Name",
                        "Typ 1",
                        "Typ 2",
                        "ROR",
                        "erz. Hilfen pro 10000",
                        "35a Hilfen pro 10000",
                        "Überörtlicher Träger",
                        "Bevölkerung 6 bis 20",
                        "Anteil Kinder a.d. Gesamtbev.",
                        "SGB II-Quote",
                        "Kinderarztdichte",
                        "KJP-Dichte",
                        "Abiturquote",
                        "Bevölkerungsdichte",
                        ]]


typ_order = ["Kreisfreie Stadt", "Großer Kreis", "Städteregion", "Kreis"]
df_merged["Typ"] = pd.Categorical(df_merged["Typ 1"], categories=typ_order, ordered=True)
df_merged = df_merged.sort_values(by=["Typ 1", "Name"], ascending=[True, True])

numerical_cols = ['erz. Hilfen pro 10000','35a Hilfen pro 10000','Bevölkerung 6 bis 20', 'Anteil Kinder a.d. Gesamtbev.','SGB II-Quote', 'Kinderarztdichte',  'Bevölkerungsdichte', 'Abiturquote']

validate_df(df_merged,
            not_null = numerical_cols,
            positive= numerical_cols,
            numeric= numerical_cols,
            bounds={'Anteil Kinder a.d. Gesamtbev.':(0,100),'SGB II-Quote':(0,100), 'Kinderarztdichte':(0,100), "KJP-Dichte":(0,100), 'Abiturquote': (0,1)},
            key_cols=["Name"],
            df_name="Masterframe")

df_merged.to_csv("data/processed/master_2024.csv", index=False)