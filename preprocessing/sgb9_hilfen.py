from utils.preprocessing_utils import clean_and_sort, preprocess
import pandas as pd

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW) (2024c). Empfänger/-innen von Eingliederungshilfe nach Alter (6),
Leistungsart (5) und Träger (2) - Region - Stichtag
(Trägerprinzip). Landesdatenbank NRW, abgerufen am
16.02.2026. URL:https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=22161-08i"""

df = pd.read_csv("data/raw/2024/22161-08i.csv", sep=";", encoding="latin1", skiprows=6)


df = df[df["Unnamed: 0"] == "31.12.2024"]
df = df[df["Unnamed: 1"].isin(["Rheinland", "Westfalen-Lippe"])]
df = df[df["Unnamed: 3"] == "Insgesamt"]

df = df[['Unnamed: 1','Unnamed: 2','Unnamed: 3', 'unter 7 Jahre', '7 bis unter 18 Jahre']]
df = df.rename(columns={'Unnamed: 2': 'Name'})

type_dict = {'unter 7 Jahre': "int", '7 bis unter 18 Jahre': "int"}


df = preprocess(df, type_dict, name_filter=r"Kreis|krfr\. Stadt|Landschaftsverband", drop_aachen=False)



df_traeger = pd.read_csv("data/processed/traeger.csv", sep=",", encoding="UTF-8")
df_bevoelkerung = pd.read_csv("data/processed/bevoelkerung_2024.csv", sep=",", encoding="UTF-8")

base = df_traeger[["Name", "Überörtlicher Träger"]].merge(df_bevoelkerung[["Name", "Bevölkerung u6","Bevölkerung 6 bis 18"]], on="Name", how="inner")

df_traeger_aggregate = (
    base.groupby("Überörtlicher Träger")
        .agg(
            first_sum=("Bevölkerung u6", "sum"),
            sec_sum=("Bevölkerung 6 bis 18", "sum")
        )
)


data = [
    ["Sonstige EGH im Rheinland 7 bis 18",
     ((df.iloc[0][["unter 7 Jahre", "7 bis unter 18 Jahre"]].sum()+ df.iloc[1]["7 bis unter 18 Jahre"])
     /df_traeger_aggregate.loc["LVR", "sec_sum"]*10000).round(0)],

    ["LVR u7",
     ((df.iloc[1]["unter 7 Jahre"]/df_traeger_aggregate.loc["LVR", "first_sum"])*10000).round(2)],

    ["Sonstige EGH in LWL 7 bis 18",
     ((df.iloc[2][["unter 7 Jahre", "7 bis unter 18 Jahre"]].sum()+ df.iloc[3]["7 bis unter 18 Jahre"])
     / df_traeger_aggregate.loc["LWL", "sec_sum"] * 10000).round(0)],

    ["LWL u7",
    ((df.iloc[3]["unter 7 Jahre"] / df_traeger_aggregate.loc["LVR", "first_sum"]) * 10000).round(2)]
]


result = pd.DataFrame(data, columns=["Name", "Hilfen auf 10000 Kinder"]).set_index("Name")
result.to_csv("data/processed/sgb9_hilfen_2024.csv")