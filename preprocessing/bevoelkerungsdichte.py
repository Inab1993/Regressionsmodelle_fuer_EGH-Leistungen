import pandas as pd
from utils.preprocessing_utils import clean_and_sort, preprocess

"""Quelle: Information und Technik Nordrhein-Westfalen (IT.NRW)(2024c). Katasterfläche (qkm), Be-
völkerung und Bevölkerungsdichte - Stichtag - Gemeinden. Landesdatenbank NRW, abgerufen am 28.01.2026. 
URL:https://www.landesdatenbank.nrw.de/ldbnrw//online?operation=table&code=12411-15i"""
# EW pro qkm
df = pd.read_csv("data/raw/2024/12411-15i.csv", sep=";", encoding="latin1", skiprows=3)


df = df[['Unnamed: 2', 'Bevölkerungsdichte']]
df = df.rename(columns={"Unnamed: 2": "Name"})

type_dict = {"Bevölkerungsdichte": "float",}


df = preprocess(df, type_dict)


df = clean_and_sort(df,"Bevölkerungsdichte")


df.to_csv("data/processed/bevoelkerungsdichte_2024.csv", index=False)