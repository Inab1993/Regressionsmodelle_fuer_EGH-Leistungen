import pandas as pd
from helper.preprocessing_utils import clean_and_sort

"""Quelle Landschaftsverband Rheinland (LVR) (2026). Gebiet und Mitglieder. Landschaftsver-
band Rheinland (LVR), abgerufen am 28.01.2026. URL: https://lvr.de/de/nav_main/derlvr/organisation/gebietundmitglieder/gebietundmitglieder_1.jsp"""
lvr_kreise = "StädteRegion Aachen | Kreis Düren | Kreis Euskirchen | Kreis Heinsberg | Kreis Kleve | Kreis Mettmann | Kreis Viersen | Kreis Wesel | Oberbergischer Kreis | Rhein-Kreis Neuss | Rheinisch-Bergischer Kreis | Rhein-Erft-Kreis | Rhein-Sieg-Kreis"
lvr_staedte = "Stadt Bonn | Stadt Duisburg | Stadt Düsseldorf | Stadt Essen | Stadt Köln | Stadt Krefeld | Stadt Leverkusen | Stadt Mönchengladbach | Stadt Mülheim an der Ruhr | Stadt Oberhausen | Stadt Remscheid | Stadt Solingen | Stadt Wuppertal"


lvr = pd.DataFrame(
    {"Name": [s.strip() for s in (lvr_staedte + " | " + lvr_kreise).split("|")], "Überörtlicher Träger": "LVR"}
)

lwl = pd.read_csv("data/raw/lwl.txt", header=None, names=["Name"], encoding="UTF-8")
lwl["Überörtlicher Träger"] = "LWL"

df = pd.concat([lvr, lwl], ignore_index=True)

df["Name"] = df["Name"].str.strip()

df = clean_and_sort(df, "Überörtlicher Träger")

df.to_csv("data/processed/traeger.csv", index=False)