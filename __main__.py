import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":

# Achtung, der Download der Tabelle 12521-03i muss manuell für den Stichtag 2024 selektiert werden!
# Hinweis: manuelle Auswahl des Codes "KREIS" im Tabellenaufbau des Bevölkerungsstandes erforderlich

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "preprocessing/hilfen.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/raumordnungsregionen.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/arztdichte.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/traeger.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bevoelkerungsstand.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/sgb_ii.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bevoelkerungsdichte.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/auslaender.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bildung.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/merge.py"], check=False)

    subprocess.run([sys.executable, "preprocessing/sgb9_hilfen.py"], check=False)

    subprocess.run([sys.executable, "analysis/descriptive/descriptive_analysis_hilfen.py"], check=False)
    subprocess.run([sys.executable, "analysis/descriptive/descriptive_analysis_vars.py"], check=False)
    subprocess.run([sys.executable, "analysis/descriptive/descriptive_analysis_on_ROR_aggregate.py"], check=False)