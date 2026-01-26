import subprocess
import sys

#def run_single_file(file_path):
#    subprocess.run([sys.executable, file_path], check=False)

if __name__ == "__main__":
#    datei = r"/pfad/zu/deiner/datei.py"  # ← hier anpassen
#    run_single_file(datei)


    subprocess.run([sys.executable, "preprocessing/hilfen.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/raumordnungsregionen.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/arztdichte.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/traeger.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bevoelkerungsstand.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/sgb_ii.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bevoelkerungsdichte.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/bildung.py"], check=False)
    subprocess.run([sys.executable, "preprocessing/merge.py"], check=False)