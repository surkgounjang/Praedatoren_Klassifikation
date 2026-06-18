"""
Erstellt: 09.12.2025

Beschreibung:
Dieses Skript teilt den Datensatz in
- Train (80%),
- Test (10%),
- Validation (10%)

Update: 17.06.2026
- Dieses Skript teilt den Datensatz dynamisch basierend auf der tatsächlichen Anzahl der Bild in jedem Tierarten-Ordner.
"""

import os
import random
import shutil
from pathlib import Path

# Standard-Einstellung
SRC_PATH = Path(os.getcwd())
PROJECT_PATH = SRC_PATH.parent.parent

INPUT_PATH = PROJECT_PATH / 'data' / 'final'
OUTPUT_PATH = PROJECT_PATH / 'data' / 'split'

def dynamic_split_dataset(input_folder, output_folder):
    # Prüfen, ob der Eingabeordner existiert.
    if not input_folder.exists():
        raise FileNotFoundError(f"Der Eingabeordner wurde nicht gefunden: {input_folder}")

    # Prüfen, ob der Ausgabeordner existiert.from
    if not output_folder.exists():
        print(f"Fehler: Der Ausgabeordner wurde nicht gefunden: {output_folder}")
        # Falls der Ausgabeordner nicht existiert, wird er angelegt.
        output_folder.mkdir(parents=True, exist_ok=True)
        print("Der Ausgabeordner wird neu erstellt.")
        print("\n")

    # sorted: Aufsteigend sortieren
    # Z.B. andere, austernfischer, fuchs usw.
    classes = sorted([d for d in input_folder.iterdir() if d.is_dir()])

    print("-" * 30)
    print(f"Starte Datensplit für {len(classes)} Tierarten")
    print("Die Aufteilung passt sich automatisch der Anzahl der Bilder an.")
    print("-" * 30)

    for class_dir in classes:
        images = [img for img in class_dir.rglob("*") if img.name.lower().endswith((".jpg", ".jpeg", ".png"))]
        count = len(images)

        # Wenn keine Bild vorhanden sind,
        # gehe zum nächsten Ordner (class_dir)
        if count < 3:
            print(f"Überspringen: {class_dir} har nur {count} Bilder (Minimum ist 3).")
            continue

        random.shuffle(images)

        # ==========
        # Automatischer Verteilungsalgorithmus
        # basierend auf der Dateianzahl.
        # Mindestens 3 Bilder.
        # ==========
        if count >= 10:
            n_test = int(count * 0.1)           # test: 10%
            n_val = int(count * 0.1)            # validation: 10%
            n_train = count - n_test - n_val    # train: 80%
        elif count >= 4:
            n_test = 1
            n_val = 1
            n_train = count -2
        else:
            n_test = 1
            n_val = 1
            n_train = 1

        split_map = {
            'train': images[:n_train],
            'test': images[n_train:n_train+n_test],
            'val': images[n_train+n_test:]
        }

        # Statusausgabe
        # ljust(width, fillcahr): Left Justify
        print(f"{class_dir.name.ljust(15)} | Gesamt: {str(count).ljust(4)} | -> Train={n_train} | Test={n_test} | Val={n_val}")

        for split_type, file_list in split_map.items():
            dest_dir = output_folder / split_type / class_dir.name

            # Prüfen, ob dest_dir existiert.
            if not dest_dir.exists():
                print(f"{dest_dir} nicht gefunden.")
                dest_dir.mkdir(parents=True, exist_ok=True)
                print(f"{dest_dir} wird neu erstellt.")

            for file in file_list:
                shutil.copy2(file, dest_dir / file.name)

    print("-" * 30)
    print("Fertig: Alle Tierarten wurden aufgeteilt.")

def main():
    dynamic_split_dataset(INPUT_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()



