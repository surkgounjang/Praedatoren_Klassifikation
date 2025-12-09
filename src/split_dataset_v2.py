"""
Geburt: 09.12.2025

Beschreibung:
Dieses Skript teilt den Datensatz in
- Train (80%),
- Test (10%),
- Validation (10%)

Änderungen gemäß Feedback:
1. Verwendung von 'argparse' für flexible Pfadeingaben.
2.'os.makedirs(exist_ok=True)' ersetzt manuelle Existenzprüfung.

Verwendung (Terminal)
python split_dataset_v2.py --input nabu_regrouped --output nabu_split
"""

import os
import shutil
import random
import argparse
import sys

# Standard-Einstellung
DEFAULT_INPUT = 'nabu_regrouped'
DEFAULT_OUTPUT = 'nabu_split'

TRAIN_RATIO = 0.8 # Train (80%)
TEST_RATIO = 0.1 # Test (10%)
VAL_RATIO = 0.1 # Validation (10%)

def split_dataset(input_folder, output_folder):
    # Eingabeordner prüfen
    if not os.path.exists(input_folder):
        print(f"Fehler: Eingabeordner '{input_folder}' nicht gefunden.")
        sys.exit(1)

    # Klassen (Tierarten) auflisten
    classes = []
    for file in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, file)):
            classes.append(file)
    classes.sort() # Alphabetisch sortieren

    print(f"Starte Datensplit (8:1:1) für {len(classes)} Tierarten.")
    print(f" Quelle: {input_folder} -> Ziel: {output_folder}")
    print("_" * 60)

    for class_name in classes:
        src_dir = os.path.join(input_folder, class_name)

        # Nur Bilddateien sammeln
        images = []
        for image in os.listdir(src_dir):
            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(image)

        random.shuffle(images) # Zufällig mischen
        count = len(images)

        # Generalisierte Berechnung der Split-Größe nach Ratio
        n_test = int(count * TEST_RATIO)
        n_val = int(count * VAL_RATIO)
        n_train = count - n_test - n_val

        # Zuweisung
        split_map = {
            "train" : images[0 : n_train],
            "test" : images[n_train : n_train + n_test],
            "validation" : images[n_train + n_test :]
        }

        print(f"{class_name}: {count}: Bilder -> Train={n_train}, Test={n_test}, Validation={n_val}")

        # Kopieren
        for split_type, file_list in split_map.items():
            dest_dir = os.path.join(output_folder, split_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for file in file_list:
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    print("-" * 60)
    print(f"Fertig. Bitte prüfen Sie kleine Klassen (z.B. Hermelin) manuell")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    split_dataset(args.input, args.output)



