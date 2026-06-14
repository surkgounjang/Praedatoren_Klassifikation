# Code zum Zuschneiden der Metadaten (Bilder)
# Update: 14.06.2026

import os
from pathlib import Path

import cv2
import numpy as np

# --- Pfad-Konfiguration ---
# /home/surkgoun/nabu-project/src/01_preprocessing
SRC_PATH = Path(os.getcwd())
# /home/surkgoun/nabu-project
PROJECT_PATH = SRC_PATH.parent.parent

INPUT_DIR = PROJECT_PATH / 'data' / 'processed'
OUTPUT_DIR = PROJECT_PATH / 'data' / 'final'

# in den unteren 12% des Bildes zuschneiden
default_crop_percentage = 0.12

# NEU ==========
# Update: 12.06.2026
# Dynamische Erkennung der Metadaten-Leiste
# return: Y-Koordinaten (start_y, end_y)
# ==========
def dynamic_find_crop_height(img, default_crop_percentage):
    # z.B. img.shape: (1512, 2688, 3)
    height, width = img.shape[:2]

    # Standardwerte: Falls keine Leiste gefunden wird.
    start_y = 0
    end_y = height

    # Suchbereich: 25% der Bildhöhe für oben und unten
    search_area = int(height * 0.25)

    # Schwellenwert:
    # Mindestens 10% der Bildbreite muss eine Linie sein.
    # 'width': Bildbreite.
    # 255: Maximaler Pixelwert.
    # -0: Schwarz/Keine weiße Pixel.
    # - 255: Weißpixel.
    # 0.10: Bestimmt, dass die gefundene Linie mindestens zu 10% der Bildbreite
    # aus weißen Pixeln bestehen muss, um als Trennlinie akzeptiert zu werden.
    threshold = width * 255 * 0.10

    # ==========
    # 1. Oben prüfen
    # ==========

    # img[y_Koordinate_start_y:y_Koordinate_end_y, x_Koordinate_start_x:x_Koordinate_end_x]
    top_region =img[0:search_area,:]
    # In Graustufen umwandeln.
    gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    # Kantenerkennung mit Canny-Edge-Detection
    edges_top = cv2.Canny(gray_top, 50, 150)
    # Weiße Pixelwerte pro Zeile aufsummieren.
    # ros_sum_top ist array
    # axis=1: X-Achse.
    row_sums_top = np.sum(edges_top, axis=1)
    # Die Zeile mit der stärksten horizontalen Kante finden.
    max_edge_row_top = int(np.argmax(row_sums_top))

    if row_sums_top[max_edge_row_top] > threshold:
        # Wenn oben eine Linie gefunden wurde.
        start_y = max_edge_row_top

    # ==========
    # Unten prüfen
    # ==========
    bottom_region = img[height - search_area:, :]
    gry_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    edges_bottom = cv2.Canny(gry_bottom, 50, 150)
    row_sums_bottom = np.sum(edges_bottom, axis=1)
    max_edge_row_bottom = int(np.argmax(row_sums_bottom))

    if row_sums_bottom[max_edge_row_bottom] > threshold:
        end_y = (height - search_area) + max_edge_row_bottom

    # ==========
    # Fallback
    # ==========
    # Wenn werder oben noch unten einen Metadaten-Leiste erkannt wurde,
    # schneide den Standard-Prozentsatz ab.
    if start_y == 0 and end_y == height:
        end_y = int(height * (1 - default_crop_percentage))

    return start_y, end_y

# ==========
# Hauptfunktion: Bilder verarbeiten und speichern.
# ==========
def zuschneiden(input_folder, output_folder, default_crop_percentage):

    # Prüfen, ob der Eingabeordner existiert.
    if not input_folder.exists():
        raise FileNotFoundError(f"Der Eingabeordner wurde nicht gefunden: {input_folder}")

    # Prüfen, ob der Ausgabeordner existiert.
    if not output_folder.exists():
        print(f"Fehler: Der Ausgabeordner wurde nicht gefunden:{output_folder}")
        # Falls der Ausgabeordner nicht existiert, wird er angelegt.
        output_folder.mkdir(parents=True, exist_ok=True)
        print("Der Ausgabeordner wird neu erstellt.")
        print("\n")

    # ==========
    # Alle Bilddateien auflisten.
    # rglob: Recursive Glob
    # ==========
    image_files = [file for file in input_folder.rglob('*')
                   if file.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print("-" * 30)
    print(f"Start die Verarbeitung von {len(image_files)} Bildern.")
    print("Dynamische Erkennung wird angewendet (oben und unten).")
    print("-" * 30)

    count = 0

    for file in image_files:
        img = cv2.imread(str(file))

        # Prüfen, ob das Bild erfolgreich geladen wurde.
        if img is None:
            print(f"Fehler: Datei {file} konnte nicht geladen werden.")
            continue

        width = img.shape[1]

        # NEU (12.06.2026): Dynamische Koordinate berechnen
        start_y, end_y = dynamic_find_crop_height(img, default_crop_percentage)

        # NEU (12.06.2026): Bild mit den neuen Koordinaten zuschneiden
        cropped_img = img[start_y:end_y, 0:width]
        output_path = output_folder / file.name
        cv2.imwrite(str(output_path), cropped_img)
        count += 1

    print("-" * 30)
    print(f"Fertig. {count} Bilder wurden verarbeitet.")

def main():
    # Alle Artenverzeichnisse durchlaufen
    species_list = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]

    # Prüfen, ob species_list existiert.
    if not species_list:
        raise FileNotFoundError(f"Abbruch: Keine Tierarten-Ordner gefunden in '{INPUT_DIR}'")

    print("-"*30)
    print("Gefundene Tierarten:")
    for i, name in enumerate(species_list, start=1):
        print(f"{i}. {name}")
    print("-"*30)

    for species in species_list:
        input_folder = INPUT_DIR / species
        output_folder = OUTPUT_DIR / species
        zuschneiden(input_folder, output_folder, default_crop_percentage)

    print("\n" + "=" * 40)
    print("Alle Verarbeitungen abgeschlossen.")

if __name__ == "__main__":
    main()