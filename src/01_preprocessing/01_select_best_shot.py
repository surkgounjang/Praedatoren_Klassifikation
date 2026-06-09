# Erstellt: 00.00.2025
# Auswahl des besten Bildes aus Serienaufnahmen (Keyframe Extraction)
# Update: 00.06.2026

import os
import shutil

import cv2
from PIL import Image, UnidentifiedImageError
from datetime import datetime

from pathlib import Path
from operator import itemgetter

# --- Pfad-Konfiguration ---
SRC_PATH = Path(os.getcwd())
PROJECT_PATH = SRC_PATH.parent.parent
DEFAULT_INPUT = PROJECT_PATH / 'data' / 'raw'
DEFAULT_OUTPUT = PROJECT_PATH / 'data' / 'processed'

# BURST_GAP_SECONDS:
# Zeitintervall für Serienaufnahmen in Sekunden.
# Wenn der Zeitstand zwischen Bildern kleiner als dieser Wert ist,
# werden sie als eine zusammengehörige Serie betrachtet.
burst_gap_seconds = 1.0

# ==========
# Liest das Aufnahmedatum aus den Exif-Metadaten des Bildes aus.
# exif (Exchangeable Image File Format): Bildinfos wie Datum, GPS, usw.
# ==========
def get_date(img_path):
    try:
        # 'with' sorgt dafür, dass die Bilddatei nach dem Lesen automatisch geschlossen wird.
        with Image.open(img_path) as img:
            # _getexif() wird verwendet statt getexit(),
            # um die versteckten Unterordner (ExifOffset) der Metadaten zu durchsuchen.
            exif_data = img._getexif()

            # Prüfen, ob EXIF-Daten im Bild vorhanden sind
            if exif_data:
                # 36867: ID für DateTimeOriginal
                # 306: ID für DateTime (Änderungsdatum als Backup)
                time_str = exif_data.get(36867) or exif_data.get(306)

                # Wenn ein Datum gefunden wurde.
                if time_str:
                    return time_str

    # Fängt Fehler ab, falls die Datei beschädigt ist,
    # nicht existiert oder kein gültiges Bildformat aufweist.
    except (OSError, UnidentifiedImageError):
        return None

    # Falls das Bild erfolgreich geöffnet wurde,
    # aber weder 36867 noch 306 existieren.
    return None

# ==========
# Berechnet die Schärfe des Bildes mithilfe der Laplace-Varianz.
# Ein höherer Wert bedeutet ein schärferes Bild
# ==========
def get_sharpness(img_path):
    try:
        # 1. Bild direkt in Grundstufen einlesen
        # str(img_path): stellt sicher, dass pathlib Path-Objekte verarbeitet werden.
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Prüfen, ob das Bild erfolgreich geladen wurde.
        if img is None:
            return 0.0

        # 2. Laplace-Varianz berechnen.
        # Maß für die Kantenstärke bzw. Schärfe.
        # Der Rückgabewert der Laplace-Varianz (.var()) ist ein Fließkommazahl (float).
        return cv2.Laplacian(img, cv2.CV_64F).var()

    except cv2.error:
        return 0.0


# ==========
# Wählt das schärfste Bild aus jeder Serienaufnahme (Burst) aus.
def select_best_shots(input_folder, output_folder, burst_gap_seconds):

    # Prüfen, ob der Eingabeordner existiert.
    if not input_folder.exists():
        raise FileNotFoundError(f"Der Eingabeordner wurde nicht gefunden: {input_folder}")

    # Prüfen, ob der Ausgabeordner existiert.
    if not output_folder.exists():
        print(f"Fehler: Der Ausgabeordner wurde nicht gefunden: {output_folder}")
        # Falls der Ausgabeordner nicht existiert, wird er angelegt.
        output_folder.mkdir(parents=True, exist_ok=True)
        print("Der Ausgabeordner wird neu erstellt.")
        print("\n")

    print(f"Zeitfenster für Serienaufnahmen: {burst_gap_seconds} Sekunden.")
    print("\n")

    # ==========
    # Alle Bilddateien auflisten
    # rglob: Recursive Glob
    image_files = [
        str(file) for file in input_folder.rglob('*')
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Starte die Analyse von {len(image_files)} Bildern zur Auswahl der besten Aufnahmen.")
    print("\n")

    # ==========
    # Liste für Dateiinformationen erstellen
    file_data = []

    for file_path in image_files:
        # Liest das Aufnahmedatum aus den Exif-Daten aus
        # und speichert es als String.
        time_str = get_date(file_path)

        if time_str:
            # z.B. datetime.strptime(2015, 5, 1, 1, 37, 8)
            dt = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            file_data.append(
                {
                    'filename': Path(file_path).name,
                    'filepath': file_path,
                    'timestamp': dt
                }
            )

    # ==========
    # Chronologische Sortierung der Dateien.
    # Sortiert die Daten aufsteigend nach dem Zeitstempel.
    file_data.sort(key=itemgetter('timestamp'))

    # ==========
    # Start der Gruppierung
    # ==========
    print("Verarbeitung läuft...")
    # Initialisiert die erste Bildserie
    current_serie = [file_data[0]]
    # Anzahl der ausgewählten besten Bilder
    selected_count = 0

    # Ab dem zweiten Bild vergleichen
    for i in range(1, len(file_data)):
        prev_file = file_data[i - 1]
        current_file = file_data[i]

        # ==========
        # Zeitdifferenz zwischen den beiden Bildern (prev_file und current_file) berechnen
        time_diff = (current_file['timestamp']-prev_file['timestamp']).total_seconds()

        if time_diff <= burst_gap_seconds:
            # Aktuelles Bild (current_file) zur current_series hinzufügen
            current_serie.append(current_file)
        else:
            # Das beste Bild aus der vorherigen Gruppe wählen.
            best_img = None
            # Der Schärfewert (Laplace-Varianz) ist immer positiv (>=0).
            # Da -1.0 kleiner ist als jeder mögliche Schärfewert,
            # gewinnt das erste Bild sofort und als aktuelles Maximum gespeichert.
            max_sharpness = -1.0

            # ==========
            # Schärfe vergleichen
            for i in current_serie:
                sharpness = get_sharpness(i['filepath'])
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_img = i

            if best_img:
                target_path = output_folder / best_img['filename']
                shutil.copy2(best_img['filepath'], target_path)
                selected_count += 1

            # ==========
            # Neue Gruppe starten
            current_serie = [current_file]

    # Verarbeitung der letzten Gruppe nach dem Ende der Schleife.
    # Da es nach dem letzten Bild kein nächstes Bild gibt.
    print("Verarbeitung der letzten Gruppe nach dem Ende der Schleife...")
    if current_serie:
        best_img = None
        max_sharpness = -1.0
        for i in current_serie:
            sharpness = get_sharpness(i['filepath'])
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_img = i

        if best_img:
            target_path = output_folder / best_img['filename']
            shutil.copy2(best_img['filepath'], target_path)
            selected_count += 1

    print("-" * 30)
    print("Fertig.")
    print(f"Ursprüngliche Anzahl: {len(image_files)} Bilder.")
    print(f"Beste Bilder: {selected_count} Bilder.")
    print(f"Die Bilder wurden im Ordner '{output_folder}' gespeichert.")
    print("\n")


if __name__ == "__main__":
    # Alle Artenverzeichnisse durchlaufen
    species_list = [d.name for d in DEFAULT_INPUT.iterdir() if d.is_dir()]

    # Prüfen, ob species_list existiert.
    if not species_list:
        raise FileNotFoundError(f"Abbruch: Keine Tierarten-Ordner gefunden in '{DEFAULT_INPUT}'")
    else:
        print("-"*30)
        print("Gefundene Tierarten:")
        for i, name in enumerate(species_list):
            print(f"{i+1}. {name}")
        print("-"*30)

    for species in species_list:
        input_folder = DEFAULT_INPUT / species
        output_folder = DEFAULT_OUTPUT / species
        select_best_shots(input_folder, output_folder, burst_gap_seconds)

    print("\n" + "=" * 40)
    print("Alle Verarbeitungen abgeschlossen.")