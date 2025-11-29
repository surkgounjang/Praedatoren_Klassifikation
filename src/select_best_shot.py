# Auswahl des besten Bildes aus Serienaufnahmen (Keyframe Extraction)

import os
import shutil

import cv2
from PIL import Image
from datetime import datetime

# 1. Pfad zum Ordner mit den Originalbildern
input_folder = '/home/surkgoun/nabu_flattened/fuchs/'

# 2. Pfad zum Speicherordner für die ausgewählten besten Bilder
output_folder = '/home/surkgoun/nabu_keyframes/fuchs/'


# 3. Zeitintervall für Serienaufnahmen (in Sekunden)
# Wenn der Zeitabstand zwischen zwei Bildern kleiner als dieser Wert ist,
# werden sie als ein zusammengehörige Serie betrachtet.
burst_gap_seconds = 1.0 # 10 Sekunden ->> 1 Sekunde

# ==========
# Liest das Aufnahmedatum aus den Exif-Metadaten des Bildes aus.
def get_date(path):
    try:
        return Image.open(path).getexif()[36867]
    except Exception:
        return None


# Berechnet die Schärfe des Bildes mithilfe der Laplace-Varianz
# Ein höherer Wert bedeutet ein schärferes Bild
def get_sharpness(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 0

        # Umwandlung in Graustufen für die Kantenanalyse
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplace-Varianz berechnen
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0

# ==========

# 4. Alle Bilddateien auflisten
image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Starte die Analyse von {len(image_files)} Bildern zur Auswahl der besten Aufnahmen.")

# 5. Liste für Dateiinformationen erstellen
file_data = []

for file in image_files:
    input_path = os.path.join(input_folder, file)
    time_str = get_date(input_path)

    if time_str:
        dt = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        file_data.append({'name': file, 'path': input_path, 'time': dt})
    else:
        timestamp = os.path.getmtime(input_path) # Wenn keine Exif-Daten vorhanden sind
        dt = datetime.fromtimestamp(timestamp)
        file_data.append({'name': file, 'path': input_path, 'time': dt})

# 6. Chronologisch sortieren (Sehr wichtig für die Gruppierung)
file_data.sort(key=lambda x: x['time'])

# 7. Gruppierung und Auswahl
if not file_data:
    print("Keine Bilder im Ordner gefunden.")
    exit()

current_series = [file_data[0]]
selected_count = 0 # Anzahl der ausgewählten besten Bilder

print("Verarbeitung läuft...")

# 8. Ab dem zweiten Bild vergleichen
for i in range(1, len(file_data)):
    prev_img = file_data[i-1]
    cur_img = file_data[i]

    # Zeitdifferenz berechnen (in Sekunden)
    time_diff = (cur_img['time'] - prev_img['time']).total_seconds()

    if time_diff <= burst_gap_seconds:
        current_series.append(cur_img) # Zur aktuellen Gruppen hinzufügen (gehört zur gleichen Serie)
    else: # Die Serie ist beendet. Jetzt, das beste Bild aus der vorherigen Gruppe wählen
        best_img = None
        max_score = -1
        # Der Schärfewert (Laplace-Varianz) ist immer positiv (>=0)
        # Da -1 kleiner ist als jeder mögliche Schärfewert,
        # gewinnt das erste Bild sofrt und als aktuelles Maximum gespeichert.

        # Schärfe vergleichen
        for img_info in current_series:
            score = get_sharpness(img_info['path'])
            if score > max_score:
                max_score = score
                best_img = img_info

        # Das Gewinner-Bild kopieren
        if best_img:
            shutil.copy2(best_img['path'], os.path.join(output_folder, best_img['name']))
            # shutil.copy2: Kopiert das beste Bild in den "output"-Ordner und behält die Metadaten (z.B. Zeitstempel)
            selected_count += 1

        # Neue Gruppe starten
        current_series = [cur_img]

# 9. The Last Piece
# Verarbeitung der letzten Gruppe nach dem Ende der Schleife
# Da es nach dem letzten Bild kein nächstes Bild gibt,
# muss die letzte Gruppe manuell nach der Schleife gespeichert werden.
if current_series:
    best_img = None
    max_score = -1
    for img_info in current_series:
        score = get_sharpness(img_info['path'])
        if score > max_score:
            max_score = score
            best_img = img_info

    if best_img:
        shutil.copy2(best_img['path'], os.path.join(output_folder, best_img['name']))
        selected_count += 1


print("-" * 30)
print("Fertig.")
print(f"Ursprüngliche Anzahl: {len(image_files)} Bilder.")
print(f"Beste Bilder: {selected_count} Bilder.")
print(f"Die Bilder wurden im Ordner '{output_folder}' gespeichert.")