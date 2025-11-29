# Code zum Zuschneiden der Metadaten (Bilder)

import os
import cv2

# 1. Pfad zum Ordner mit den Originalbildern (Raw Data)
input_folder = '/home/surkgoun/nabu_urdaten/nabu/Upload_Praedatoren/Sturmmöwe/Graswarder_2025_T533_Srp/'

# 2. Pfad zum Speicherordner für die bearbeiteten Bilder
output_folder = '/home/surkgoun/nabu_bereinigt/sturmmoewe'

# 3. Anteil des Beschnitts am unteren Rand (crop Percentage)
# 0.12 bedeutet, dass die unteren 12% des Bildes entfernt werden.
crop_percentage = 0.12

# 4. Liste aller Bilddateien im Verzeichnis abrufen
image_files = []
for file in os.listdir(input_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_files.append(file)

# 5.
print(f"Starte die Verarbeitung von {len(image_files)} Bildern")
print(f"Die unteren {crop_percentage*100}% werden entfernt.")

# 6
count = 0

for filename in image_files:
    # 1) Pfade generieren
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 2) Bild einlesen (OpenCV liest Bilder als NumPy-Arrays)
    img = cv2.imread(input_path) # Zurückgeben: Höhe, Breite, 3 (Blue, Green, Red) oder None

    if img is None:
        print(f"Fehler: Datei {filename} konnte nicht gelesen werden.")
        continue

    # 3) Bilddimensionen abrufen (Höhe, Breite)
    height, width, _ =img.shape

    # 4) Neue Höhe berechnen (Originalhöhe - abzuschneidender Bereich)
    new_height = int(height * (1 - crop_percentage))

    # 5) Das Zuschneiden
    # Zur neuen Höhe, Breite
    cropped_img = img[0:new_height, 0:width]

    # 6) Bearbeitetes Bild speichern
    cv2.imwrite(output_path, cropped_img)
    count += 1

    # 7) Fortschrittsanzeige
    if count % 10 == 0: # Wenn Bilder 10 sind
        print(f"Verarbeitete Bilder: {count}")

print("_" * 30)
print(f"Fertig. Insgesamt {count} Bilder wurden im Ordner '{output_folder}' gespeichert.")
