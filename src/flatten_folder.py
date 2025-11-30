"""
Hintergrund

Die von NABU bereitgestellten Rohdaten liegen in einer start
verschachtelten und inkonstruierten Ordnerstruktur vor.
Diese tiefe Schachtelung (Ordner in Ordner) erschwert den direkten Zugriff
und das Training von Mechine-Learning-Modellen.
"""

import os
import shutil
import uuid

source_folder ='/home/surkgoun/nabu_urdaten/nabu/Upload_Praedatoren/Dachs' # unstrukturierte Rohdaten
taget_folder = '/home/surkgoun/nabu-project/nabu_urdaten/dachs' # bereinigte, flache Ordner

def flatten_folder():
    # 1. Überprüfen, ob der source_folder existiert
    if not os.path.exists(source_folder):
        print(f"Fehler: Der Ordner '{source_folder}' existiert nicht.")
        return

    # 2. Zielordner prüfen
    if not os.path.exists(taget_folder):
        print(f"Fehler: Der Ordner '{taget_folder}' existiert nicht.")
        return

    print(f" Start: Kopiervorgang von '{source_folder}' nach '{taget_folder}'")

    #3. Rekursive Suche in allen Unterordnern
    count = 0 # Zähler für erfolgreich kopierte Dateien initialisieren

    # os.walk: Durchsucht das source-Verzeichnis und alle Unterordner rekursiv
    for root, dirs, files in os.walk(source_folder):
        # Iteration über alle Dateien im aktuellen Ordner
        for file in files:
            # Überprüfen, ob die Datei ein gültiges Bildformat hat
            # Groß-/Kleinschreibung wird ignoriert
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Dateinamen und Erweiterung trennen
                name, ext = os.path.splitext(file)
                # Eine eindeutige 6-stellige UUID generieren
                # um Namenskonflikte zu vermeiden
                unique_id = uuid.uuid4().hex[:6]
                # Neuen Dateinamen erstellen
                # Dateinamen_UUID.Erweiterung
                new_name = f"{name}_{unique_id}{ext}"

                # Pfade definieren
                source_path = os.path.join(root, file)
                target_path = os.path.join(taget_folder, new_name)

                try:
                    # Datei kopieren
                    # Metadaten (z.B, Zeitstempel) beibehalten
                    shutil.copy2(source_path, target_path)
                    count += 1 # Zähler erhöhen
                except Exception as e:
                    print(f"Fehler beim Kopieren von '{source_path}' nach '{target_path}': {e}")


    print(f" Fertig: {count} Dateien kopiert.")

if __name__ == "__main__":
    flatten_folder()