# Hinweis: Dieses Skript wurde für persönliche Lernzwecke erstellt.

"""
Phase 1: Sanity Check (Initialisierung & Datenladung)

Beschreibung:
Dieses Skript führt einen Sanity Check durch,
um zu prüfen, ob Modell (final_nabu_resnet_20251216.keras) und Daten
korrekt geladen werden können.
"""

import os
import tensorflow as tf
from tensorflow import keras # Importiere Keras für High-Level-API
import numpy as np # Für Array-Operationen im shuffle-Check

# ===== Einstellung =====
DATA_DIR = '/home/surkgoun/nabu-project/nabu_split/'

# Die Bildgröße, die das Modell erwartet (224x224 Pixel)
IMG_SIZE = (224, 224)

# Batch-Größe fpr das Laden der Daten (32 Bilder pro Schritt)
BATCH_SIZE = 32

# Pfad zur gespeicherten Modelldatei (.keras Format)
# Der Dateiname muss exakt stimmen
MODEL_PATH = '/home/surkgoun/nabu-project/model/final_nabu_resnet_20251216.keras'
# ===== Einstellung ist abgeschlossen =====

# ===== Funktion =====
def sanity_check():
    # Ausgabe einer Kopfzeile für die Konsole
    print("\n" + "=" * 60)
    print("[Phase 1] Sanity Check: Testumgebung wird geprüft...")
    print("=" * 60)

    # ----------
    # Step 1:
    # Modell-Verfügbarkeit: Prüfen, ob die Modelldatei existiert und zugreifbar ist
    # ----------
    if not os.path.exists(MODEL_PATH):
        # Fehlermeldung: Modelldatei nicht gefunden
        print(f"Fehler: Modelldatei '{MODEL_PATH}' nicht gefunden.")
        print("Bitte prüfen Sie, ob der Ordner existiert und der Dateiname stimmt.")
        return # Beende das Skript sofort

    # ----------
    # Step 2:
    # Lade-Integrität: Das Keras-Modell laden, um sicherzustellen, dass die Datei nicht beschädigt ist
    # __________
    try:
        print(f"Prüfe Modell-Integrität... ({MODEL_PATH})")
        # Lade das Modell mit Keras
        model =tf.keras.models.load_model(MODEL_PATH)
        print("Sanity Check: Modell erfolgreich geladen.") # Erwarteter Output

        # Überprüfe die erwartete Eingabeform (Input Shape) des Modells
        input_shape = model.input_shape
        print(f" - Input Shape: {input_shape} (Erwartet: (None,  224, 224, 3))")  # None: Batch Size

    except Exception as e:
        # Fehlermeldung: Modell ist beschädigt oder inkompatibel
        print("FAIL: Modell ist beschädigt oder kann nicht geladen werden.")
        print(f"Fehler: {e}")
        return # Beende das Skript sofort

    # ----------
    # Step 3:
    # Daten-Pfad-Check: Prüfen, ob der Validierungsordner korrekt verknüpft ist
    # ----------
    # Konstruiere den Pfad zum 'validation'-Ordner
    val_dir = os.path.join(DATA_DIR, 'validation')

    if not os.path.exists(val_dir):
        # Fehlermeldung: Validierungsordner nicht gefunden
        print(f"Fehler: Validierungsordner '{val_dir}' nicht gefunden.")
        print("Bitte prüfen Sie, ob der Ordner existiert.")
        return # Beende das Skript sofort

    # ----------
    # Step 4:
    # Daten-Konsistenz: Den Validierungsdatensatz laden
    # ----------
    print(f"Lade Validierungsdatensatz... ({val_dir})")

    try:
        # Erstelle das Dataset aus dem Verzeichnis
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,                    # Pfad zum Validierungsordner
            image_size = IMG_SIZE,      # Bildgröße (224x224 Pixel) ohne 3 RGB
            batch_size = BATCH_SIZE,    # Batch-Größe
            shuffle = False             # Nicht mischen
        )
    except Exception as e:
        # Fehlermeldung: Problem beim der Bilder
        print(f"Fehler: Problem beim Laden des Validierungssets: {e}")
        return

    # ----------
    # Step 5:
    # Überprüfung der Daten-Reihenfolge (Shuffle-Check)
    # ----------
    # Wenn shuffle = False ist, gleiche Zahlen (z.B. 0. 0. 1, 1, 2, 2, 2, 2...)
    print("Prüfe shuffle = False...")
    for images, labels in val_ds.take(1):
        lbls = labels.numpy()
        print(f" - Labels im ersten Batch: {lbls}")

        # Wenn Labels aufsteigen oder gleich bleiben,
        # ist es nicht gemischt
        is_sorted = np.all(lbls[:-1] <= lbls[1:])
        if is_sorted:
            print(" Pass: Nicht gemischt")
        else:
            print("FAIL: Gemischt")

    # ----------
    # Step 6:
    # Klassen-Plausibilität: Prüfen, on exakt 17 Tierarten gefunden werden und die Namen korrekt sind
    # ----------
    # Extrahiere die gefundenen Klassennamen aus dem Dataset val_ds
    class_names = val_ds.class_names
    num_classes = len(class_names)

    print("Dataset val_ds erfolgreich geladen.")
    print(f"Sanity Check: {num_classes} Tierarten verifiziert: {class_names}") # Erwarteter Output

    # ----------
    # Abschluss
    # ----------
    if num_classes == 17:
        print("Phase 1 (Sanity Check) bestanden - Bereit für Analyse") # Erwarteter Output
    else:
        print(f"[Sanity Check] fehlgeschlagen. Es wurden {num_classes} Klassen gefunden. Es sollte 17 Klassen geben.")
        print("Bitte Datenordner überprüfen. (Fehlen Unterordner?)")

if __name__ == "__main__":
    sanity_check()