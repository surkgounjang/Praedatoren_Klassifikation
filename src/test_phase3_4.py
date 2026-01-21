import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# =====
# Konfiguration
# =====

# Definiert den Pfad zur trainierten Modelldatei (.keras Format)
MODEL_PATH = '/home/surkgoun/nabu-project/model/20260112/final_nabu_resnet_20260112(1).keras'

# Definiert den Pfad zum Ordner mit den Validierung
VALIDATION_DIR = '/home/surkgoun/nabu-project/5_nabu_split_/nabu_split_2/validation'

# Legt die Höhe der Bilder fest, die das Modell erwartet (224 Pixel)
IMG_HEIGHT = 224
# Legt die Breite der Bilder fest, die das Modell erwartet (224 Pixel)
IMG_WIDTH = 224
# Bestimmt, wie viele Bilder gleichzeitig verarbeitet werden
BATCH_SIZE = 64

# =====
# Funktion für Phase 3: Visuelle Prüfung
# =====

def run_phase_3_visual_check(model, dataset, class_names):
    """
    Führt Phase 3 durch:
    Zeigt 16 zufällige Bilder und deren Vorhersagen an.
    """

    print("\n" + "-" * 60)
    print("Phase 3: Visuelle Prüfung")
    print("-" * 60)

    # Nimmt den ersten Batch (64 Bilder) aus dem Datensatz
    images, labels = next(iter(dataset))

    # Lässt das Modell Vorhersagen treffen.
    # verbose=0 bedeutet, dass keine Ausgabe im Terminal erfolgt
    predictions = model.predict(images, verbose = 0)

    # Ermittelt den Index der höchsten Wahrscheinlichkeit für die Vorhersage
    # argmax
    # Wenn es 16 Beispiele gibt,
    # ist pred_ids ein 1D-Array der Länge 16.
    # Wobei jedes Element enthält den vorhergesagten Klassenindex für das jeweilige Bild.
    # Beispiel: [2, 5, 0, 9, 1, ...]
    pred_ids = np.argmax(predictions, axis = 1) # axis = 1: row

    # Ermittelt den Index der wahren Klasse aus den Labels
    true_ids = np.argmax(labels, axis = 1) # z.B. [2, 5, 0, 9, 1, ...]


    # =====
    # Erstellt eine Grafik-Figur mit der Größe 16 x 16 Zoll
    plt.figure(figsize=(20, 20))

    # Setzt den Titel über die Grafik
    plt.suptitle("Phase 3: Visuelle Prüfung (Zufallsstichprobe)", fontsize=16, fontweight='bold')

    # Bestimmt, wie viele Bilder angezeigt werden sollen (maximal 16)
    num_images = min(16, len(images))

    # Schleife durch die 16 Bilder
    for i in range(num_images):
        # Erstellt ein subplot im 4x4-Raster an Position i+1
        # row: 4
        # column: 4
        ax = plt.subplot(4, 4, i + 1)

        # Konvertiert das Bilder für die Anzeige
        # Matplotlib erwartet int (0-255).
        # Da tensorflow flot32 liefert,
        # wandele ich es mit .astype("uint8") in ganze Zahlen um.
        img_display = images[i].numpy().astype("uint8")

        # Zeigt das Bild im subplot an
        plt.imshow(img_display)

        # Holt den vorhergesagten Index für das aktuelle Bild
        pred_idx = pred_ids[i]

        # Holt den wahren Index für das aktuelle Bild
        true_idx = true_ids[i]

        # Holt die Wahrscheinlichkeit der Vorhersage und rechnet sie in Prozent um
        confidence = 100 * np.max(predictions[i])

        # Holt den Nahmen der vorhergesagten Klasse aus der Liste
        pred_label = class_names[pred_idx]

        # Holt den NAmen der vorhergesagten Klasse aus der Liste
        true_label = class_names[true_idx]

        # Prüf, ob die Vorhersage korrekt ist
        if pred_idx == true_idx:
            color = 'green' # Falls die Vorhersage richtig ist
            title_text = f"Pred: {pred_label} ({confidence:.1f}%) \n True: {true_label}"
        else:
            color = 'red' # Falls die Vorhersage falsch ist
            title_text = f"Pred: {pred_label} ({confidence:.1f}%) \n True: {true_label}"

        # Setzt den Titel über das Einzelbild mit der gewählten Farbe
        plt.title(title_text, color = color, fontsize = 11, fontweight = 'bold')

        # Schaltet die x, y Koordinaten aus
        plt.axis("off")

    # Optimiert die Abstände zwischen den Bildern
    plt.tight_layout()

    # Speichert das 4x4 Raster als Bilddatei (.png)
    plt.savefig("phase3_visual_result.png", dpi = 300)

    # Gibt eine Bestätigung aus, dass das Bild gespeichert wurde
    print("Ergebnisbild gespeichert: phase3_visual_result.png")

# =====
# Funktion für Phase 4: Statistische Bobustheit
# =====

def run_phase_4_statik_check(model, dataset):
    """
    Führt Phase 4 durch:
    Berechnet Top-1 und Top-3 Genauigkeit über alle Daten
    """

    print("\n" + "-" * 60)
    print("Phase 4: Statistische Robustheit")
    print("-" * 60)

    # Initialisiert die Metrik für Top-1
    top1_acc = tf.keras.metrics.CategoricalAccuracy(name = 'top1')

    # Initialisiert die Metrik für Top-3
    top3_acc = tf.keras.metrics.TopKCategoricalAccuracy(k = 3, name = 'top3')

    # Gibt eine Nachricht aus
    print("Berechne Top-1 und Top-3 über alle Daten...")

    # Schleife über alle Batches im Validierungsdatensatz
    for images, labels in dataset:
        # Lässt das Modell Vorhersagen treffen.
        # verbose=0 bedeutet, dass keine Ausgabe im Terminal erfolgt
        preds = model.predict(images, verbose = 0)

        # Aktualisiert die Top-1 Metrik mir den Ergebnissen dieses Batches
        top1_acc.update_state(labels, preds)

        # Aktualisiert die Top-3 Metrik
        top3_acc.update_state(labels, preds)

    # Holt das Ergebnis der Top-1
    res_top1 = top1_acc.result().numpy() * 100

    # Holt das Ergebnis der Top-3
    res_top3 = top3_acc.result().numpy() * 100

    # Berechnet die Differenz zwischen Top-3 und Top-1
    diff = res_top3 - res_top1

    report = []
    report.append("\n [Ergebnis]" + "-" * 60)
    report.append(f" Top-1 Accuracy: {res_top1:.2f}%")
    report.append(f" Top-3 Accuracy: {res_top3:.2f}%")
    report.append(f" Differenz:      +{diff:.2f}%")

    with open("phase4_result.txt", "w", encoding = 'utf-8') as f:
        f.write("\n".join(report))
    print("report gespeichert: phase4_result.txt")

def main():
    print("\n" + "=" * 60)
    # Gint den Titel des Skripts aus
    print("Start der Modell-Validierung (Phase 3 & 4)")
    print("=" * 60 + "\n")

    # --- Schritt 1: Modell laden ---

    # Prüfen, ob die Modelldatei existiert
    if not os.path.exists(MODEL_PATH):
        # Falls nicht
        print(f"Fehler: Modelldatei nicht gefunden: {MODEL_PATH}")
        return # oder sys.exit(1)

    # Gibt eine Info-Nachricht aus, dass das Modell jetzt geladen wird
    print(f"Lade Modell von {MODEL_PATH}")

    # Lädt das Keras-Modell
    model = tf.keras.models.load_model(MODEL_PATH)

    # --- Schritt 2: Validierungsdaten laden ---

    # Prüfen, ob die Validierungsdaten existieren
    if not os.path.exists(VALIDATION_DIR):
        # Falls nicht
        print(f"Fehler: Validierungsordner nicht gefunden: {VALIDATION_DIR}")
        return

    # Gint eine Info-Nachricht aus, dass die Validierungsdaten jetzt geladen werden
    print(f"Lade Validierungsdaten von {VALIDATION_DIR}")
    val_ds = image_dataset_from_directory(
        VALIDATION_DIR,                         # Der Pfad zum Validierungsordner
        image_size = (IMG_HEIGHT, IMG_WIDTH),   # Die Gröpe der Bilder (224x224)
        batch_size = BATCH_SIZE,                # 64 Bilder pro Schritt
        shuffle = True,                         # Wichtig: Mischt die Bilder zufällig
        seed = 42,                              # Setzt einen fasten Zufallswert für Reproduzierbarkeit
        label_mode = 'categorical'              # für softmax nötig
    )

    # Speichert die Namen der Klassen (Ordnernamen) in einer Liste
    class_names = val_ds.class_names

    # Gibt aus, wie viele Klassen gefunden wurden
    print(f"Klassen ({len(class_names)}): {class_names}")

    # --- Schritt 3: Tests ausführen ---
    run_phase_3_visual_check(model, val_ds, class_names)

    # --- Schritt 4: Statistische Robustheit
    run_phase_4_statik_check(model, val_ds)

    print("\n" + "=" * 60)
    print("Phase 3 & 4 abgeschlossen.")
    print("=" * 60)

if __name__ == "__main__":
    main()