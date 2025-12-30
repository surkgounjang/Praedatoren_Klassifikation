# Hinweis: Dieses Skript wurde für persönliche Lernzwecke erstellt.

"""
Phase 2: Statistische Auswertung (Confusion Matrix)
Beschreibung:
Dieses Skript implementiert die statistische Analyse gemäß dem "Technischen Bericht".
Es generiert die Konfusionsmatirx, berechnet Precision, Recall und F1-Score.
Es hebt schwache Klassen (F1-Score < 0.80) hervor.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# ===== Konfiguration
# Pfad zum Validierungsdatensatz
DATA_DIR = '/home/surkgoun/nabu-project/nabu_split/validation/'
# Pfad zur trainierten Modelldatei
MODEL_PATH = '/home/surkgoun/nabu-project/model/final_nabu_resnet_20251216.keras'
# Eingabegröße der Bilder (224x224 Pixel)
IMG_SIZE = (224, 224)
# Batch-Größe für die Inferenz
BATCH_SIZE = 32
# Grenzwert für den F1-Score (Schwache Klassen werden hervorgehoben)
THRESHOLD_F1 = 0.80
# Ausgabeordner für die Ergebnisse
OUTPUT_DIR = '/home/surkgoun/nabu-project/evaluation/'
# =====

def run_statistical_analysis():
    # Startnachricht ausgeben
    print("="*60)
    print("[Phase 2] Start der statistischen Analyse...")
    print("="*60)

    # ----------
    # Step 1: Initialisierung und Datenladen
    # ----------

    # Überprüfen, ob die Modelldatei existiert
    if not os.path.exists(MODEL_PATH):
        print(f"Fehler: Die Modelldatei nicht gefunden unter {MODEL_PATH}")
        return

    # Laden des trainierten Keras-Modells
    print(f"Lade Modell: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return

    # Überprüfen, ob das Datenverzeichnis existiert
    if not os.path.exists(DATA_DIR):
        print(f"Fehler: Das Datenverzeichnis nicht gefunden unter {DATA_DIR}")
        return

    # Laden des Validierungsdatensatzes
    print(f"Lade Validierungsdatensatz: {DATA_DIR}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

    # Extrahieren der Tierarten aus dem Validierungsdatensatz
    class_names = val_ds.class_names
    print(f"Erkannte Tierarten ({len(class_names)}): {class_names})")

    # ----------
    # Step 2: Inferenz (Vorhersage)
    # ----------
    print("\n Führe Inferenz durch...")

    # Liste für tatsächliche Labels (Ground Truth) erstellen
    y_true = []
    # Liste für vorhergesagte Labels (Predictions) erstellen
    y_pred = []

    # Durchlaufen des gesamten Validierungsdatensatzes (batch für batch)
    for images, labels in val_ds:
        # Tatsächliche Labels sammeln
        y_true.extend(labels.numpy())

        # Vorhersage durch das Modell durchführen
        preds = model.predict(images, verbose = 0)

        # Umwandlung der Wahrscheinlichkeit in Klassen-Indizes (argmax)
        pred_labels =np.argmax(preds, axis=1)
        y_pred.extend(pred_labels)

    # Konvertierung in numpy-Arrays für die Weiterverarbeitung (Berechnung statistischer Kennzahlen)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("Inferenz erfolgreich abgeschlossen.")

    # ----------
    # Step 3: Berechnung statistischer Kennzahlen (Metriken)
    # ----------
    print("\n Berechne Precision, Recall und F1-Score...")

    # Generierung des Classification Reports als Dictionary für die Analyse
    # "precision", "recall", "f1-score", "support"
    report_dict = classification_report(y_true, y_pred, target_names = class_names, output_dict =True)

    # Generierung des Reports als Text für die Konsolenausgabe
    report_text = classification_report(y_true, y_pred, target_names = class_names)

    # ----------
    # Step 4: Visualisierung und Reporting
    # ----------

    # Erstellen des Ausgabeordners 'reports', falls nicht vorhanden
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4.1 Erstellung und Speicherung der Konfusionsmatrix
    print("\n Erstelle Konfusionsmatrix...")
    cm = confusion_matrix(y_true, y_pred)

    # Konfiguration des Plots (Größe, Farben, Beschriftungen)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm,
                annot = True,               # Annotations in der Matrix
                fmt = 'd',                  # Format, d = Dezimalzahl
                cmap = 'Blues',             # Colormap (Farben) blau
                xticklabels = class_names,  # x-Achsenbeschriftungen
                yticklabels = class_names   # y-Achsenbeschriftungen
                )
    plt.title('Confusion Matrix (Tierklassifikation)')
    plt.xlabel('Vorhergesagte Klasse (Predicted)')
    plt.ylabel('Tatsächliche Klasse (Ground Truth)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Speichern der Grafik als PNG-Datei im Ordner 'reports'
    save_path_cm = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path_cm)
    print(f"Konfusionsmatrix erfolgreich gespeichert unter {save_path_cm}")

    # 4.2 Identifikation schwacher Klassen (F1-Score < 0.80)
    print(f"\n Identifikation schwacher Klassen (F1-Score < {THRESHOLD_F1}):")

    weak_classes = []

    # Iteration über alle Klassen im Report
    for cls in class_names:
        # Zugriff auf die Metriken der aktuellen Klasse
        metrics = report_dict[cls]
        f1 = metrics['f1-score']

        # Überprüfen, ob der F1-Score kleiner als der Grenzwert ist
        if f1 < THRESHOLD_F1:
            weak_classes.append({
                "Tierart" : cls,
                "F1-Score" : f1,
                "Precision" : metrics['precision'],
                "Recall" : metrics['recall'],
                "Anzahl Bilder" : metrics["support"]
            })

    # 4.3 Ausgabe der schwachen Klassen
    print("-"*60)
    print("Classification Report")
    print(report_text)
    print("-"*60)

    # Wenn schwache Klassen gefunden wurden, Warnung ausgeben und CSV speichern
    if len(weak_classes) > 0:
        print(f"Warnung: {len(weak_classes)} Tierarten haben einen F1-Score unter {THRESHOLD_F1}.")

        # Erstellen eines Pandas DataFrame
        df_weak = pd.DataFrame(weak_classes)

        # Formatierte Ausgabe in der Konsole
        print(df_weak.to_string(index=False, formatters = {
            "F1-Score": "{:.2f}".format,
            "Precision": "{:.2f}".format,
            "Recall": "{:.2f}".format
        }))

        # Speichern der Details in der CSV-Datei
        save_path_weak = os.path.join(OUTPUT_DIR, 'weak_classes.csv')
        df_weak.to_csv(save_path_weak, index=False)
        print(f"Details der schwachen Klassen wurden in {save_path_weak} gespeichert.")
    else:
        print("\n Erfolg: Alle Klassen erreichen den geforderten F1-Score.")

if __name__ == "__main__":
    run_statistical_analysis()