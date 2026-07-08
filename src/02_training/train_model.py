"""
Versionshistorie

Ursprünglicher Ansatz: 13.12.2025
Dieses Skript trainiert ein ResNet50-Modell mittels Transfer Learning.
Besonderheiten:
- 'Weight Reusing'
- Anstatt die Gewichte zu initialisieren, werden die Gewichte aus dem ImageNet-Datensatz kopiert.

Strategie:
1. Fuchs (NABU) → Red Fox (ImageNet ID 277)
2. Möwen (NABU) → Albatross (ImageNet ID 146) [Visueller Proxy]
3. Krähen (NABU) → Zufällige Initialisierung (Kein passendes Label)

Update: 05.01.2026
1. Auflösung der ID-Konflikte (kite: ImageNet ID 21)
2. Data Augmentation zur Kompensation der geringen Datenmenge.
3. Einführung eines 2-Phasen-Trainings (Head-Training + Fine-Tuning

Update: 08.01.2026 (Fix Class Imbalance)
Integration von Class Weights, um das Ungleichgewicht der Daten (z.B.viele Fuchs, wenig Dachs) auszugleichen.

Update: 12.03.2026 (Refactoring auf Keras 3 Standard)
Umstellung im gesamten Skript: Ersetzt von tf.kears durch das keras-Paket.

Update: 18.06.2026
"""
# Für Betriebssystem-Operationen (z.B. Dateipfade)
import os
from pathlib import Path

import tensorflow as tf

# Keras-Komponenten
import keras
from keras import layers, optimizers, callbacks
from keras.utils import image_dataset_from_directory

# Update: 27.05.2026
from keras.layers import BatchNormalization

# Zum Erstellen von DataFrames und Speichern der Ergebnisse als CSV
import pandas as pd
import matplotlib.pyplot as plt

# Für die Berechnung der Klassengewichte
from sklearn.utils import class_weight

# für numerische Berechnungen
import numpy as np

# ===== Einstellung
SRC_DIR = Path(os.getcwd())
PROJECT_ROOT = SRC_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / 'data' / 'split'

MODEL_PATH = PROJECT_ROOT / 'models' / 'final_model.keras'
BEST_MODEL_PATH = PROJECT_ROOT / 'best_model' / 'best_model.keras'

CSV_PATH = PROJECT_ROOT / 'train_history' / 'train_history.csv'
PLOT_PATH = PROJECT_ROOT / 'train_history' / 'train_history.svg'

# Legt die Bildgröße fest (Standard: 224x 224 Pixel)
IMG_SIZE = (224, 224)

# Bestimmt die Anzahl der Bilder, die pro Trainingsschritt verarbeitet werden.
BATCH_SIZE = 64

# Training-Konfiguration
# Phase 1: Nur den Klassifikator trainieren.
EPOCHS_HEAD = 200
# Phase 2: Fine-Tuning des Basis-Modells.
EPOCHS_FINE = 400

LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINE = 10e-6

# ===== Mapping-Tabelle für die 'Weight Reusing (Transfer Leaning)'-Strategie
# die Anzahl der Tierarten.
# WICHTIG: Wenn sich die Anzahl der Tierarten ändert,
# muss dieser Wert unbedingt aktualisiert werden.
NUM_CLASSES = 15

# Verknüpft Tierarten mit ImageNet-IDs
# Wichtig!
# Die Namen im IMAGENET_MAP (Dictionany) müssen exakt mit den Ordnernamen übereinstimmen.
# Bitte auf Groß-/Kleinschreibung achten.
IMAGENET_MAP = {
    "fuchs": 277,           # 277 = Red Fox, exaktes Match
    "kraehen": None,        # Kein passendes Label, Random Initialization
    "kolkrabe": None,       # Kein passendes Label, Random Initialization
    "marderhund": None,     # 388 = giant_panda, Visueller Proxy (Waschbär) -> Kein passendes Label, None Random Initialization
    "moewen": 146,          # 146 = albatross, Proxy-Strategie (nach WordNet-Logik)
    "seeadler": 22,         # 22 = bald_eagle, Semantischer Proxy
    "iltis": 358,           # 361 = polecat, exaktes Match -> 358 = polecat
    "greifvoegel": 21,
    "rind": 345,            # 345 = ox, Semantischer Proxy (Ochse)
    "steinmarder": 359,     # 358 = polecat -> 359 = black-footed-ferret
    "steinwaelzer": 139,    # 139 = ruddy_turnstone, exaktes Match
    "igel": 334,            # 334 = porcupine
    "austernfischer": 143,  # 144 -> 143= oystercatcher, exaktes Match
    "hermelin": 356,        # 356 = weasel
    "andere" : None         # Kein passendes Label, Random Initialization
}

# ===== Hilfsfunktion: Visualisierung
# history: ein Dictionary, das die Trainingsmetriken (accuracy, loss) enthält.
def plot_history(history, fine_tune_epoch = None):
    training_accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    training_loss = history['loss']
    val_loss = history['val_loss']

    # ==========
    plt.figure(figsize=(12,6))
    # ==========
    # Training und Validation accuracy-Plot
    # links
    plt.subplot(1,2,1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')

    if fine_tune_epoch:
        plt.plot(
            # x-Achse
            [fine_tune_epoch, fine_tune_epoch],
            # Y-Achse
            plt.ylim(),
            label='Start Fine-Tuning Epoch',
            linestyle='--'
        )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # ==========
    # Training und Validation loss-Plot
    # rechts
    plt.subplot(1,2,2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    if fine_tune_epoch:
        plt.plot(
            [fine_tune_epoch, fine_tune_epoch],
            plt.ylim(),
            label='Start Fine-Tuning Epoch',
            linestyle='--'
        )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(PLOT_PATH, format='svg', bbox_inches='tight')
    print(f"History Grafik gespeichert: {PLOT_PATH}")

    plt.close()

# ===== Funktion zum Erstellen des ResNet-Modells
# class_names enthält die Namen der Unterordner im Trainingsdaten-Ordner (train).
# return: model, base_model.
# Die Funktion gibt das base_model zurück, um es später für fine-tuning zu verwenden.
def build_resnet(class_names):
    # Gibt eine Statusmeldung auf der Konsole aus.
    print("Baue ResNet50-Modell mit Weight Reusing (Transfer Learning)...\n")

    # =====
    # Step 1.
    # Initialisierung beider Modelle.
    # ResNet50-Modell A und ResNet50-Modell B.

    # Erstellt ResNet50-Modell A.
    # base_model analysiert die Bilddaten.
    base_model = keras.applications.ResNet50(
        # Nutzt vortrainierte Gewichte aus ImageNet (Transfer Learning)
        weights='imagenet',
        # entfernt den originalen Klassifikator (Head)-Layers
        include_top=False,
        # Erwartete Bildgröße und 3 Farbkanäle (RGB)
        # input_shape=(224,224,3)
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # WICHTIG.
    # Zuerst einfrieren (freeze), um die vortrainierten Merkmale nicht zu zerstören.
    base_model.trainable = False

    # Erstellt ResNet50-Modell B
    # Reference Model, sogenannt 'Der Wissensspender'
    ref_model = keras.applications.ResNet50(
        weights='imagenet',
        # Lädt das Modell mit dem Klassifikator (Head)-Layers,
        # um Gewicht zu extrahieren.
        include_top=True
    )

    # ==========
    print("Extrahiert die Gewichte des letzten Layers...\n")
    # get_weights()[0]: Gewichte
    imagenet_weights = ref_model.layers[-1].get_weights()[0]
    # get_weights()[1]: Bias-Werte
    imagenet_bias = ref_model.layers[-1].get_weights()[1]

    # Erstellt leere Matrizen für die neuen Gewichte.
    # shape: 2048 x len(class_names)
    # 2048: Merkmale von Tierart.
    new_weights = np.zeros((2048, len(class_names)))
    # Erstellt leere Matrizen für die neuen Bias-Werte.
    new_bias = np.zeros(len(class_names))
    # ==========

    # =====
    # Step 2.
    # Iteration über alle Tierarten und Prüfung der IMAGENET_MAP-Tabelle.

    # Gibt eine Überschrift für die folgende Tabelle aus.
    print("-" * 50)
    print(f"{'Tierart':<20} | {'Strategie':<40} | {'ImageNet-ID'}")
    print("-" * 50)

    for i, class_name in enumerate(class_names):
        # class_name prüfen
        if class_name not in IMAGENET_MAP:
            raise ValueError(f"Fehler: Tierart {class_name} nicht in IMAGENET_MAP gefunden.")

        # Sucht die passende ImageNet-ID aus IMAGENET_MAP.
        # z.B. fuchs → 277.
        imagenet_id = IMAGENET_MAP[class_name]

        if imagenet_id is not None:
            # kopiert die Gewichte von imagenet_weights in new_weights.
            new_weights[:, i] = imagenet_weights[:, imagenet_id]
            # kopiert den Bias-Wert von imagenet_bias in new_bias.
            new_bias[i] = imagenet_bias[imagenet_id]
            print(f"{class_name:<20} | {'Weight Reusing':<40} | {imagenet_id}")
        else:
            # Wenn eine passende ImageNet-ID gefunden wurde,
            # imagenet_id == None
            # initialisiert die Gewichte mit kleinen Zufallswerten (Normalverteilung)
            new_weights[:, i] = np.random.normal(0,0.01,(2048,))
            print(f"{class_name:<20} | {'Random Initialization':<40} | {'-'}")

    print("\n")
    print("-" * 50)

    # =====
    # Datenaugmentation:
    # Künstliche Erweiterung der Trainingsdaten, um Overfitting zu vermeiden.
    print("Konfiguriere Data Augmentation...")
    data_augmentation = keras.Sequential(
        [layers.RandomFlip('horizontal')],
        name='data_augmentation'
    )
    print("Data Augmentaion Pipeline erfolgreich erstellt.")
    # =====

    # =====
    # Definiert den Input-Layer des Modells
    print("Erstelle den Input-Layer...")
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))       # (224 Pixel, 224 Pixel, 3 RGB)

    x = data_augmentation(inputs)
    # Wendet die ResNet50 auf die Bilder an.
    x = keras.applications.resnet50.preprocess_input(x)
    # base_model analysiert die Bilddaten.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Fügt Dropout hinzu, um Overfitting zu vermeiden.
    x = layers.Dropout(0.5)(x)

    # Erstellt den Output-Layer des Modells.
    # softmax wandelt die Ausgabe in Wahrscheinlichkeit um.
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='custom_head')(x)

    # Baut das Modell zusammen.
    model = keras.Model(inputs, output)

    # Setzt die manuell vorbereiteten Gewichte in den output-layer ein.
    model.get_layer('custom_head').set_weights([new_weights, new_bias])

    print("Keras Modell erstellt.")
    return model, base_model


def main():
    train_dir = DATA_DIR / 'train'
    val_dir = DATA_DIR / 'val'

    # Prüfen, ob train_dir existiert.
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} nicht gefunden.")
    # Prüfen, ob val_dir existiert.
    if not val_dir.is_dir():
        raise FileNotFoundError(f"{val_dir} nicht gefunden.")

    print("Lade Trainingsdatensatz...")
    # Lädt den Trainingsdatensatz aus dem Ordner 'train'
    train_ds = image_dataset_from_directory(
        train_dir,                  # Pfad zum Trainingsdatensatz
        image_size=IMG_SIZE,        # Bildgröße: 224x224Pixel ohne 3 RGB
        batch_size=BATCH_SIZE,      # Batchgröße
        shuffle=True                # Bilder zufällig mischen
    )

    # Lädt den Validierungsdatensatz aus dem Ordner 'val'
    validation_ds = image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False               # Validierung muss nicht gemischt werden.
    )
    print("-"*50)

    # Speichert die Ordnernamen aus dem Ordner 'train'
    class_names = train_ds.class_names
    pad = len(str(len(class_names)))
    print("\n===== Gefundene Tierarten =====")
    for i, class_name in enumerate(class_names):
        print(f"{i:{pad}d} | {class_name}")

    # Optimiert die Datenpipeline für Performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # ===== Berechnung der Class Weights. =====
    # Class Weights: Gleicht ein Ungleichgewicht in train_ds aus.
    # Gibt seltenen Tierarten ein höheres Gewicht.
    print("\n===== Berechnung der Class Weights =====")

    train_all_labels = []
    # train_ds liefert pro Iteration ein Tuple (images, labels)
    for _, labels in train_ds:
        train_all_labels.extend(labels.numpy())

    train_all_labels =np.array(train_all_labels)

    # Berechne Class Weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_all_labels),
        y=train_all_labels
    )

    # Konvertierung in Dictionary-Format
    class_weight_dictionary = dict(enumerate(class_weights))

    for tierart_id, weight in class_weight_dictionary.items():
        tierart_name = class_names[tierart_id]
        print(f"Tierart ID: {tierart_id:2d} | Tierart Name: {tierart_name:<20} | Weight: {weight}")
    print("-"*50)

    # ==================================================
    #
    # ==================================================

    # ===== Phase 1: Nur den Klassifikator(Head)- Layer trainieren
    print("\n=== Phase 1: Nur den Klassifikator (model) trainieren ===")
    # Erstellt base_model und model
    model, base_model = build_resnet(class_names)

    # Konfiguriert 'model' für das Training.
    # 'model.compile' legt die Trainingsregel für 'model' fest.
    model.compile(
        # Verwendet den Adam-Optimierer mit einer niedrigen Lernrate.
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        # Sparse Categorical Crossentropy: Verlustfunktion
        loss='sparse_categorical_crossentropy',
        # Überwacht die Genauigkeit (accuracy) während des Trainings
        metrics=['accuracy']
    )

    # Start den Head-Trainingsprozess
    print("\n--- Starte Training des Klassifikators (Head)... ---")
    head_training_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight_dictionary,
        callbacks=[
            # Stopp, wenn val_loss 5 Epochen lang nicht mehr verbessert wird.
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Phase 2: Fine-Tuning
    # base_model lernt die spezifischen Merkmale meiner Bilder.
    print("\n=== Phase 2: Fine-Tuning (base_model)... ===")

    # base_model enteisen.
    # Dieser Befehl entsperrt alles in base_model.
    # z.B. die oberen Schichten, die unteren Schichten und alle BatchNormalization-Schichten.
    base_model.trainable = True

    # Aber ich will nicht alles trainieren.
    # Ich friere alle ein, außer die letzte 30 Layer.
    fine_tune_at = len(base_model.layers) - 30

    # die unteren Schichten (Bottom Layers) und
    # die darin enthaltenen unteren BN-Schichten werden eingefroren.
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Update: 27.05.2026
    # in den geöffneten oberen Schichten werden die BN (BatchNormalization)-Schichten wieder eingefroren.
    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Scheduler
    # Wenn Validation-Loss stagniert, Lernrate reduzieren.
    reduce_learning_rate = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_delta=1e-7,
        verbose=1
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n Start des Fine-Tuning... ---")
    fine_tuning_training_history = model.fit(
        train_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE,
        validation_data=validation_ds,
        initial_epoch=head_training_history.epoch[-1] + 1,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True),
            reduce_learning_rate
        ]
    )

    # Speichert das finale Modell
    print("\nSpeichere finales Modell...")
    if not MODEL_PATH.parent.is_dir():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"final_model gespeichert : {MODEL_PATH}")

    #==================================================

    # Historien zusammenfügen für die Auswertung
    # accuracy
    acc = head_training_history.history['accuracy'] + fine_tuning_training_history.history['accuracy']
    val_acc = head_training_history.history['val_accuracy'] + fine_tuning_training_history.history['val_accuracy']

    # loss
    loss = head_training_history.history['loss'] + fine_tuning_training_history.history['loss']
    val_loss = head_training_history.history['val_loss'] + fine_tuning_training_history.history['val_loss']

    history_dict = {
        'accuracy':acc,
        'val_accuracy':val_acc,
        'loss':loss,
        'val_loss':val_loss
    }

    if not CSV_PATH.parent.is_dir():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_dict).to_csv(CSV_PATH, index=False)
    print(f"CSV-Datei gespeichert : {CSV_PATH}")

    # Grafik plotten
    plot_history(history_dict, fine_tune_epoch=head_training_history.epoch[-1]+1)
    print(f"Grafik gespeichert:{PLOT_PATH}")

if __name__ == "__main__":
    main()