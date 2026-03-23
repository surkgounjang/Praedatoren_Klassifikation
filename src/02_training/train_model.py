# Hinweis: Dieses Skript wurde für persönliche Lernzwecke erstellt.

"""
Versionshistorie (für Dokumentationszweck)

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
"""

import os # Für Betriebssystem-Operationen (z.B. Dateipfade)
import numpy as np # für numerische Berechnungen

import tensorflow as tf # tensorflow Version: 2.20.0
import keras
from keras import layers, optimizers, callbacks # Keras-Komponenten
from keras.utils import image_dataset_from_directory

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight # Für die Berechnung der Klassengewichte

# ===== Einstellung
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(current_dir, '../../data/nabu_split/')) # Pfad zum Trainingsdatensatz
MODEL_PATH = os.path.abspath(os.path.join(current_dir, '../../models/final_nabu_resnet.keras')) # Pfad zum Modell
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Legt die Bildgröße fest (Standard: 224x224 Pixel)
IMG_SIZE = (224, 224)
# Bestimmt die Anzahl der Bilder, die pro Traingsschritt verarbeitet werden
BATCH_SIZE = 64 # 32 -> 64

# Training-Konfiguration
EPOCHS_HEAD = 100 # Phase 1: Nur den Klassifikator trainieren 20->40
EPOCHS_FINE = 200 # Phase 2: Fine-Tuning des Basis-Modells 80 -> 200

NUM_CLASSES = 14 # 17 -> 14 Update: 12.01.2026

LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINE = 1e-6 # 1e-5 -> 5e-6 -> 1e-6

# ===== Mapping-Tabelle für die 'Weight Reusing'-Strategie
# Verknüpft Tierarten mit ImageNet-IDs
# None: keine passende ImageNet-Klasse existiert (Random Initialization)
# Wichtig: Die Namen im IMAGENET_MAP müssen exakt mit den Ordnernamen übereinstimmen
# bitte auf Groß-/Kleinschreibung achten
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
}
# =====

# ===== Hilfsfunktion: Visualisierung
def plot_history(history, fine_tune_epoch = None):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(12,6))

    # Accuracy Plot
    plt.subplot(1,2,1) # links
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if fine_tune_epoch:
        plt.plot(
            [fine_tune_epoch, fine_tune_epoch], # x
            plt.ylim(), # y
            label='Start Fine-tuning Epoch',
            linestyle='--',
        )
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1,2,2) # rechts
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if fine_tune_epoch:
        plt.plot(
            [fine_tune_epoch, fine_tune_epoch],
            plt.ylim(),
            label='Start Fine-tuning Epoch',
            linestyle='--',
        )
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('train_history_plot.png')
    print("Grafik gespeichert: train_history_plot.png")


# ===== Funktion zum Erstellen des ResNet-Modells
# return: model
def build_resnet(class_names):
    # Gibt eine Statusmeldung auf der Konsole aus
    print("Baue ResNet50-Modell mit Weight Reusing...")

    # Step 1: Initialisierung beider Modelle (ResNet50-Modell A und ResNet50-Modell B)
    # ResNet50-Modell A (Base Model, "Der Empfänger")
    base_model = keras.applications.ResNet50(
        weights = 'imagenet', # Nutzt vortrainierte Gewichte aus ImageNet (Transfer Learning)
        include_top = False, # entfernt den originalen Klassifikations-Layers (Ich baue einen eigenen)
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3) # Erwartete Bildgröße (224x224 Pixel) und 3 Farbkanäle (RGB)
    )

    # WICHTIG: Zuerst einfrieren (Freeze), um die vortrainierten Merkmale nicht zu zerstören
    base_model.trainable = False

    # Lädt ein zweites ResNet50-Modell (diesmal MIT Top-Layer) als Referenz
    # ResNet50-Modell B (Reference Model, "Der Wissensspender")
    print("Extrahiere Gewichte aus ResNet50-Referenzmodell...")
    ref_model = keras.applications.ResNet50(
        weights = 'imagenet',
        include_top = True # Lädt das Modell mit dem Klassifikations-Kopf (um Gewicht zu extrahieren)
    )

    # Extrahiert die Gewichte (Weights) des letzten Layers (shape: 2048x1000)
    imagenet_weights = ref_model.layers[-1].get_weights()[0]
    # Extrahiert die Bias-Werte (Biases) des letzten Layers (shape: 1000)
    imagenet_bias = ref_model.layers[-1].get_weights()[1]

    # Erstellt leere Matrizen für die neuen Gewichte (shape: 2048x17)
    # 2048: Merkmale. Ein Beispiel dafür: Sind die Ohren spitz?
    new_weights = np.zeros((2048, len(class_names))) # ein NumPy-Array (Matrix) mit 2048 Zeilen und 17 Spalten
    # Erstellt leere Matrizen für die neuen Bias-Werte
    new_bias = np.zeros(len(class_names))

    # Gibt eine Überschrift für die folgende Tabelle aus
    print("-"*50)
    print(f"{'NABU Klasse':<20} | {'Strategie':<20} | {'ImageNet-ID'}")
    print("-"*50)

    # Step 2: Iteration über alle Tierarten und Prüfung der IMAGENET_MAP-Tabelle
    # Schleife über alle Tierarten, um die Gewichte zuzuweisen
    for i, class_name in enumerate(class_names):
        # Mapping prüfen
        if class_name not in IMAGENET_MAP:
            raise ValueError(f"Fehler: Klasse '{class_name}' nicht in IMAGENET_MAP gefunden.")

        # Sucht die passende ImageNet-ID aus IMAGENET_MAP
        taget_id = IMAGENET_MAP[class_name] # z.B fuchs -> 277

        # Wenn eine passende ID gefunden wurde (Gewichtswiederverwendung)
        if taget_id is not None:
            # Step 3: Bei Vorhandensein einer ImageNet-ID: Zugriff auf den letzten Layer von ResNet50-Modell B
            # Kopiert die Gewichte von ImageNet in meiner Modell
            new_weights[:, i] = imagenet_weights[:, taget_id]
            # Kopiert den Bias-Wert von ImageNet in meinen Modell
            new_bias[i] = imagenet_bias[taget_id]
            print(f"{class_name:<20} | {'Weight Reusing':<25} | {taget_id}")
        # Wenn keine ID gefunden wurde (Random Initialization)
        else:
            # Gibt Status "Random Initialization" auf der Konsole aus"
            print(f"{class_name:<20} | {'Random Initialization':<25} | {'-'}")
            # Initialisiert die Gewichte mit kleinen Zufallswerten (Normalverteilung)
            new_weights[:,i] = np.random.normal(0,0.01,(2048,))

    # Git einen Trennlinie aus
    print("-"*50)

    # Daten Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
    ], name="data_augmentation")

    # Definiert den Input-Layer des Modells (Bildgröße + 3 RGB)
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = data_augmentation(inputs)

    # Wendet die ResNEt50 auf die Bilder an
    x = keras.applications.resnet50.preprocess_input(x)
    # Leitet die Daten durch das Basis-Modell
    x = base_model(x, training=False) # training=False: Jetzt ist keine Lernzeit (Training), sondern Prüfungszeit
    # Reduziert die Dimensionen durch Global Average Pooling (2048 Feature)
    x = layers.GlobalAveragePooling2D()(x)
    # Fügt Dropout hinzu, um Overfitting zu verhindern
    x = layers.Dropout(0.5)(x) # 0.2 -> 0.5

    # Erstellt den Output-Layer mit Neuronen
    # 'softmax' wandelt die Ausgabe in Wahrscheinlichkeit um
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='custom_head')(x)

    # Baut das Modell zusammen (Input -> Output)
    model = keras.Model(inputs, output)

    # Setzt die manuell vorbereiteten Gewicht (Weigt Reusing) in den Output-Layer ein
    model.get_layer('custom_head').set_weights([new_weights, new_bias])

    print("Modell erstellt.")
    # Gibt das gertige Modell zurück.
    # Gibt auch das base_model zurück, um es später für Fine-Tuning zu verwenden.
    return model, base_model

# ===== Funktion main()
def main():
    # Zweck: Lädt den Trainingsdatensatz aus dem Ordner
    # Ausgabe: Erstellt ein tf.data.Dataset für Training
    # Verwendet die Namen der Unterordner automatisch als Labels
    print("Lade Trainingsdatensatz...")
    train_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),    # Pfad zum Trainingsdatensatz
        image_size = IMG_SIZE,              # Bildgröße (224x224 Pixel) ohne 3 RGB
        batch_size = BATCH_SIZE,            # Batch-Größe
        shuffle = True                      # Shuffle der Bilder (Wichtig: Bilder zufällig mischen
    )

    # Lädt den Validierungsdatensatz aus dem Ordner
    print("Lade Validierungsdatensatz...")
    val_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, 'validation'),  # Pfad zum Validierungsdatensatz
        image_size = IMG_SIZE,          # Bildgroße (224x224 Pixel) ohne 3 RGB
        batch_size = BATCH_SIZE,
        shuffle = False                 # Validierung muss nicht gemischt werden
    )

    # Speichert die Ordnernamen aud dem Datensatz
    class_names = train_ds.class_names
    print(f"Klassen gefunden: {class_names}")

    # Optimiert die Datenpipeline für Performance (caching/Prefetching)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # ===== Berechnung der Class Weights
    print("\n--- Berechnung der Class Weights ---")
    print("Sammle Labels für die Gewichtung...")

    # Da Dataset ein Generator ist,
    y_train_all = []

    for images, labels in train_ds:
        y_train_all.extend(labels.numpy())

    y_train_all = np.array(y_train_all)

    # Berechne Gewicht: 'balanced' sogt dafür, dass seltene Klassen höhere Gewichte bekommen
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train_all),
                                                      y=y_train_all)

    # Konvertierung in Dictionary Format {0: 1.5, 1: 0.8, ...}
    class_weight_dict = dict(enumerate(class_weights))

    print("Class Weights:", class_weight_dict)
    print("--------------------------------")
    # =====

    # Ruf model, base_model auf
    model, base_model = build_resnet(class_names)

    # Phase 1: Training des Klassifikators (Head)
    print("Phase 1: Training des Klassifikators (Head)...")

    # Konfiguriert das Modell für das Training
    model.compile(
        # Verwendet den Adam-Optimierer mit einer niedrigen Lernrate
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        # Verwendet Sparse Categorical Crossentropy als Verlustfunktion
        loss = 'sparse_categorical_crossentropy',
        # Überwacht die Genauigkeit (accuracy) wärend des Trainings
        metrics = ['accuracy']
    )

    # Startet den Trainingsprozess
    print("Starte Training...")
    history_head = model.fit(
        train_ds,                           # Trainingsdaten
        validation_data = val_ds,           # Validierungsdaten
        epochs = EPOCHS_HEAD,               # Maximale Anzahl der Epochen
        class_weight = class_weight_dict,   # Gewichtung der Klassen
        callbacks = [
            # EarlyStopping: Stoppt, wenn val_loss 5 Epochen lang nicht sinkt
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ]
    )

    # Phase 2: Fine-Tuning
    print("Phase 2: Fine-Tuning...")

    # Basis-Modell auftauen
    base_model.trainable = True

    # Ich will nicht alles trainieren, sondern nur die oberen Schichten.
    # ResNet 50 hat viele Layer.
    # Ich friere alle ein, außer die letzten 30.
    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Scheduler: Wenn Val-Loss stagniert, Lernrate reduzieren
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_delta=1e-7,
        verbose=1,
    )

    early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)

    # Modell neu kompilieren mit sehr niedriger Lernrate
    # Eine zu hohe Lernrate würde das vortrainierte Wissen zerstören.
    model.compile(
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE_FINE, epsilon=1e-08),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # Epochen korrekt fortsetzen
    # Initail epoch muss exakt der letzte Epoch der Phase 1 sein

    history_fine = model.fit(
        train_ds,
        epochs = EPOCHS_HEAD + EPOCHS_FINE,
        validation_data = val_ds,
        initial_epoch = history_head.epoch[-1] + 1,
        class_weight = class_weight_dict,               # Gewichtung der Klassen beim Fine-Tuning
        callbacks = [
            early_stopping, checkpoint, reduce_lr]
    )

    # Speichert das finale Modell nach dem Training
    print("Speichere Modell...")
    # Speichert im aktuellen Ordner
    model.save(MODEL_PATH)

    # Historien zusammenfügen für die Auswertung
    acc = history_head.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_head.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_head.history['loss'] + history_fine.history['loss']
    val_loss = history_head.history['val_loss'] + history_fine.history['val_loss']

    history_dict = {
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }

    pd.DataFrame(history_dict).to_csv('history.csv', index=False)

    # Grafik plotten
    plot_history(history_dict, fine_tune_epoch=history_head.epoch[-1] +1)
    print("Trainingsverlauf als CSV gespeichert.")

if __name__ == "__main__":
    main()
