# Hinweis: Dieses Skript wurde für persönliche Lernzwecke erstellt.

"""
Beschreibung:
Dieses Skript trainiert ein ResNet50-Modell mittels Transfer Learning.
Besonderheiten:
- 'Weight Reusing'
- Anstatt die Gewichte zu initialisieren, werden die Gewichte aus dem ImageNet-Datensatz kopiert.

Strategie:
1. Fuchs (NABU) → Red Fox (ImageNet ID 277)
2. Möwen (NABU) → Albatross (ImageNet ID 146) [Visueller Proxy]
3. Krähen (NABU) → Zufällige Initialisierung (Kein passendes Label)
"""

import os # für Betriebssystem-Operationen (z.B. Dateipfade)
import numpy as np # für numerische Berechnungen
import tensorflow as tf # tensorflow Version: 2.20.0
from tensorflow.keras import layers, models, optimizers # Keras-Komponenten

# ===== Einstellung
DATA_DIR = '/home/surkgoun/nabu-project/nabu_split' # Pfad zum Trainingsdatensatz
# Legt die Bildgröße fest (Standard: 224x224 Pixel)
IMG_SIZE = (224, 224)
# Bestimmt die Anzahl der Bilder, die pro Traingsschritt verarbeitet werden
BATCH_SIZE = 32

# Maximale Anzahl der Epochen
# Dank EarlyStopping wird das Training früher enden.
EPOCHS = 100
# Die Lernrate (sehr niedrig für Fine-tuning)
LEARNING_RATE = 0.0001
# Anzahl der Tierarten
NUM_CLASSES = 17

# ===== Mapping-Tabelle für die 'Weight Reusing'-Strategie
# Verknüpft Tierarten mit ImageNet-IDs
# None: keine passende ImageNet-Klasse existiert (Random Initialization)
# Wichtig: Die Namen im IMAGENET_MAP müssen exakt mit den Ordnernamen übereinstimmen
# bitte auf Groß-/Kleinschreibung achten
IMAGENET_MAP = {
    "fuchs": 277, # 277 = Red Fox, exaktes Match
    "kraehen": None, # Kein passendes Label, Random Initialization
    "kolkrabe": None, # Kein passendes Label, Random Initialization
    "marderhund": 388, # 388 = raccoon, Visueller Proxy (Waschbär)
    "moewen": 146, # 146 = albatross, Proxy-Strategie (nach WordNet-Logik)
    "seeadler": 22, # 22 = bald_eagle, Semantischer Proxy
    "iltis": 361, # 361 = polecat, exaktes Match
    "rind": 345, # 345 = ox, Semantischer Proxy (Ochse)
    "steinmarder": 358, # 358 = marten, exaktes Match
    "steinwaelzer": 139, # 139 = turnstone, exaktes Match
    "rohrweihe": 21, # 21 = kite, Semantischer Proxy (Gleitaar)
    "igel": 334, # 334 = hedgehog, exaktes Match
    "austernfischer": 144, # 144 = oystercatcher, exaktes Match
    "dachs": 362, # 362 = badger, exaktes Match
    "habicht": 21, # 21 = kite, Semantischer Proxy (Gleitaar)
    "maeusebussard": 21, # 21 = kite, Semantischer Proxy (Gleitaar)
    "hermelin": 356, # 356 = weasel, Semantischer Proxy (wiesel)
}
# =====

# ===== Funktion zum Erstellen des ResNet-Modells
# return : model
def build_resnet(class_names):
    # Gibt eine Statusmeldung auf der Konsole aus
    print("Baue ResNet50-Modell mit Weight Reusing...")

    # Step 1: Initialisierung beider Modelle (ResNet50-Modell A und ResNet50-Modell B)
    # ResNet50-Modell A (Base Model, "Der Empfänger")
    base_model = tf.keras.applications.ResNet50(
        weights = 'imagenet', # Nutzt vortrainierte Gewichte aus ImageNet (Transfer Learning)
        include_top = False, # entfernt den originalen Klassifikations-Layers (Ich baue einen eigenen)
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3) # Erwartete Bildgröße (224x224 Pixel) und 3 Farbkanäle (RGB)
    )

    # Friert die Gewichte des Basis-Modells ein, damit sie nicht trainiert werden
    base_model.trainable = False

    # Lädt ein zweites ResNet50-Modell (diesmal MIT Top-Layer) als Referenz
    # ResNet50-Modell B (Reference Model, "Der Wissensspender")
    print("Baue ResNet50-Referenzmodell...")
    ref_model = tf.keras.applications.ResNet50(
        weights = 'imagenet',
        include_top = True # Lädt das Modell mit dem Klassifikations-Kopf (um Gewicht zu extrahieren)
    )

    # Extrahiert die Gewichte (Weights) des letzten Layers (shape: 2048x1000)
    imagenet_weights = ref_model.layers[-1].get_weights()[0]
    # Extrahiert die Bias-Werte (Biases) des letzten Layers (shape: 1000)
    imagenet_bias = ref_model.layers[-1].get_weights()[1]

    # Erstellt leere Matrizen für die neuen Gewichte (shape: 2048x17)
    # 2048: Merkmale. Ein Beispiel dafür: Sind die Ohren spitz?
    # 17: Tierarten
    new_weights = np.zeros((2048, len(class_names))) # ein NumPy-Array (Matrix) mit 2048 Zeilen und 17 Spalten
    # Erstellt leere Matrizen für die neuen Bias-Werte (shape: 17)
    new_bias = np.zeros(len(class_names))

    # Gibt eine Überschrift für die folgende Tabelle aus
    print("-"*50)
    print(f"{'NABU Klasse':<20} | {'Strategie':<20} | {'ImageNet-ID'}")
    print("-"*50)

    # Step 2: Iteration über alle 17 Tierarten und Prüfung der IMAGENET_MAP-Tabelle
    # Schleife über alle Tierarten, um die Gewichte zuzuweisen
    for i, class_name in enumerate(class_names):
        # Sucht die passende ImageNet-ID aus IMAGENET_MAP
        taget_id = IMAGENET_MAP[class_name] # z.B fuchs -> 277

        # Wenn eine passende ID gefunden wurde (Gewichtswiederverwendung)
        if taget_id is not None:
            # Step 3: Bei Vorhandensein einer ImageNet-ID: Zugriff auf den letzten Layer von ResNet50-Modell B
            # Kopiert die Gewichte von ImageNet in meiner Modell
            new_weights[:, i] = imagenet_weights[:, taget_id]
            # Kopiert den Bias-Wert von ImageNet in meinen Modell
            new_bias[i] = imagenet_bias[taget_id]
            print(f"{class_name:<20} | {'Weight Reusing'} | {taget_id}")
        # Wenn keine ID gefunden wurde (Random Initialization)
        else:
            # Gibt Status "Random Initialization" auf der Konsole aus"
            print(f"{class_name:<20} | {'Random Initialization'} | {'-'}")
            # Initialisiert die Gewichte mit kleinen Zufallswerten (Normalverteilung)
            new_weights[:,i] = np.random.normal(0,0.01,(2048,))

    # Git einen Trennlinie aus
    print("-"*50)

    # Definiert den Input-Layer des Modells (Bildgröße + 3 RGB)
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Wendet die ResNet50 auf die Bilder an
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    # Leitet die Daten durch das Basis-Modell
    x = base_model(x, training=False) # training=False: Jetzt ist keine Lernzeit (Training), sondern Prüfungszeit
    # Reduziert die Dimensionen durch Global Average Pooling (2048 Feature)
    x = layers.GlobalAveragePooling2D()(x)
    # Fügt Dropout hinzu, um Overfitting zu verhindern
    x = layers.Dropout(0.2)(x)

    # Erstellt den Output-Layer mit 17 Neuronen (für 17 Tierarten)
    # 'softmax' wandelt die Ausgabe in Wahrscheinlichkeit um
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='custom_head')(x)

    # Baut das Modell zusammen (Input -> Output)
    model = tf.keras.Model(inputs, output)

    # Setzt die manuell vorbereiteten Gewicht (Weigt Reusing) in den Output-Layer ein
    model.get_layer('custom_head').set_weights([new_weights, new_bias])

    # Gibt das gertige Modell zurück
    print("Modell erstellt.")
    return model

# ===== Funktion main()
def main():
    # Zweck: Lädt den Trainingsdatensatz aus dem Ordner
    # Ausgabe: Erstellt ein tf.data.Dataset für Training
    # Verwendet die Namen der Unterordner automatisch als Labels
    print("Lade Trainingsdatensatz...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),    # Pfad zum Trainingsdatensatz
        image_size = IMG_SIZE,              # Bildgröße (224x224 Pixel) ohne 3 RGB
        batch_size = BATCH_SIZE,            # Batch-Größe
        shuffle = True                      # Shuffle der Bilder (Wichtig: Bilder zufällig mischen
    )

    # Lädt den Validierungsdatensatz aus dem Ordner
    print("Lade Validierungsdatensatz...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
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

    # Ruf model auf
    model = build_resnet(class_names)

    # Konfiguriert das Modell für das Training
    model.compile(
        # Verwendet den Adam-Optimierer mit einer niedrigen Lernrate
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE),
        # Verwendet Sparse Categorical Crossentropy als Verlustfunktion
        loss = 'sparse_categorical_crossentropy',
        # Überwacht die Genauigkeit (accuracy) wärend des Trainings
        metrics = ['accuracy']
    )

    # Startet den Trainingsprozess
    print("Starte Training...")
    history = model.fit(
        train_ds,                   # Trainingsdaten
        validation_data = val_ds,   # Validierungsdaten
        epochs = EPOCHS,            # Maximale Anzahl der Epochen
        callbacks = [
            # EarlyStopping: Stoppt, wenn val_loss 5 Epochen lang nicht sinkt
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            # ModelCheckpoint: Speichert immer das beste Modell (höchste val_accuracy)
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
    )

    # Speichert das finale Modell nach dem Training
    print("Speichere Modell...")
    # Speichert im aktuellen Ordner
    model.save('final_nabu_resnet.keras')

    # Speichert den Trainingsverlauf (Loss/Accuracy) als CSV-Datei
    import pandas as pd
    pd.DataFrame(history.history).to_csv('training_history.csv')
    print("Training verlauf gespeichert.")

if __name__ == "__main__":
    main()
