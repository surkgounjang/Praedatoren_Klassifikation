# Bildbasierte Erkennung von Prädatoren von Nestern.

Dieses Repository enthält eine Pipeline zur effizienten Filterung und Vorverarbeitung von Bilddaten der NABU-Wildkameras. Das Hauptziel ist es, die optimalen Frames aus Serienaufnahmen (Bursts) zu extrahieren und störende Metadaten-Bereiche zu entfernen, um einen hochwertigen Trainingsdatensatz für die Wildtierklassifikation zu erstellen.

## Verzeichnisstruktur

Alle Skripte arbeiten mit relativen Pfaden und setzen die folgende Standard-Struktur voraus.

``` plain
📁ROOT/
├┈┈ 📂data/                     # nicht im Repository enthalten
│   └── 📂raw/                  # Originalbilder vom NABU nach Tierart sortiert
│       ├── 📂austernfischer/   # Bilder der Tierart Austernfischer ohne weitere Unterverzeichnisse
│       ├── 📂fuchs/            # Bilder der Tierart Fuchs ohne weitere Unterverzeichnisse
│       ┊   ...                 # weitere Tierarten
│       └── 📂steinwaelzer/     # Bilder der Tierart Steinwaelzer ohne weitere Unterverzeichnisse
├── 📁src/
│   ├── 📁01_preprocessing/     # Vorverarbeitung
│   ├── 📁02_training/          # Modelltraining
│   ├── 📁03_evaluation/        # Qualitätskontrolle
│   └── 📁04_deployment/
│        └── logo.jpg         # NABU-logo
├┈┈ 📂evaluation/               # nicht im Repository enthalten, Output für Analyse-Reports
├┈┈ 📂models/                   # nicht im Repository enthalten, Output für Training
├── 📄Dockerfile
├── logo.jpg                  # NABU-logo
├── 📄requirements-dev.txt      # Notwendige Bibliotheken für Preprocessing und Machine Learning
└── 📄requirements.txt          # Erforderlich für die Installation von Streamlit (Deployment)

```

Die Bilddaten, die für dieses Projekt benötigt werden, sind nicht öffentlich verfügbar und nur für Projektangehörige zugänglich. Die Vorverarbeitungsskripte erwarten die Bilder nach Tierart sortiert in `data/raw/` und die Verzeichnisnamen müssen den Klassennamen entsprechen. Der Ordner `data/raw/` muss manuell erstellt und mit Daten gefüllt werden.

## Workflow: Ausführung der Pipeline

Führen Sie die Skripte in der folgenden Reihenfolge aus, um das Projekt zu reproduzieren oder neue Daten zu verarbeiten.

### Schritt 1: Datenvorverarbeitung (Preprocessing)

1. Best Shot Auswahl: Identifiziert und extrahiert das schärfste Bild aus jeder Burst-Serie basierend auf der Laplace-Varianz.

   ``` bash
   python src/01_preprocessing/01_select_best_shot.py
   ```

   > [!IMPORTANT]
   >
   > Bevor Sie dieses Skript zum ersten Mal ausführen, müssen Sie `input_folder` und `output_folder` innerhalb der Datei `01_select_best_shot.py` manuell an die Tierart anpassen, die Sie gerade verarbeiten möchten.

   Zu bearbeitende Codezeile:

   ``` python
   input_folder = os.path.join(current_dir, '../../data/raw/fuchs/')
   output_folder = os.path.join(current_dir, '../../data/processed/fuchs/')
   ```

2. Bild-Cropping: Entfernt den unteren Metadaten-Balken (ca. 12%), um sicherzustellen, dass das Modell nur auf relevante Bildmerkmale trainiert wird.

   ``` bash
   python src/01_preprocessing/02_zuschneiden.py
   ```

   > [!IMPORTANT]
   >
   > Genau wie bei der Best-Shot-Auswahl müssen Sie auch vor der Ausführung von `02_zuschneiden.py` die Verzeichnisse im Quellcode manuell anpassen.

   Zu bearbeitende Codezeilen:

   ``` python
   input_folder = os.path.join(current_dir, '../../data/processed/fuchs/')
   output_folder = os.path.join(current_dir, '../../data/final/fuchs/')
   ```

3. Datensatz-Splitting: Teilt die bereinigten Daten automatisch in Train (80%), Val (10%) und Test (10%) auf.

   ``` bash
   python src/01_preprocessing/03_split_dataset_v2.py
   ```

   - Input: `data/final/`
   - Output: `data/nabu_split/`

### Schritt 2: Modelltraining (Training)

- Trainiert ein ResNet50-Modell auf Basis des vorbereiteten nabu_split Datensatzes.
- Transfer Learning: Verwendung von ImageNet-Gewichten mit spezifischem Mapping für Wildtierarten.
- Class Weights: Automatischer Ausgleich von Klassenimbalancen während des Trainings.

``` bash
python src/02_training/train_model.py
```

- Data-Input: `data/nabu_split/train`
- Model-Output: `models/final_nabu_resnet.keras`

### Schritt 3: Evaluation und Validierung

Führt eine umfassende Qualitätssicherung in vier Phasen durch:

1. Phase (Sanity Check): Überprüfung der Modell-Ladefähigkeit und Daten-Integrität.

   ``` bash
   python src/03_evaluation/test_phase1.py
   ```

2. Phase (Statistik): Erstellung der Confusion Matrix und Identifikation "schwacher" Klassen (F1-Score < 0.80).

   ``` bash
   python src/03_evaluation/test_phase2.py
   ```

3. & 4. Phase (Visual & Robustness): Visuelle Prüfung von Vorhersagen und Messung der Top-K Accuracy.

   ``` bash
   python src/03_evaluation/test_phase3_4.py
   ```

Alle Ergebnisse werden automatisch im Ordner `evaluation/` gespeichert.

### Schritt 4: Deployment (Interaktive Anwendung)

Starten Sie die Streamlit-Anwendung für die interaktive Modellinferenz.

``` bash
python -m streamlit run src/04_deployment/streamlit_app.py
```
