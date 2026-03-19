# Bildbasierte Erkennung von Prädatoren von Nestern.

<div align="justify">
Dieses Repository enthält eine Pipeline zur effizienten Filterung und Vorverarbeitung von Bilddaten der NABU-Wildkameras. Das Hauptziel ist es, die optimalen Frames aus Serienaufnahmen (Bursts) zu extrahieren und störende Metadaten-Bereiche zu entfernen, um einen hochwertigen Trainingsdatensatz für die Wildtierklassifikation zu erstellen.
</div>

## $\color{blue}{Verzeichnisstruktur}$
<div align="justify">
Alle Skripte arbeiten mit relativen Pfaden und setzen die folgende Standard-Struktur voraus.
</div>
<br></br>

```
ROOT/
├── data/                     # nicht im Repository enthalten
│   └── raw/                  # Originalbilder vom NABU nach Tierart sortiert
│       ├── austernfischer/   # Bilder der Tierart Austernfischer ohne weitere Unterverzeichnisse
│       ├── fuchs/            # Bilder der Tierart Fuchs ohne weitere Unterverzeichnisse
│       ┊   ...               # weitere Tierarten
│       └── steinwaelzer/     # Bilder der Tierart Steinwaelzer ohne weitere Unterverzeichnisse
├── src/
│   ├── 01_preprocessing/     # Vorverarbeitung
│   ├── 02_training/          # Modelltraining
│   ├── 03_evaluation/        # Qualitätskontrolle
│   └── 04_deployment/
│        └── logo.jpg         # NABU-logo
├── evaluation/               # nicht im Repository enthalten, Output für Analyse-Reports
├── Dockerfile
├── logo.jpg                  # NABU-logo
├── requirements-dev.txt      # Notwendige Bibliotheken für Preprocessing und Machine Learning
└── requirements.txt          # Erforderlich für die Installation von Streamlit (Deployment)

```

Die Bilddaten, die für dieses Projekt benötigt werden, sind nicht öffentlich verfügbar und nur für Projektangehörige zugänglich. Die Vorverarbeitungsskripte erwarten die Bilder nach Tierart sortiert in `data/raw/` und die Verzeichnisnamen müssen den Klassennamen entsprechen. Der Ordner `data/raw/` muss manuell erstellt und mit Daten gefüllt werden.

## **$\color{blue}{Workflow: Ausführung der Pipeline}$**
<div align="justify">
Führen Sie die Skripte in der folgenden Reihenfolge aus, um das Projekt zu reproduzieren oder neue Daten zu verarbeiten.
</div>

### **$\color{green}{Schritt 1: Datenvorverarbeitung (Preprocessing)}$**
<div align="justify">
Wichtig:<br></br>
Bevor Sie dieses Skript zum ersten Mal ausführen, müssen Sie input_folder und output_folder innerhalb der Datei 01_select_best_shot.py manuell an die Tierart anpassen, die Sie gerade verarbeiten möchten.
<br></br>
Zu bearbeitende Codezeile:
</div>

```
input_folder = os.path.join(current_dir, '../../data/raw/fuchs/')
output_folder = os.path.join(current_dir, '../../data/processed/fuchs/')
```
<div align="justify">
Best Shot Auswahl: Identifiziert und extrahiert das schärfste Bild aus jeder Burst-Serie basierend auf der Laplace-Varianz.
</div>

```
Bash
python src/01_preprocessing/01_select_best_shot.py
```
---

<div align="justify">
Wichtig: <br></br>
Genau wie bei der Best-Shot-Auswahl müssen Sie auch vor der Ausführung von 02_zuschneiden.py die Verzeichnisse im Quellcode manuell anpassen.

Zu bearbeitende Codezeilen:
```
input_folder = os.path.join(current_dir, '../../data/processed/fuchs/')
output_folder = os.path.join(current_dir, '../../data/final/fuchs/')
```
<br></br>
Bild-Cropping: Entfernt den unteren Metadaten-Balken (ca. 12%), um sicherzustellen, dass das Modell nur auf relevante Bildmerkmale trainiert wird.
</div>
<br></br>

```
Bash
python src/01_preprocessing/02_zuschneiden.py
```
---

<div align="justify">
Datensatz-Splitting: Teilt die bereinigten Daten automatisch in Train (80%), Val (10%) und Test (10%) auf.
</div>
<br></br>

```
Bash
python src/01_preprocessing/03_split_dataset_v2.py
```

---

### **$\color{green}{Schritt 2: Modelltraining (Training)}$**
<div align="justify">
Trainiert ein ResNet50-Modell auf Basis des vorbereiteten nabu_split Datensatzes.
<br></br>
Transfer Learning: Verwendung von ImageNet-Gewichten mit spezifischem Mapping für Wildtierarten.
<br></br>
Class Weights: Automatischer Ausgleich von Klassenimbalancen während des Trainings.
</div>
<br></br>

```
Bash
python src/02_training/train_model.py
```
---
### **$\color{green}{Schritt 3: Evaluation und Validierung}$**
<div align="justify">
Führt eine umfassende Qualitätssicherung in vier Phasen durch:

Phase 1 (Sanity Check): Überprüfung der Modell-Ladefähigkeit und Daten-Integrität.
</div>
<br></br>

```
Bash
python src/03_evaluation/test_phase1.py
```
<div align="justify">
Phase 2 (Statistik): Erstellung der Confusion Matrix und Identifikation "schwacher" Klassen (F1-Score < 0.80).
</div>
<br></br>

```
Bash
python src/03_evaluation/test_phase2.py
```
<div align="justify">
Phase 3 & 4 (Visual & Robustness): Visuelle Prüfung von Vorhersagen und Messung der Top-K Accuracy.
</div>
<br></br>
    
```
Bash
python src/03_evaluation/test_phase3_4.py
```
Alle Ergebnisse werden automatisch im Ordner evaluation/ gespeichert.

---
### **$\color{green}{Schritt 4: Deployment (Interaktive Anwendung)}$**
<div align="justify">
Wichtiger Hinweis:<br></br>
Die Dateien model.keras und logo.jpg müssen sich zwingend im selben Ordner befinden, damit die Anwendung ordnungsgemäß geladen werden kann. <br></br>
Starten Sie die Streamlit-Anwendung für die interaktive Modellinferenz.
</div>
<br></br>

```
Bash
python -m streamlit run src/04_deployment/streamlit_app.py
