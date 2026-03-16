# Bildbasierte Erkennung von Prädatoren von Nestern.

<div align="justify">
Dieses Repository enthält eine Pipeline zur effizienten Filterung und Vorverarbeitung von Bilddaten der NABU-Wildkameras. Das Hauptziel ist es, die optimalen Frames aus Serienaufnahmen (Bursts) zu extrahieren und störende Metadaten-Bereiche zu entfernen, um einen hochwertigen Trainingsdatensatz für die Wildtierklassifikation zu erstellen.
</div>

## **$\color{red}{Wichtiger Hinweis zur Datenvorbereitung (Manuelle Sortierung)}$**
<div align="justify">
<b>ACHTUNG:</b> Die von NABU bereitgestellten Originalbilder sind unstrukturiert und enthalten oft tief verschachtelte Unterordner (Ordner innerhalb von Ordnern). Da die Pipeline eine klare Struktur erwartet, müssen die Daten <b>zwingend manuell</b> vorbereitet werden:
<br></br>
1. Überprüfen Sie alle Unterordner der Rohdaten gründlich.<br></br>
2. Erstellen Sie für jede Tierart (Spezies) einen eigenen, eindeutig benannten Ordner innerhalb von <code>data/raw/</code>. <br></br>
3. Sortieren Sie alle relevanten Bilder manuell in diese artspezifischen Ordner ein.
<br></br>
Ohne diese manuelle Konsolidierung der Daten kann die Pipeline die Bilder nicht korrekt zuordnen oder verarbeiten.
</div>


## $\color{blue}{Verzeichnisstruktur}$
<div align="justify">
Alle Skripte arbeiten mit relativen Pfaden und setzen die folgende Standard-Struktur voraus. Der Ordner data/raw/ muss manuell erstellt und mit Daten gefüllt werden; alle anderen Verzeichnisse werden bei Bedarf automatisch von den Skripten generiert.
</div>
<br></br>

```
ROOT/
├── data/
│   ├── raw/                  # Input: Originalbilder vom NABU
├── src/
│   ├── 01_preprocessing/     # Vorverarbeitung
│   ├── 02_training/          # Modelltraining
│   └── 03_evaluation/        # Qualitätskontrolle
│   └── 04_deployment/
│        └── logo.jpg         # NABU-logo
├── evaluation/               # Analyse-Reports (Confusion Matrix, Visualisierungen)
├── Dockerfile
├── logo.jpg                  # NABU-logo
├── requirements-dev.txt      # Notwendige Bibliotheken für Preprocessing und Machine Learning
└── requirements.txt          # Erforderlich für die Installation von Streamlit (Deployment)

```

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
Starten Sie die Streamlit-Anwendung für die interaktive Modellinferenz:
<br></br>
```
Bash
python -m streamlit run src/04_deployment/streamlit_app.py
