# NABU Wildkamera: Automatische Bild-Selektion & Vorverarbeitung

Dieses Repository enthält Python-Skripte zur effizienten Filterung und Vorverarbeitung von Rohdaten aus NABU-Wildkameras. Ziel des Projekts ist es, aus einer großen Menge an Serienaufnahmen (Bursts) einen bereinigten, hochwertigen Datensatz für das Training von Machine-Learning-Modellen zu erstellen.

## 📌 Funktionen (Features)

* **Intelligente Serienbild-Bereinigung:** Gruppiert Bilder basierend auf dem Aufnahmezeitpunkt (Zeitstempel) und entfernt redundante Aufnahmen.
* **Keyframe-Extraction:** Wählt automatisch die schärfsten Bilder aus einer Serie aus, um Bewegungsunschärfe zu vermeiden.
* **Preprocessing (Cropping):** Entfernt den unteren Informationsbalken (Metadaten, ca. 12% des Bildes), um Bias im Modelltraining zu verhindern.

## 🛠️ Voraussetzungen (Requirements)

Das Projekt basiert auf **Python 3.10. 19**. Folgende Bibliotheken werden benötigt:

* `opencv-python` (Bildverarbeitung & Schärfe-Berechnung)
* `Pillow` (Bildmanipulation)
* `numpy` (Numerische Operationen)
* `tqdm` (Fortschrittsbalken - optional, aber empfohlen)

Installieren Sie die Abhängigkeiten mit:

```bash
pip install -r requirements.txt
