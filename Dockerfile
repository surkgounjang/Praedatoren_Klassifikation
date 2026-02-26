# 1. Basis-Image: Python 3.9
FROM python:3.10-bullseye

# 2. Arbeitsverziechnis im Container festlegen
WORKDIR /app

# 3. System-Abhängigkeiten installieren
# Notwendig für Bildverarbeitung/OpenCV
# build-essential: Compiler für bestimmte python-Pakete
# libgl1-mesa-glx: Grafikbibliothek für Bildoperationen
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# 4. Requirements kopieren und installieren
# (Dieser Schritt wird gecached, um spätere Builds zu beschleunigen)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Den gesamten Projektcode (App, Modell, Logo, Config) in den Container kopieren
COPY . .

# 6. Streamlit-Konfigurationsordner sicherstellen
RUN mkdir -p .streamlit

# 7. Port 8501 freigeben (Standardport für Streamlit)
EXPOSE 8501

# 8. Healthcheck (Prüft regelmäßig, ob die App noch reagiert)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Startbefehl: App auf Port 8501 starten und für alle Netzwerk-Schnittstellen (0.0.0.0) freigeben
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
