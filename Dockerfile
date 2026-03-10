# 1. Basis-Image: Python 3.12
FROM python:3.12-trixie

# 2. Arbeitsverziechnis im Container festlegen
WORKDIR /app

# 3. Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Den gesamten Projektcode (App, Modell, Logo, Config) in den Container kopieren
#    assume training has been performed in git-root
COPY best_model.keras model.keras
COPY src/04_deployment/.config .streamlit/config.toml
COPY logo.jpg src/04_deployment/streamlit_app.py .

# 5. Streamlit-Konfigurationsordner sicherstellen
RUN mkdir -p .streamlit

# 6. Port 8501 freigeben (Standardport für Streamlit)
EXPOSE 8501

# 7. Healthcheck (Prüft regelmäßig, ob die App noch reagiert)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 8. Startbefehl: App auf Port 8501 starten und für alle Netzwerk-Schnittstellen (0.0.0.0) freigeben
ENV STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
