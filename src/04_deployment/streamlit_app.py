import io
import os
import streamlit as st
import keras
import pandas as pd  # excel
import numpy as np
from PIL import Image  # pillow-Bibliothek für die Bildverarbeitung (öffnen, resizing)
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_cropper import st_cropper

# =====
# 1. Konfiguration & Pfad-Management
# =====
logo_path = "logo.jpg"

# das NABU-Logo zu laden
logo_img = Image.open(logo_path)

# Grundlegende Konfiguration der streamlit-App (Titel, NABU-Logo, Layout)
st.set_page_config(
    page_title="NABU Wildtier-Erkennung",
    page_icon=logo_img,
    layout="wide"  # Nutzt die Bildschirmbreite für das Dashboard
)

# Einbindung des NABU-Logos oben links in sidebar
st.logo(logo_img, size="large")

# =====
# CSS
# =====
# Hier werden Schriftarten, Farben und Container-Stile definiert.
st.markdown("""
    <style>
    /* --- SCHRIFTARTEN --- */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    /* --- LOGO-ANPASSUNG --- */
    [data-testid="stLogo"] {
        width: 180px !important;
        height: auto !important;
    }

    /* --- HERO HEADER (Banner) --- */
    .hero-header {
        background: linear-gradient(90deg, #193256 0%, #009EE3 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .hero-subtitle {
        font-size: 16px;
        margin-top: 8px;
        opacity: 0.95;
        font-weight: 400;
    }

    /* --- CUSTOM SECTION HEADER (Überschriften) --- */
    .custom-header {
        background-color: #F4F8FB;
        border-left: 5px solid #009EE3; /* NABU Blau */
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        color: #193256;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }

    /* --- CONTAINER-STYLING (Card-Look) --- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        padding: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* --- BUTTON-DESIGN --- */
    div.stButton > button {
        background-color: #009EE3;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 12px;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background-color: #0077AA;
    }

    div.stButton > button[kind="primary"] {
        background-color: #DC3545 !important; /* rot */
        border: 1px solid #DC3545 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #C82333 !important; /* rot */
        border: 1px solid #BD2130 !important;
    }

    /* --- METRIKEN (Zahlenwerte) --- */
    [data-testid="stMetricValue"] {
        color: #193256;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# =====
# 2. Modell & Logik
# =====

# Pfad zur trainierten Modelldatei (.keras Format)
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'model.keras')

# Erwartete Eingabegröße des ResNet50-Modells (224x224px)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Klassennamen definieren.
# Anstatt eine externe 'classes.txt' zu laden,
# definiere ich die Liste direkt in der App.
# Das ist sicherer für die Verteilung (docker),
# da keine Datei verloren gehen kann.
# Wichtig!!!: Die Reihenfolge muss exakt der alphabetischen Reihenfolge entsprechen
CLASS_NAMES = [
    "austernfischer",  # 0
    "fuchs",  # 1
    "greifvoegel",  # 2
    "hermelin",  # 3
    "igel",  # 4
    "iltis",  # 5
    "kolkrabe",  # 6
    "kraehen",  # 7
    "marderhund",  # 8
    "moewen",  # 9
    "rind",  # 10
    "seeadler",  # 11
    "steinmarder",  # 12
    "steinwaelzer"  # 13
    # usw. später hinzufügen
]


# Funktion zum Laden des Modells.
# @st.cache_resource verhindert, dass das Modell bei jeder Interaktion neu geladen wird.
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None


# Initialisiere das Modell
model = load_model()

# ---------------------------------------------------------
# Inferenz Funktion
# führt die Vorhersage für ein Bild durch.
# ---------------------------------------------------------
def predict_image(image):
    # 1. Bildgröße anpassen (224x224px)
    img_resized = image.resize((IMG_HEIGHT, IMG_WIDTH))

    # 2. Umwandlung in Numpy-Array und Hinzufügen der Batch-Dimension
    # shape wird (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # 3. Vorhersage durch das Modell.
    # Das Modell gibt ein 2D-Array zurück.
    # z.B. [[0.05, 0.90, 0.05, ...]]
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]

    # 4. Sortieren der Ergebnisse: Top-3
    # return index of top 3
    top_3 = np.argsort(scores)[-3:][::-1]

    results = []
    for i in top_3:
        results.append(
            {
                "class_name": CLASS_NAMES[i],
                "score": float(scores[i])
            }
        )

    return results

# ---------------------------------------------------------
# sliding-window-Algorithmus für batch-Verarbeitung
# ---------------------------------------------------------
def predict_image_sliding_window(image, window_size=(224, 224), step_size=168):
    """
    :param image: Das Image Objekt (RGB).
    :param window_size: Größe des Eingabefensters für das Modell.
    :param step_size: Die Schrittweite des Fensters. 168px bedeutet 25% Überlappung.
    :um Objekte am Rand nicht zu verpassen.
    :return: best_result
    """
    width, height = image.size
    best_result = None
    best_score = -1.0

    # Wenn da Bild kleiner als das Fenster (224x224) ist,
    # normale Vorhersagen (predict_image) nutzen.
    if width < window_size[0] or height < window_size[1]:
        return predict_image(image)

    # Schleife über die y-Achse (Höhe)
    for y in range(0, height - window_size[1] + 1, step_size):
        # Schleife über die x-Achse (Breite)
        for x in range(0, width - window_size[0] + 1, step_size):
            # 1. Ausschnitt erstellen (crop)
            # Kordinaten: (links,, oben, rechts, unten)
            patch = image.crop((x, y, x + window_size[0], y + window_size[1]))

            # 2. Vorhersage für diesen Ausschnitt durchführen
            try:
                result = predict_image(patch)

                # Das Ergebnis ist eine Liste von Dicts.
                # Ich nehme das Top-1 Ergebnis
                if result:
                    top_score = result[0]["score"]
                    if top_score > best_score:
                        best_score = top_score
                        best_result = result

            except Exception:
                # Falls ein Fehler auftritt, ignorieren wir diesen Patch.
                # Das Modell kann nicht auf dieses Bild anwenden.
                pass

    # Falls kein valides Ergebnis gefunden wurde (z.B. technischer Fehler)
    if best_result is None:
        return predict_image(image)

    return best_result

# ---------------------------------------------------------
# Helper Funktion für konsistente HTML-Überschriften
# ---------------------------------------------------------
def styled_header(text):
    st.markdown(f'<div class="custom-header">{text}</div>', unsafe_allow_html=True)


# =====
# Home Page (Startseite)
# =====
def page_home():
    # Den aktuellen Seitenstatus auf "home" setzen
    st.session_state["last_page"] = "home"

    # --- 1. Liste der 16 identifizierbaren Tierarten ---
    species_list = [
        "Austernfischer", "Fuchs", "Greifvögel", "Hermelin",
        "Igel", "Iltis", "Kolkrabe", "Krähen", "Marderhund",
        "Möwen", "Rind", "Seeadler", "Steinmarder", "Steinwälzer"
    ]

    # --- 2. HTML-Generierung für die Tags ---
    tags_html = ""
    for s in species_list:
        tags_html += f'<div class="species-tag">{s}</div>'

    # --- 3. CSS-Styling (Theme-Aware: Passt sich automatisch an) ---
    st.markdown("""
        <style>
        /* Grundlegende Schriftart */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

        /* [Hintergrund] Automatische Anpassung an das Theme (Light/Dark) */
        [data-testid="stAppViewContainer"] {
            background-color: var(--background-color) !important; /* System-Hintergrund */
            font-family: 'Roboto', sans-serif !important;
            color: var(--text-color) !important; /* System-Textfarbe */
        }

        /* Trennlinien */
        hr { border-color: var(--text-color) !important; opacity: 0.2; }

        /* Haupttitel */
        .main-title {
            color: var(--text-color) !important;
            font-weight: 900;
            font-size: 2.5rem;
            margin-bottom: 5px;
            line-height: 1.1;
        }
        /* Untertitel */
        .sub-title {
            color: var(--text-color) !important;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 30px;
            opacity: 0.6;
        }

        /* Willkommens-Überschrift (Links) - Bleibt NABU-Grün */
        .welcome-title {
            font-size: 2.8rem;
            font-weight: 900;
            color: #009640 !important; /* NABU Green */
            margin-bottom: 20px;
            display: block;
        }

        /* Einleitungstext */
        .intro-text {
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 20px;
            color: var(--text-color) !important;
        }

        /* [Tierart-Tags] Container */
        .species-wrapper {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        /* [Tierart-Tags] Einzelner Tag */
        .species-tag {
            /* Hintergrund: System-Sekundärfarbe (bleibt theme-aware) */
            background-color: var(--secondary-background-color) !important;
            /* Text: System-Textfarbe */
            color: var(--text-color) !important;

            /* Rahmen: Neutrales, halb-transparentes Grau für bessere Sichtbarkeit */
            border: 1px solid #a0a0a080 !important;

            padding: 6px 14px;
            border-radius: 4px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: default;
            transition: all 0.3s ease;

            /* Leichter Schatten für Tiefe */
            box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        }

        /* [Tierart-Tags] Hover-Effekt */
        .species-tag:hover {
            background-color: #009640 !important; /* NABU Green */
            color: white !important;
            border-color: #009640 !important;
            transform: translateY(-2px);
            box-shadow: 0 0 10px rgba(0, 150, 64, 0.5) !important;
        }

        /* Funktionsliste (Rechts) - Header */
        .feature-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color) !important;
            margin-bottom: 25px;
            border-bottom: 1px solid var(--text-color) !important;
            opacity: 0.8;
            padding-bottom: 10px;
            margin-top: 10px;
        }

        /* Funktionsliste - Items */
        .feature-item {
            margin-bottom: 30px;
            padding-left: 20px;
            border-left: 4px solid var(--text-color) !important; /* Linie in Textfarbe */
        }
        .feature-name {
            font-weight: 700;
            color: var(--text-color) !important;
            font-size: 1.2rem;
            display: block;
            margin-bottom: 6px;
        }
        .feature-desc {
            color: var(--text-color) !important;
            font-size: 0.95rem;
            line-height: 1.5;
            opacity: 0.7;
        }

        /* Fußzeile */
        .footer-caption {
            margin-top: 80px;
            color: var(--text-color) !important;
            font-size: 1.0rem;
            border-top: 1px solid var(--secondary-background-color) !important;
            padding-top: 20px;
            text-align: center;
            opacity: 0.5;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- 4. Layout-Aufbau ---

    # 4.1. Header (Volle Breite)
    # Titel links, Logo rechts
    col_head_txt, col_head_img = st.columns([3.5, 1], vertical_alignment="center")
    with col_head_txt:
        st.markdown('<div class="main-title">NABU Nest-Protector</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Automatisierte Wildtier-Erkennung</div>', unsafe_allow_html=True)
    with col_head_img:
        st.markdown('<div style="display: flex; justify-content: flex-end;">', unsafe_allow_html=True)
        st.image(logo_img, width=140)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 4.2. Hauptinhalt (Split Layout)
    col_left, col_right = st.columns([1.3, 1], gap="large")

    # [LINKE SPALTE]
    with col_left:
        left_content = f"""
        <div>
            <span class="welcome-title">Willkommen!</span>
            <div class="intro-text">
                Dieses System unterstützt Sie bei der effizienten Auswertung von Wildkamera-Aufnahmen.<br>
                <div style="height: 10px;"></div>
                Das Modell erkennt und unterscheidet automatisch <b>{len(species_list)} Tierarten</b>:
            </div>
            <div class="species-wrapper">
                {tags_html}
            </div>
        </div>
        """
        st.markdown(left_content, unsafe_allow_html=True)

    # [RECHTE SPALTE]
    with col_right:
        st.markdown("""
            <div class="feature-header">Verfügbare Funktionen</div>

            <div class="feature-item">
                <span class="feature-name">📷  Einzelbild-Analyse</span>
                <span class="feature-desc">
                    Laden Sie ein einzelnes Foto hoch, um das Ergebnis sofort zu überprüfen.
                    Ideal für schnelle Tests.
                </span>
            </div>

            <div class="feature-item">
                <span class="feature-name">📂  Batch-Modus</span>
                <span class="feature-desc">
                    Analysieren Sie komplette Ordnerinhalte (z.B. SD-Karten).
                    Leere Bilder werden gefiltert und Ergebnisse als Excel exportiert.
                </span>
            </div>

            <div style="margin-top: 30px; opacity: 0.6; font-style: italic; font-size: 0.9rem;"></div>
        """, unsafe_allow_html=True)

        st.info("💡 Bitte wählen Sie die Funktion im Menü links aus.")

    # 4.3. Fußzeile (Volle Breite)
    st.markdown("""
        <div class="footer-caption">
            Ein Kooperationsprojekt der <b>Hochschule Bochum</b> und dem <b>Michael-Otto-Institut im NABU</b>.
        </div>
    """, unsafe_allow_html=True)


# =====
# 3. SEITEN-DEFINITIONEN (FRONTEND)
# =====

# Einzelbild-Analyse
def page_single_analysis():
    # [Logik] Seite merken
    st.session_state["last_page"] = "single"

    # ---------------------------------------------------------
    # [Logik] Initialisierung des dynamischen Schlüssels für Single-Upload
    # ---------------------------------------------------------
    if "single_uploader_key" not in st.session_state:
        st.session_state["single_uploader_key"] = 0

    # Callback: Löscht das Bild und das Ergebnis
    def clear_single_state():
        st.session_state["single_uploader_key"] += 1
        if "single_result" in st.session_state:
            del st.session_state["single_result"]

    # Hero-Banner
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">📷 Einzelbild-Analyse</div>
        </div>
        """, unsafe_allow_html=True)

    # Layout: Zweispaltig
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    image_to_predict = None    # Variable für das Bild, das analysiert werden soll (Originalbild oder Ausschnitt)
    analyze_click = False

    # --- Linke Spalte: Upload & Buttons
    with col_upload:
        with st.container(border=True):
            styled_header("1. Bildquelle")
            st.info("""
                 **💡 Anleitung:**
                - **Bildauswahl:** Bitte laden Sie ein einzelnes Bild (JPG, PNG) für die Detail-Analyse hoch.
                - **Neues Bild:** Um ein anderes Bild zu analysieren, klicken Sie auf **"🗑️ Entfernen"** oder laden Sie direkt eine neue Datei hoch.
            """)

            st.write("")

            with st.container(border=True):
                # Schwellenwert-Slider (confidence threshold)
                # ermöglicht dem Benutzer, Ergebnisse mit niedriger Wahrscheinlichkeit auszufiltern
                st.markdown("**Schwellenwert-Filter (optional):**")
                threshold = st.slider(
                    "📉 Schwellenwert %",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Wenn die Wahrscheinlichkeit unter diesem Wert liegt, wird das Bild als 'Kein Prädator' markiert."
                )

                # checkbox für st_cropper-Funktion
                st.markdown("**Bild-Ausschnitt (optional):**")
                use_cropper = st.toggle(
                    "✂️ Bild zuschneiden (Cropping)",
                    value=False,
                    help="Aktivieren Sie dies, um einen spezifischen Bereich des Bildes manuell auszuwählen."
                )

            st.write("")

            # File Uploader mit dynamischem Key (zum Resetten)
            uploaded_file = st.file_uploader(
                "Bitte Bilddatei hochladen (JPG, PNG)",
                type=['jpg', 'png'],
                key=f"single_{st.session_state['single_uploader_key']}"
            )

            if uploaded_file:
                # 1. Vorschau anzeigen
                image = Image.open(uploaded_file)
                # Screenshots (PNG) enthalten oft einen Alphakanal (4 Kanäle: RGBA)
                # das Modell erwartet jedoch 3 Kanäle (RGB)
                # daher wird das Bild in RGB konvertiert.
                image = image.convert("RGB")

                # Entscheidung: Originalbild oder cropper
                if use_cropper:
                    st.info("💡 Bitte wählen Sie den Bereich mit dem Tier aus")
                    cropped_image = st_cropper(image,
                                               realtime_update=True,
                                               box_color='#009640',         # NABU Green
                                               aspect_ratio=(1, 1),         # Quadratisch festlegen
                                               should_resize_image=True
                                               )
                    st.caption("Vorschau des Ausschnitts:")
                    st.image(cropped_image, width=150)

                    # Das zu analysierende Bild ist der Ausschnitt (cropped_image)
                    image_to_predict = cropped_image
                else:
                    # Einfache Anzeige des Originalbildes
                    st.image(image, use_container_width=True, caption="Originalbild")
                    image_to_predict = image

                st.divider()

                # 2. Buttons nebeneinander (Wie im Batch-Modus)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    analyze_click = st.button("🔍 Analysieren", key="btn_single_ana", use_container_width=True)
                with col_btn2:
                    st.button("🗑️ Entfernen", on_click=clear_single_state, key="btn_single_clr", type="primary",
                              use_container_width=True)
            else:
                st.info("💡 Bitte laden Sie ein Bild hoch.")

    # --- Rechte Spalte: Ergebnisse
    with col_result:
        with st.container(border=True):
            styled_header("2. Analyseergebnisse")

            # A. Analyse durchführen, wenn Button geklickt.
            if uploaded_file and analyze_click and model and (image_to_predict is not None):
                with st.spinner("Das Modell analysiert das Bild..."):
                    result = predict_image(image_to_predict)
                    # Ergebnis im Session State speichern (damit es stehen bleibt)
                    st.session_state["single_result"] = result

            # B. Gespeichertes Ergebnis anzeigen
            if "single_result" in st.session_state and uploaded_file:
                result = st.session_state["single_result"]
                top1 = result[0]

                # Prüfung gegen den Schwellenwert
                threshold_decimal = threshold / 100

                # Fall 1:
                # Wahrscheinlichkeit ist zu niedrig: kein Prädator
                if top1['score'] < threshold_decimal:
                    st.warning(f"⚠️Unsicheres Ergebnis: Wahrscheinlichkeit unter {threshold}%.")
                    st.markdown(f"# Ergebnis: Kein Praedator")
                    st.caption(
                        f"Das Modell erkannte '{top1['class_name']}' nur mit {top1['score'] :.1%}. Daher wird es als negativ gewertet.")

                # Fall 2:
                # Wahrscheinlichkeit ist genug: Prädator erkannt
                else:
                    # Metric (Große Zahlen)
                    st.markdown(f"# Tierart: {top1['class_name'].title()}")

                    # Wahrscheinlichkeits-Verteilung
                    st.info("#### Wahrscheinlichkeit:")
                    score_pct = top1['score'] * 100
                    bar_color = "#009EE3"  # NABU blau

                    st.markdown(f"""
                                        <div style="
                                            background-color: #e0e0e0;
                                            border-radius: 15px;
                                            height: 30px;
                                            width: 100%;
                                            margin-bottom: 20px;
                                            overflow: hidden;
                                        ">
                                            <div style="
                                                width: {score_pct}%;
                                                background-color: {bar_color};
                                                height: 100%;
                                                border-radius: 15px;
                                                text-align: right;
                                                line-height: 30px;
                                                padding-right: 15px;
                                                color: white;
                                                font-weight: bold;
                                                font-family: sans-serif;
                                                white-space: nowrap;
                                                transition: width 0.5s;
                                            ">
                                                {score_pct:.1f}%
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                    st.divider()

                    # Tabelle
                    st.markdown("#### 📊 Top 3 Vorhersagen")
                    df_c = pd.DataFrame(result)
                    df_c['score'] = df_c['score'] * 100  # Für ProgressColumn skalieren

                    st.dataframe(
                        df_c,
                        column_config={
                            "class_name": "Tierart",
                            "score": st.column_config.ProgressColumn(
                                "Wahrscheinlichkeit", format="%.2f%%", min_value=0, max_value=100)
                        },
                        hide_index=True,
                        use_container_width=True
                    )

            # C. Leerer Zustand oder Aufforderung
            elif not uploaded_file:
                st.info("💡 Bitte laden Sie links ein Bild hoch.")
            elif uploaded_file and "single_result" not in st.session_state:
                st.info("Klicken Sie links auf **'🔍 Analysieren'**, um zu starten.")


# =====
# 4. Seite 2: Batch-Modus mit Top-3 Ergebnissen
# =====
def page_batch_analysis():
    st.session_state["last_page"] = "batch"

    # Hero-banner
    st.markdown("""
        <div class="hero-header">
            <div class="hero-title">📂 Batch-Verarbeitung</div>
        </div>
    """, unsafe_allow_html=True
                )

    # Logik: Initialisierung eds dynamischen Schlüssels (Session State)
    # Dies ist notwendig, um das file_uploader-widget programmgesteuert zurückzusetzen
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # callback-Funktion zum Zurücksetzen des file_uploader-widgets.
    # erhöht den Schlüsselwert, wodurch streamlit das Widget neu rendert und leert
    def clear_upload_state():
        st.session_state["uploader_key"] += 1
        # Wichtig
        # auch die gespeicherten Ergebnisse löschen
        if "batch_results" in st.session_state:
            del st.session_state["batch_results"]

    # Layout: Zweispaltig
    # Linke Spalte(1 Teil): Batch Upload-Bild-Feld
    # Rechte Spalte(1.2 Teile): Ergebnis-Feld
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    # --- Linke Spalte: Batch Upload-Feld
    with col_upload:
        with st.container(border=True):
            styled_header("1. Bildquelle")
            st.info("""
                 **💡 Anleitung (Wichtig):**
                - **Dateien hinzufügen:** Neue Bilder werden zur bestehenden Liste *hinzugefügt*.
                - **Neue Analyse:** Wenn Sie eine komplett neue Gruppe analysieren möchten, klicken Sie bitte zuerst auf **"🗑️ Entfernen"**.
            """)

            st.write("")

            with st.container(border=True):
                # 1. Filter-Einstellung
                st.markdown("**Schwellenwert-Filter (optional):**")
                threshold = st.slider(
                    "📉 Schwellenwert %",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Wenn die Wahrscheinlichkeit unter diesem Wert liegt, wird das Bild als 'Kein Prädator' markiert."
                )
                st.write("")  # Abstand

                # 2. sliding window-Algorithmus auswählen
                st.markdown("**Smart Scan (optional):**")
                use_sliding_window = st.toggle(
                    "🔍 Smart Scan aktivieren (Sliding Window)",
                    value=False,
                    help="Empfohlen für Weitwinkelaufnahmen. Das Bild wird rasterartig abgesucht, um kleine Objekte zu finden. (Rechenintensiv)"
                )

                st.write("")

            # 3. Datei-Upload
            uploaded_files = st.file_uploader(
                "Bitte Bilddateien hochladen (JPG, PNG)",
                type=['jpg', 'png'],
                accept_multiple_files=True,
                key=f"batch_{st.session_state['uploader_key']}"
            )

            analyse_button = False

            # 4. Aktions-Buttons
            # nur anzeigen, wenn Dateien hochgeladen wurden.
            if uploaded_files:
                st.markdown(f"""
                    <div style="font-size: 18px; font-weight: bold; color: #FF4B4B; margin: 10px 0;">
                        📂 <span style="color: #009EE3;">{len(uploaded_files)}</span> Dateien ausgewählt.
                    </div>
                """, unsafe_allow_html=True)

                # Buttons nebeneinander anzeigen
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    analyse_button = st.button("🔍 Analysieren", key="btn_analyse", use_container_width=True)
                with col_btn2:
                    st.button("🗑️ Entfernen", on_click=clear_upload_state, use_container_width=True, type="primary")

    # --- Rechte Spalte: Auswertung und Top-3
    with col_result:
        with st.container(border=True):
            styled_header("2. Analyseergebnisse")

            if analyse_button and model:
                # Initialisierung des Fortschrittsbalkens
                progress_bar = st.progress(0, text="Initialisiere...")

                rows = []
                threshold_decimal = threshold / 100  # Prozent in Dezimalzahl umwandeln

                # Schleife über alle hochgeladenen Bilder
                for i, f in enumerate(uploaded_files):
                    f: UploadedFile
                    try:
                        img = Image.open(f)
                        # Screenshots (PNG) enthalten oft einen Alphakanal (4 Kanäle: RGBA)
                        # das Modell erwartet jedoch 3 Kanäle (RGB)
                        # daher wird das Bild in RGB konvertiert.
                        img = img.convert("RGB")

                        full_predictions = predict_image(img)
                        top1_full = full_predictions[0]

                        result = full_predictions

                        # ---------------------------------------------------------
                        # sliding window Algorithmus
                        # ---------------------------------------------------------
                        if use_sliding_window and top1_full['score'] < threshold_decimal:
                            # Option A: sliding window
                            sliding_prediction = predict_image_sliding_window(img)
                            top1_sliding = sliding_prediction[0]

                            if top1_sliding['score'] > top1_full['score']:
                                result = sliding_prediction
                            else:
                                result = full_predictions
                        else:
                            # Option B: normaler Scan
                            result = predict_image(img)

                        # Top-1 Ergebnis
                        top1 = result[0]
                        top1_name = top1['class_name']
                        top1_score = top1['score']

                        # Filterung nicht erkannter Tiere und Leerbilder.
                        # wenn der Score unter dem Schwellenwert liegt → kein Prädator
                        if top1_score < threshold_decimal:
                            final_result = "Kein Praedator"

                            out_t1_art = "-"
                            out_t1_score = 0.0
                            out_t2_art = "-"
                            out_t2_score = 0.0
                            out_t3_art = "-"
                            out_t3_score = 0.0
                        else:
                            final_result = top1_name

                            out_t1_art = top1_name
                            out_t1_score = top1_score
                            out_t2_art = result[1]['class_name']
                            out_t2_score = result[1]['score']
                            out_t3_art = result[2]['class_name']
                            out_t3_score = result[2]['score']

                        # Datensatz erstellen (Top-1, Top-2 und Top-3)
                        rows.append(
                            {
                                "Datei": f.name,
                                "Finales Ergebnis": final_result,
                                "Top-1 Art": out_t1_art,
                                "Top-1 %": out_t1_score,
                                "Top-2 Art": out_t2_art,
                                "Top-2 %": out_t2_score,
                                "Top-3 Art": out_t3_art,
                                "Top-3 %": out_t3_score
                            }
                        )
                    except Exception as e:
                        st.error(f"Fehler beim Analysieren von {f.name}: {e}")

                    # Fortschrittsbalken aktualisieren
                    prog = (i + 1) / len(uploaded_files)
                    progress_bar.progress(prog, text=f"Verarbeite Bild {i + 1}/{len(uploaded_files)}")

                # Balken entfernen nach Abschluss
                progress_bar.empty()

                # Wichtig: Ergebnis in session state speichern
                st.session_state["batch_results"] = pd.DataFrame(rows)
                st.success("✅ Analyse erfolgreich abgeschlossen.")

            # Gespeicherte Ergebnisse anzeigen (auch nach excel Download)
            if 'batch_results' in st.session_state:
                df = st.session_state["batch_results"]

                # Statistische Zusammenfassung
                st.markdown("##### 📈 Zusammenfassung")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Bildanzahl", len(df))

                # Nur tatsächliche Prädatoren berücksichtigen
                predators_only = df[df["Finales Ergebnis"] != "Kein Praedator"]

                # Durchschnittliche Wahrscheinlichkeit
                if not predators_only.empty:
                    avg_score = predators_only["Top-1 %"].mean()
                    col_m2.metric("Ø Top-1 Scores", f"{avg_score:.1%}")

                    modes = predators_only["Finales Ergebnis"].mode()
                    top_animal = modes[0] if not modes.empty else "-"
                    col_m3.metric("Häufigste Art", top_animal.title())
                else:
                    col_m2.metric("Ø Confidence", "-")
                    col_m3.metric("Häufigste Art", "Keine Praedatoren")

                st.divider()

                # Detailliste mit visuellen Balken für Top-1, 2, 3
                st.markdown("##### 📋 Detaillierte Analyseergebnisse (Top 3)")
                st.dataframe(
                    df,
                    column_config={
                        "Datei": st.column_config.TextColumn("Dateiname", width="medium"),
                        # Spalten für Top-1
                        "Top-1 Art": st.column_config.TextColumn("🥇 Top-1 Art"),
                        "Top-1 %": st.column_config.ProgressColumn("Score 1", format="%.2f%%", min_value=0,
                                                                   max_value=1),
                        # Spalten für Top-2
                        "Top-2 Art": st.column_config.TextColumn("🥈 Top-2 Art"),
                        "Top-2 %": st.column_config.ProgressColumn("Score 2", format="%.2f", min_value=0, max_value=1),

                        # Spalten für Top-3
                        "Top-3 Art": st.column_config.TextColumn("🥉 Top-3 Art"),
                        "Top-3 %": st.column_config.ProgressColumn("Score 3", format="%.2f", min_value=0, max_value=1),
                    }, use_container_width=True, height=400
                )

                # excel-Export
                st.divider()

                st.markdown("##### 📥 Export")
                col_ex1, col_ex2 = st.columns(2)

                try:
                    import xlsxwriter
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                        df.to_excel(writer,
                                    index=False,
                                    sheet_name='Ergebnisse'
                                    )

                        workbook = writer.book
                        worksheet = writer.sheets['Ergebnisse']

                        percent_format = workbook.add_format({'num_format': '0.00%'})

                        worksheet.set_column('D:D', 12, percent_format)
                        worksheet.set_column('F:F', 12, percent_format)
                        worksheet.set_column('H:H', 12, percent_format)

                        worksheet.set_column('A:A', 30)
                        worksheet.set_column('B:B', 20)
                        worksheet.set_column('C:C', 20)
                        worksheet.set_column('E:E', 20)
                        worksheet.set_column('G:G', 20)

                    with col_ex1:
                        st.download_button("Excel (.xlsx)", buf.getvalue(), "NABU_Results.xlsx",
                                           use_container_width=True)
                except ImportError:
                    with col_ex1:
                        st.error("excel-Export nicht möglich: xlsxwriter-Paket fehlt.")
                except Exception as e:
                    with col_ex1:
                        st.error(f"Fehler beim excel-Export: {e}")

                with col_ex2:
                    st.download_button("CSV (.csv)", df.to_csv(index=False).encode('utf-8'), "NABU_Results.csv",
                                       use_container_width=True)

            elif not uploaded_files:
                st.info("💡 Bitte laden Sie zuerst Bilder auf der linken Seite hoch.")


# =====
# 5. NAVIGATION & START
# =====
pg_home = st.Page(page_home, title="Startseite", icon="🏠", default=True)
pg1 = st.Page(page_single_analysis, title="Einzelbild-Analyse", icon="📷")
pg2 = st.Page(page_batch_analysis, title="Batch-Modus", icon="📂")

pg = st.navigation({
    "Startseite": [pg_home],
    "Funktionen": [pg1, pg2]
})

pg.run()
