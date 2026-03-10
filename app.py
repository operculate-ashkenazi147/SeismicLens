"""
SeismicLens — Interactive Seismic Waveform Analyzer
Professional geophysics tool: synthetic waveforms, real MiniSEED, FFT,
STA/LTA picking, Butterworth filtering, spectrograms, and export.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io

from utils.signal_processing import (
    bandpass_filter, compute_fft, compute_spectrogram,
    compute_sta_lta, detect_p_wave, taper_signal,
    spectral_metrics, compute_psd,
)
from utils.data_loader import (
    load_mseed, load_csv_signal, generate_synthetic_quake,
    CRUSTAL_LAYERS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internationalisation (i18n)
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGES = {
    "English":  "en",
    "Italiano": "it",
    "Français": "fr",
    "Español":  "es",
    "Deutsch":  "de",
}

STRINGS = {
    # ── Sidebar labels ──────────────────────────────────────────────────────
    "language":         {"en": "Language", "it": "Lingua", "fr": "Langue", "es": "Idioma", "de": "Sprache"},
    "data_source":      {"en": "Data source", "it": "Sorgente dati", "fr": "Source de données", "es": "Fuente de datos", "de": "Datenquelle"},
    "src_synthetic":    {"en": "Synthetic earthquake", "it": "Terremoto sintetico", "fr": "Séisme synthétique", "es": "Sismo sintético", "de": "Synthetisches Erdbeben"},
    "src_mseed":        {"en": "Upload MiniSEED", "it": "Carica MiniSEED", "fr": "Importer MiniSEED", "es": "Cargar MiniSEED", "de": "MiniSEED laden"},
    "src_csv":          {"en": "Upload CSV", "it": "Carica CSV", "fr": "Importer CSV", "es": "Cargar CSV", "de": "CSV laden"},
    "mseed_hint":       {
        "en": "Download free data from <b>IRIS</b> (ds.iris.edu) or <b>INGV</b> (webservices.ingv.it). Export format: MiniSEED.",
        "it": "Scarica dati gratuiti da <b>IRIS</b> (ds.iris.edu) o <b>INGV</b> (webservices.ingv.it). Formato: MiniSEED.",
        "fr": "Téléchargez des données gratuites depuis <b>IRIS</b> (ds.iris.edu) ou <b>INGV</b> (webservices.ingv.it). Format: MiniSEED.",
        "es": "Descarga datos gratuitos de <b>IRIS</b> (ds.iris.edu) o <b>INGV</b> (webservices.ingv.it). Formato: MiniSEED.",
        "de": "Kostenlose Daten von <b>IRIS</b> (ds.iris.edu) oder <b>INGV</b> (webservices.ingv.it). Format: MiniSEED.",
    },
    "csv_hint":         {
        "en": "Expected: 1 column (amplitude) or 2 columns (time, amplitude).",
        "it": "Formato atteso: 1 colonna (ampiezza) o 2 colonne (tempo, ampiezza).",
        "fr": "Format attendu : 1 colonne (amplitude) ou 2 colonnes (temps, amplitude).",
        "es": "Formato esperado: 1 columna (amplitud) o 2 columnas (tiempo, amplitud).",
        "de": "Erwartet: 1 Spalte (Amplitude) oder 2 Spalten (Zeit, Amplitude).",
    },
    "syn_section":      {"en": "Synthetic Generator", "it": "Generatore sintetico", "fr": "Générateur synthétique", "es": "Generador sintético", "de": "Synthetischer Generator"},
    "magnitude":        {"en": "Magnitude M", "it": "Magnitudo M", "fr": "Magnitude M", "es": "Magnitud M", "de": "Magnitude M"},
    "depth":            {"en": "Focal depth (km)", "it": "Profondità focale (km)", "fr": "Profondeur focale (km)", "es": "Profundidad focal (km)", "de": "Herdtiefe (km)"},
    "distance":         {"en": "Epicentral distance (km)", "it": "Distanza epicentrale (km)", "fr": "Distance épicentrale (km)", "es": "Distancia epicentral (km)", "de": "Epizentraldistanz (km)"},
    "noise":            {"en": "Noise level", "it": "Livello di rumore", "fr": "Niveau de bruit", "es": "Nivel de ruido", "de": "Rauschpegel"},
    "duration":         {"en": "Duration (s)", "it": "Durata (s)", "fr": "Durée (s)", "es": "Duración (s)", "de": "Dauer (s)"},
    "filter_section":   {"en": "Butterworth Filter", "it": "Filtro Butterworth", "fr": "Filtre Butterworth", "es": "Filtro Butterworth", "de": "Butterworth-Filter"},
    "filter_toggle":    {"en": "Zero-phase bandpass", "it": "Bandpass zero-phase", "fr": "Passe-bande zéro-phase", "es": "Pasa-banda fase cero", "de": "Nullphasen-Bandpass"},
    "f_low":            {"en": "Low cut (Hz)", "it": "Taglio basso (Hz)", "fr": "Coupure basse (Hz)", "es": "Corte bajo (Hz)", "de": "Untere Grenzfrequenz (Hz)"},
    "f_high":           {"en": "High cut (Hz)", "it": "Taglio alto (Hz)", "fr": "Coupure haute (Hz)", "es": "Corte alto (Hz)", "de": "Obere Grenzfrequenz (Hz)"},
    "filter_order":     {"en": "Order", "it": "Ordine", "fr": "Ordre", "es": "Orden", "de": "Ordnung"},
    "stalta_section":   {"en": "STA/LTA Detector", "it": "Rivelatore STA/LTA", "fr": "Détecteur STA/LTA", "es": "Detector STA/LTA", "de": "STA/LTA-Detektor"},
    "sta_window":       {"en": "STA window (s)", "it": "Finestra STA (s)", "fr": "Fenêtre STA (s)", "es": "Ventana STA (s)", "de": "STA-Fenster (s)"},
    "lta_window":       {"en": "LTA window (s)", "it": "Finestra LTA (s)", "fr": "Fenêtre LTA (s)", "es": "Ventana LTA (s)", "de": "LTA-Fenster (s)"},
    "threshold":        {"en": "Trigger threshold", "it": "Soglia di trigger", "fr": "Seuil de déclenchement", "es": "Umbral de disparo", "de": "Auslöseschwelle"},
    "display_section":  {"en": "Display Options", "it": "Opzioni di visualizzazione", "fr": "Options d'affichage", "es": "Opciones de visualización", "de": "Anzeigeoptionen"},
    "show_raw":         {"en": "Raw signal overlay", "it": "Overlay segnale grezzo", "fr": "Superposition signal brut", "es": "Superposición señal bruta", "de": "Rohsignal einblenden"},
    "show_phase":       {"en": "Phase spectrum", "it": "Spettro di fase", "fr": "Spectre de phase", "es": "Espectro de fase", "de": "Phasenspektrum"},
    "show_psd":         {"en": "Power Spectral Density", "it": "Densità spettrale di potenza", "fr": "Densité spectrale de puissance", "es": "Densidad espectral de potencia", "de": "Leistungsspektraldichte"},
    "show_spec":        {"en": "Spectrogram", "it": "Spettrogramma", "fr": "Spectrogramme", "es": "Espectrograma", "de": "Spektrogramm"},
    "show_stalta":      {"en": "STA/LTA chart", "it": "Grafico STA/LTA", "fr": "Graphique STA/LTA", "es": "Gráfico STA/LTA", "de": "STA/LTA-Diagramm"},
    # ── Hero ─────────────────────────────────────────────────────────────────
    "hero_sub":         {
        "en": "Interactive Seismic Waveform Analyzer · Signal Processing & Spectral Analysis",
        "it": "Analizzatore interattivo di forme d'onda sismiche · Elaborazione del segnale e analisi spettrale",
        "fr": "Analyseur interactif de formes d'onde sismiques · Traitement du signal et analyse spectrale",
        "es": "Analizador interactivo de formas de onda sísmicas · Procesado de señal y análisis espectral",
        "de": "Interaktiver seismischer Wellenform-Analysator · Signalverarbeitung und Spektralanalyse",
    },
    "no_data":          {
        "en": "No data loaded. Select a source in the sidebar or use the Synthetic generator.",
        "it": "Nessun dato caricato. Seleziona una sorgente nella barra laterale o usa il generatore sintetico.",
        "fr": "Aucune donnée chargée. Sélectionnez une source dans la barre latérale ou utilisez le générateur synthétique.",
        "es": "No hay datos cargados. Selecciona una fuente en la barra lateral o usa el generador sintético.",
        "de": "Keine Daten geladen. Wähle eine Quelle in der Seitenleiste oder nutze den synthetischen Generator.",
    },
    # ── Metric labels ─────────────────────────────────────────────────────────
    "m_fs":             {"en": "Sampling Rate", "it": "Freq. campionamento", "fr": "Fréq. échantillonnage", "es": "Frec. muestreo", "de": "Abtastrate"},
    "m_dur":            {"en": "Duration", "it": "Durata", "fr": "Durée", "es": "Duración", "de": "Dauer"},
    "m_peak":           {"en": "Peak Amplitude", "it": "Ampiezza picco", "fr": "Amplitude crête", "es": "Amplitud pico", "de": "Spitzenamplitude"},
    "m_domf":           {"en": "Dominant Freq.", "it": "Freq. dominante", "fr": "Fréq. dominante", "es": "Frec. dominante", "de": "Dominanzfrequenz"},
    "m_parr":           {"en": "P Arrival", "it": "Arrivo onda P", "fr": "Arrivée onde P", "es": "Llegada onda P", "de": "P-Wellenankunft"},
    "m_spdelay":        {"en": "S–P Delay", "it": "Ritardo S–P", "fr": "Retard S–P", "es": "Retardo S–P", "de": "S–P-Verzögerung"},
    "metadata_exp":     {"en": "Trace / Synthetic metadata", "it": "Metadati traccia / sintetici", "fr": "Métadonnées trace / synthétiques", "es": "Metadatos traza / sintéticos", "de": "Spur-/Synthetische Metadaten"},
    # ── Tab names ─────────────────────────────────────────────────────────────
    "tab_wave":         {"en": "Waveform", "it": "Forma d'onda", "fr": "Forme d'onde", "es": "Forma de onda", "de": "Wellenform"},
    "tab_fft":          {"en": "Spectral Analysis", "it": "Analisi spettrale", "fr": "Analyse spectrale", "es": "Análisis espectral", "de": "Spektralanalyse"},
    "tab_spec":         {"en": "Spectrogram", "it": "Spettrogramma", "fr": "Spectrogramme", "es": "Espectrograma", "de": "Spektrogramm"},
    "tab_stalta":       {"en": "STA/LTA", "it": "STA/LTA", "fr": "STA/LTA", "es": "STA/LTA", "de": "STA/LTA"},
    "tab_model":        {"en": "Velocity Model", "it": "Modello di velocità", "fr": "Modèle de vitesse", "es": "Modelo de velocidad", "de": "Geschwindigkeitsmodell"},
    "tab_theory":       {"en": "Theory & Math", "it": "Teoria e matematica", "fr": "Théorie & Maths", "es": "Teoría y Matemáticas", "de": "Theorie & Mathematik"},
    "tab_export":       {"en": "Export", "it": "Esporta", "fr": "Export", "es": "Exportar", "de": "Export"},
    # ── Waveform tab ──────────────────────────────────────────────────────────
    "wave_title":       {"en": "Seismic Waveform", "it": "Forma d'onda sismica", "fr": "Forme d'onde sismique", "es": "Forma de onda sísmica", "de": "Seismische Wellenform"},
    "wave_filtered":    {"en": "filtered", "it": "filtrata", "fr": "filtrée", "es": "filtrada", "de": "gefiltert"},
    "wave_unfiltered":  {"en": "unfiltered", "it": "non filtrata", "fr": "non filtrée", "es": "sin filtrar", "de": "ungefiltert"},
    "wave_p_detected":  {
        "en": "P-wave detected at <b>{t:.2f} s</b> (STA/LTA threshold {thr}).",
        "it": "Onda P rilevata a <b>{t:.2f} s</b> (soglia STA/LTA {thr}).",
        "fr": "Onde P détectée à <b>{t:.2f} s</b> (seuil STA/LTA {thr}).",
        "es": "Onda P detectada a <b>{t:.2f} s</b> (umbral STA/LTA {thr}).",
        "de": "P-Welle erkannt bei <b>{t:.2f} s</b> (STA/LTA-Schwelle {thr}).",
    },
    "wave_s_info":      {
        "en": "Synthetic S-wave arrival: <b>{s:.2f} s</b> · S–P delay = <b>{sp:.2f} s</b> → epicentral distance ≈ <b>{d:.1f} km</b> (Wadati method).",
        "it": "Arrivo onda S sintetica: <b>{s:.2f} s</b> · Ritardo S–P = <b>{sp:.2f} s</b> → distanza epicentrale ≈ <b>{d:.1f} km</b> (metodo Wadati).",
        "fr": "Arrivée onde S synthétique : <b>{s:.2f} s</b> · Retard S–P = <b>{sp:.2f} s</b> → distance épicentrale ≈ <b>{d:.1f} km</b> (méthode Wadati).",
        "es": "Llegada onda S sintética: <b>{s:.2f} s</b> · Retardo S–P = <b>{sp:.2f} s</b> → distancia epicentral ≈ <b>{d:.1f} km</b> (método Wadati).",
        "de": "Synthetische S-Wellenankunft: <b>{s:.2f} s</b> · S–P-Verzögerung = <b>{sp:.2f} s</b> → Epizentraldistanz ≈ <b>{d:.1f} km</b> (Wadati-Methode).",
    },
    "filter_details_title": {
        "en": "Zero-phase Butterworth filter — details",
        "it": "Filtro Butterworth zero-phase — dettagli",
        "fr": "Filtre Butterworth zéro-phase — détails",
        "es": "Filtro Butterworth fase cero — detalles",
        "de": "Nullphasen-Butterworth-Filter — Details",
    },
    # ── Spectral tab ──────────────────────────────────────────────────────────
    "fft_domf":         {"en": "Dominant Freq.", "it": "Freq. dominante", "fr": "Fréq. dominante", "es": "Frec. dominante", "de": "Dominanzfrequenz"},
    "fft_centroid":     {"en": "Spectral Centroid", "it": "Centroide spettrale", "fr": "Centroïde spectral", "es": "Centroide espectral", "de": "Spektralzentroid"},
    "fft_bw":           {"en": "Bandwidth (RMS)", "it": "Larghezza di banda (RMS)", "fr": "Largeur de bande (RMS)", "es": "Ancho de banda (RMS)", "de": "Bandbreite (RMS)"},
    "phase_info":       {
        "en": "The phase spectrum ∠X(f) is the argument of the complex FFT coefficient X(f) = |X(f)| · exp(j·φ(f)). Phase is mostly random for noise; coherent phases indicate structured arrivals or dispersive wave trains.",
        "it": "Lo spettro di fase ∠X(f) è l'argomento del coefficiente FFT complesso X(f) = |X(f)| · exp(j·φ(f)). La fase è casuale per il rumore; fasi coerenti indicano arrivi strutturati o treni d'onda dispersivi.",
        "fr": "Le spectre de phase ∠X(f) est l'argument du coefficient FFT complexe X(f) = |X(f)| · exp(j·φ(f)). La phase est principalement aléatoire pour le bruit ; des phases cohérentes indiquent des arrivées structurées.",
        "es": "El espectro de fase ∠X(f) es el argumento del coeficiente FFT complejo X(f) = |X(f)| · exp(j·φ(f)). La fase es aleatoria para ruido; fases coherentes indican llegadas estructuradas.",
        "de": "Das Phasenspektrum ∠X(f) ist das Argument des komplexen FFT-Koeffizienten X(f) = |X(f)| · exp(j·φ(f)). Phase ist bei Rauschen meist zufällig; kohärente Phasen zeigen strukturierte Einsätze.",
    },
    "psd_info":         {
        "en": "The Welch PSD divides the signal into overlapping segments, computes the FFT of each, and averages the squared magnitudes. This reduces variance compared to a single FFT. Units: counts²/Hz.",
        "it": "La PSD di Welch divide il segnale in segmenti sovrapposti, calcola la FFT di ciascuno e media i quadrati delle ampiezze. Riduce la varianza rispetto a una singola FFT. Unità: counts²/Hz.",
        "fr": "La DSP de Welch divise le signal en segments chevauchants, calcule la FFT de chacun et moyenne les carrés des amplitudes. Cela réduit la variance. Unités : counts²/Hz.",
        "es": "La PSD de Welch divide la señal en segmentos solapados, calcula la FFT de cada uno y promedia los cuadrados de las amplitudes. Reduce la varianza. Unidades: counts²/Hz.",
        "de": "Die Welch-PSD unterteilt das Signal in überlappende Segmente, berechnet die FFT jedes Segments und mittelt die quadrierten Amplituden. Einheit: counts²/Hz.",
    },
    # ── Spectrogram tab ───────────────────────────────────────────────────────
    "spec_info":        {
        "en": "The spectrogram shows how frequency content evolves over time. P-waves appear as a high-frequency pulse near the P-arrival; surface waves appear as a low-frequency, long-duration stripe.",
        "it": "Lo spettrogramma mostra come il contenuto in frequenza evolve nel tempo. Le onde P appaiono come un impulso ad alta frequenza vicino all'arrivo P; le onde superficiali come una striscia a bassa frequenza e lunga durata.",
        "fr": "Le spectrogramme montre l'évolution du contenu fréquentiel dans le temps. Les ondes P apparaissent comme une impulsion haute fréquence près de l'arrivée P ; les ondes de surface comme une bande basse fréquence prolongée.",
        "es": "El espectrograma muestra cómo evoluciona el contenido frecuencial en el tiempo. Las ondas P aparecen como un pulso de alta frecuencia cerca de la llegada P; las ondas superficiales como una franja de baja frecuencia.",
        "de": "Das Spektrogramm zeigt, wie sich der Frequenzinhalt über die Zeit entwickelt. P-Wellen erscheinen als hochfrequenter Impuls nahe der P-Ankunft; Oberflächenwellen als niederfrequenter, langandauernder Streifen.",
    },
    "spec_disabled":    {
        "en": "Enable the Spectrogram toggle in the sidebar.",
        "it": "Abilita l'opzione Spettrogramma nella barra laterale.",
        "fr": "Activez l'option Spectrogramme dans la barre latérale.",
        "es": "Activa la opción Espectrograma en la barra lateral.",
        "de": "Aktiviere die Spektrogramm-Option in der Seitenleiste.",
    },
    # ── STA/LTA tab ───────────────────────────────────────────────────────────
    "stalta_trigger_ok": {
        "en": "First trigger: <b>{t:.2f} s</b> · Total trigger windows: <b>{n}</b>",
        "it": "Primo trigger: <b>{t:.2f} s</b> · Finestre di trigger totali: <b>{n}</b>",
        "fr": "Premier déclenchement : <b>{t:.2f} s</b> · Fenêtres de déclenchement totales : <b>{n}</b>",
        "es": "Primer disparo: <b>{t:.2f} s</b> · Ventanas de disparo totales: <b>{n}</b>",
        "de": "Erster Trigger: <b>{t:.2f} s</b> · Trigger-Fenster gesamt: <b>{n}</b>",
    },
    "stalta_no_trigger": {
        "en": "No trigger above threshold. Try lowering the threshold or reducing the LTA window.",
        "it": "Nessun trigger sopra la soglia. Prova ad abbassare la soglia o a ridurre la finestra LTA.",
        "fr": "Aucun déclenchement au-dessus du seuil. Essayez d'abaisser le seuil ou de réduire la fenêtre LTA.",
        "es": "Ningún disparo por encima del umbral. Prueba bajando el umbral o reduciendo la ventana LTA.",
        "de": "Kein Trigger über der Schwelle. Versuche, die Schwelle zu senken oder das LTA-Fenster zu verkleinern.",
    },
    "stalta_disabled":  {
        "en": "Enable the STA/LTA chart toggle in the sidebar.",
        "it": "Abilita il grafico STA/LTA nella barra laterale.",
        "fr": "Activez le graphique STA/LTA dans la barre latérale.",
        "es": "Activa el gráfico STA/LTA en la barra lateral.",
        "de": "Aktiviere das STA/LTA-Diagramm in der Seitenleiste.",
    },
    # ── Velocity model tab ────────────────────────────────────────────────────
    "vel_layer":        {"en": "Layer", "it": "Strato", "fr": "Couche", "es": "Capa", "de": "Schicht"},
    "vel_depth":        {"en": "Depth (km)", "it": "Profondità (km)", "fr": "Profondeur (km)", "es": "Profundidad (km)", "de": "Tiefe (km)"},
    "vel_density":      {"en": "Density (g/cm³)", "it": "Densità (g/cm³)", "fr": "Densité (g/cm³)", "es": "Densidad (g/cm³)", "de": "Dichte (g/cm³)"},
    "vel_layers": {
        "en": ["Upper crust", "Middle crust", "Lower crust", "Upper mantle"],
        "it": ["Crosta superiore", "Crosta media", "Crosta inferiore", "Mantello superiore"],
        "fr": ["Croûte supérieure", "Croûte moyenne", "Croûte inférieure", "Manteau supérieur"],
        "es": ["Corteza superior", "Corteza media", "Corteza inferior", "Manto superior"],
        "de": ["Obere Kruste", "Mittlere Kruste", "Untere Kruste", "Oberer Mantel"],
    },
    # ── Export tab ────────────────────────────────────────────────────────────
    "exp_signal_title": {"en": "Processed Signal", "it": "Segnale elaborato", "fr": "Signal traité", "es": "Señal procesada", "de": "Verarbeitetes Signal"},
    "exp_signal_desc":  {"en": "Time series of the filtered waveform.", "it": "Serie temporale della forma d'onda filtrata.", "fr": "Série temporelle de la forme d'onde filtrée.", "es": "Serie temporal de la forma de onda filtrada.", "de": "Zeitreihe der gefilterten Wellenform."},
    "exp_fft_title":    {"en": "FFT Spectrum", "it": "Spettro FFT", "fr": "Spectre FFT", "es": "Espectro FFT", "de": "FFT-Spektrum"},
    "exp_fft_desc":     {"en": "Frequency, amplitude and phase.", "it": "Frequenza, ampiezza e fase.", "fr": "Fréquence, amplitude et phase.", "es": "Frecuencia, amplitud y fase.", "de": "Frequenz, Amplitude und Phase."},
    "exp_psd_title":    {"en": "Power Spectral Density", "it": "Densità spettrale di potenza", "fr": "Densité spectrale de puissance", "es": "Densidad espectral de potencia", "de": "Leistungsspektraldichte"},
    "exp_psd_desc":     {"en": "Welch PSD estimate (counts²/Hz).", "it": "Stima PSD Welch (counts²/Hz).", "fr": "Estimation DSP Welch (counts²/Hz).", "es": "Estimación PSD Welch (counts²/Hz).", "de": "Welch-PSD-Schätzung (counts²/Hz)."},
    "exp_stalta_title": {"en": "STA/LTA Ratio", "it": "Rapporto STA/LTA", "fr": "Rapport STA/LTA", "es": "Razón STA/LTA", "de": "STA/LTA-Verhältnis"},
    "dl_signal":        {"en": "Download signal CSV", "it": "Scarica segnale CSV", "fr": "Télécharger signal CSV", "es": "Descargar señal CSV", "de": "Signal CSV herunterladen"},
    "dl_fft":           {"en": "Download FFT CSV", "it": "Scarica FFT CSV", "fr": "Télécharger FFT CSV", "es": "Descargar FFT CSV", "de": "FFT CSV herunterladen"},
    "dl_psd":           {"en": "Download PSD CSV", "it": "Scarica PSD CSV", "fr": "Télécharger PSD CSV", "es": "Descargar PSD CSV", "de": "PSD CSV herunterladen"},
    "dl_stalta":        {"en": "Download STA/LTA CSV", "it": "Scarica STA/LTA CSV", "fr": "Télécharger STA/LTA CSV", "es": "Descargar STA/LTA CSV", "de": "STA/LTA CSV herunterladen"},
    "preview_title":    {"en": "Signal preview (first 200 samples)", "it": "Anteprima segnale (primi 200 campioni)", "fr": "Aperçu signal (200 premiers échantillons)", "es": "Vista previa señal (primeras 200 muestras)", "de": "Signalvorschau (erste 200 Samples)"},
}


def T(key: str, lang: str, **kwargs) -> str:
    """Translate a string key to the target language, with optional format args."""
    text = STRINGS[key].get(lang, STRINGS[key]["en"])
    if kwargs:
        # Use str.format_map with a safe dict so un-matched {placeholders} are left as-is
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SeismicLens",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — iOS-inspired glassmorphism
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: linear-gradient(145deg, #0a0e1a 0%, #0d1117 40%, #0a1628 100%);
    color: #e8edf5;
    min-height: 100vh;
}
section[data-testid="stSidebar"] {
    background: rgba(13,17,23,0.97);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container { padding: 1rem 1.2rem; }

.sl-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: border-color .25s ease;
}
.sl-card:hover { border-color: rgba(88,166,255,0.3); }

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    transition: all .2s ease;
}
.metric-card:hover {
    background: rgba(88,166,255,0.08);
    border-color: rgba(88,166,255,0.35);
    transform: translateY(-2px);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #58a6ff;
    line-height: 1;
}
.metric-unit { font-size: 11px; color: rgba(255,255,255,0.3); margin-left: 3px; }
.metric-card.green  .metric-value { color: #3fb950; }
.metric-card.amber  .metric-value { color: #d29922; }
.metric-card.violet .metric-value { color: #bc8cff; }
.metric-card.coral  .metric-value { color: #f78166; }

.hero {
    background: linear-gradient(135deg,
        rgba(88,166,255,0.12) 0%,
        rgba(63,185,80,0.06) 50%,
        rgba(188,140,255,0.08) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(88,166,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 30px; font-weight: 700; color: #e8edf5; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.hero-sub { color: rgba(255,255,255,0.45); font-size: 13px; }
.hero-badge {
    display: inline-block;
    background: rgba(88,166,255,0.15);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 10px;
}

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 6px 0 10px 0;
    margin: 28px 0 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* Guide box — step-by-step instructions */
.guide-box {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.guide-box h4 { color: #58a6ff; font-size: 13px; margin: 0 0 10px 0; font-weight: 600; }
.guide-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    line-height: 1.5;
}
.guide-num {
    background: rgba(88,166,255,0.2);
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}

.info-box {
    background: rgba(88,166,255,0.07);
    border-left: 3px solid #58a6ff;
    padding: 11px 14px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin: 10px 0;
    line-height: 1.6;
}
.warn-box {
    background: rgba(210,153,34,0.08);
    border-left: 3px solid #d29922;
    padding: 11px 14px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin: 10px 0;
}
.success-box {
    background: rgba(63,185,80,0.07);
    border-left: 3px solid #3fb950;
    padding: 11px 14px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin: 10px 0;
}
.math-block {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: #3fb950;
    margin: 10px 0;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.7;
}
.theory-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.theory-title { font-size: 13px; font-weight: 600; color: #d29922; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.theory-body  { font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.75; }

.vel-table { width: 100%; border-collapse: separate; border-spacing: 0 4px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.vel-table th { color: rgba(255,255,255,0.3); text-transform: uppercase; font-size: 10px; letter-spacing: 1px; padding: 6px 10px; text-align: left; }
.vel-table td { background: rgba(255,255,255,0.03); padding: 8px 10px; color: rgba(255,255,255,0.65); border: 1px solid rgba(255,255,255,0.05); }
.vel-table td:first-child { border-radius: 8px 0 0 8px; }
.vel-table td:last-child  { border-radius: 0 8px 8px 0; }
.vp { color: #58a6ff; } .vs { color: #3fb950; }

.sb-section {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.25);
    text-transform: uppercase;
    letter-spacing: 1.8px;
    padding: 14px 0 6px 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 6px;
}
label { color: rgba(255,255,255,0.55) !important; font-size: 12px !important; }
.js-plotly-plot { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); }
.stDownloadButton > button {
    background: rgba(88,166,255,0.12) !important;
    border: 1px solid rgba(88,166,255,0.3) !important;
    color: #58a6ff !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    transition: all .2s ease !important;
    padding: 8px 16px !important;
}
.stDownloadButton > button:hover {
    background: rgba(88,166,255,0.22) !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] { border-radius: 9px; font-size: 13px; color: rgba(255,255,255,0.5) !important; padding: 6px 18px; }
.stTabs [aria-selected="true"] { background: rgba(88,166,255,0.15) !important; color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
_PLOT_BASE = dict(
    paper_bgcolor="rgba(13,17,23,0.0)",
    plot_bgcolor="rgba(0,0,0,0.25)",
    font=dict(family="JetBrains Mono, monospace", color="rgba(255,255,255,0.5)", size=11),
    margin=dict(l=56, r=20, t=44, b=44),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
               linecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
               linecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10)),
)


def _title(text):
    return dict(text=text, font=dict(color="rgba(255,255,255,0.8)", size=13), x=0.0, xanchor="left")


def make_waveform_fig(t, signal, raw=None, p_time=None, s_time=None, title="Waveform"):
    fig = go.Figure()
    if raw is not None:
        fig.add_trace(go.Scatter(x=t, y=raw, mode="lines",
                                 line=dict(color="rgba(255,255,255,0.15)", width=0.9),
                                 name="Raw", opacity=0.6))
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines",
                             line=dict(color="#58a6ff", width=1.4), name="Signal"))
    if p_time is not None:
        fig.add_vline(x=p_time, line_width=2, line_dash="dash", line_color="#f78166",
                      annotation_text="  P", annotation_font_color="#f78166", annotation_font_size=12)
    if s_time is not None:
        fig.add_vline(x=s_time, line_width=2, line_dash="dash", line_color="#3fb950",
                      annotation_text="  S", annotation_font_color="#3fb950", annotation_font_size=12)
    fig.update_layout(**_PLOT_BASE, title=_title(title),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (counts)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
    return fig


def make_fft_fig(freqs, amplitudes, dominant_f, centroid_f=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=amplitudes, mode="lines", fill="tozeroy",
                             line=dict(color="#3fb950", width=1.6),
                             fillcolor="rgba(63,185,80,0.07)", name="|A(f)|"))
    fig.add_vline(x=dominant_f, line_width=1.8, line_dash="dot", line_color="#d29922",
                  annotation_text=f"  Peak {dominant_f:.2f} Hz",
                  annotation_font_color="#d29922", annotation_font_size=11)
    if centroid_f is not None:
        fig.add_vline(x=centroid_f, line_width=1.4, line_dash="dash", line_color="#bc8cff",
                      annotation_text=f"  Centroid {centroid_f:.2f} Hz",
                      annotation_font_color="#bc8cff", annotation_font_size=11,
                      annotation_position="bottom right")
    fig.update_layout(**_PLOT_BASE, title=_title("FFT — Amplitude Spectrum"),
                      xaxis_title="Frequency (Hz)", yaxis_title="|A(f)|  (counts)")
    return fig


def make_phase_fig(freqs, phases_deg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=phases_deg, mode="lines",
                             line=dict(color="#bc8cff", width=1.2), name="Phase"))
    fig.update_layout(**_PLOT_BASE, title=_title("FFT — Phase Spectrum  ∠X(f)"),
                      xaxis_title="Frequency (Hz)", yaxis_title="Phase (deg)")
    return fig


def make_psd_fig(f, psd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=10 * np.log10(psd + 1e-20), mode="lines", fill="tozeroy",
                             line=dict(color="#d29922", width=1.4),
                             fillcolor="rgba(210,153,34,0.06)", name="PSD"))
    fig.update_layout(**_PLOT_BASE, title=_title("Welch PSD  S(f)"),
                      xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB re 1 count²/Hz)")
    return fig


def make_spectrogram_fig(t, f, Sxx):
    z = 10 * np.log10(Sxx + 1e-20)
    fig = go.Figure(go.Heatmap(x=t, y=f, z=z, colorscale="Inferno",
                               colorbar=dict(title="dB", tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
                               zmin=np.percentile(z, 5), zmax=np.percentile(z, 99)))
    fig.update_layout(**_PLOT_BASE, title=_title("Spectrogram (STFT)"),
                      xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    return fig


def make_stalta_fig(t, stalta, threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=stalta, mode="lines",
                             line=dict(color="#bc8cff", width=1.4), name="STA/LTA"))
    trigger_mask = stalta > threshold
    if trigger_mask.any():
        y_fill = np.where(trigger_mask, stalta, np.nan)
        fig.add_trace(go.Scatter(x=t, y=y_fill, mode="lines", fill="tozeroy",
                                 line=dict(color="rgba(0,0,0,0)", width=0),
                                 fillcolor="rgba(247,129,102,0.12)",
                                 name="Triggered", showlegend=False))
    fig.add_hline(y=threshold, line_width=1.6, line_dash="dash", line_color="#f78166",
                  annotation_text=f"  Threshold = {threshold}",
                  annotation_font_color="#f78166", annotation_font_size=11)
    fig.update_layout(**_PLOT_BASE, title=_title("STA/LTA Characteristic Function"),
                      xaxis_title="Time (s)", yaxis_title="STA/LTA ratio")
    return fig


def make_velocity_model_fig(layer_names):
    vp_vals = [l[2] for l in CRUSTAL_LAYERS]
    vs_vals = [l[3] for l in CRUSTAL_LAYERS]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=layer_names, x=vp_vals, orientation="h", name="Vp (km/s)",
                         marker_color="#58a6ff", text=[f"{v} km/s" for v in vp_vals], textposition="auto"))
    fig.add_trace(go.Bar(y=layer_names, x=vs_vals, orientation="h", name="Vs (km/s)",
                         marker_color="#3fb950", text=[f"{v} km/s" for v in vs_vals], textposition="auto"))
    fig.update_layout(**_PLOT_BASE, title=_title("Crustal Velocity Model (IASP91-like)"),
                      xaxis_title="Velocity (km/s)", barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — language selector first, then controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SeismicLens")
    st.markdown("<div style='font-size:11px;color:rgba(255,255,255,0.3);margin-bottom:6px;'>v2.0</div>",
                unsafe_allow_html=True)

    lang_label = st.selectbox(
        "Language / Lingua / Langue / Idioma / Sprache",
        list(LANGUAGES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    lang = LANGUAGES[lang_label]

    # ── Data source ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('data_source', lang)}</div>", unsafe_allow_html=True)
    src_options = [
        T("src_synthetic", lang),
        T("src_mseed", lang),
        T("src_csv", lang),
    ]
    data_source_label = st.radio("source", src_options, index=0, label_visibility="collapsed")
    data_source = src_options.index(data_source_label)  # 0=synthetic, 1=mseed, 2=csv

    uploaded_file = None
    if data_source == 1:
        uploaded_file = st.file_uploader("MiniSEED (.mseed)", type=["mseed"])
        st.markdown(f"<div class='info-box' style='font-size:11px;'>{T('mseed_hint', lang)}</div>",
                    unsafe_allow_html=True)
    elif data_source == 2:
        uploaded_file = st.file_uploader("CSV", type=["csv"])
        st.markdown(f"<div class='info-box' style='font-size:11px;'>{T('csv_hint', lang)}</div>",
                    unsafe_allow_html=True)

    # ── Synthetic ────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('syn_section', lang)}</div>", unsafe_allow_html=True)
    syn_magnitude = st.slider(T("magnitude", lang), 2.0, 8.0, 5.5, 0.5)
    syn_depth     = st.slider(T("depth", lang), 5, 200, 30, 5)
    syn_dist      = st.slider(T("distance", lang), 10, 500, 80, 10)
    syn_noise     = st.slider(T("noise", lang), 0.0, 1.0, 0.12, 0.01)
    syn_duration  = st.slider(T("duration", lang), 30, 300, 120, 10)

    # ── Filter ───────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('filter_section', lang)}</div>", unsafe_allow_html=True)
    filter_on    = st.toggle(T("filter_toggle", lang), value=True)
    f_low        = st.slider(T("f_low", lang), 0.1, 10.0, 1.0, 0.1)
    f_high       = st.slider(T("f_high", lang), 1.0, 50.0, 10.0, 0.5)
    filter_order = st.selectbox(T("filter_order", lang), [2, 4, 6, 8], index=1)

    # ── STA/LTA ──────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('stalta_section', lang)}</div>", unsafe_allow_html=True)
    sta_len   = st.slider(T("sta_window", lang), 0.2, 5.0, 0.5, 0.1)
    lta_len   = st.slider(T("lta_window", lang), 5.0, 60.0, 20.0, 1.0)
    threshold = st.slider(T("threshold", lang), 1.0, 15.0, 3.5, 0.5)

    # ── Display ──────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('display_section', lang)}</div>", unsafe_allow_html=True)
    show_raw         = st.toggle(T("show_raw", lang), value=False)
    show_phase       = st.toggle(T("show_phase", lang), value=False)
    show_psd_toggle  = st.toggle(T("show_psd", lang), value=True)
    show_spec_toggle = st.toggle(T("show_spec", lang), value=True)
    show_stalta_tog  = st.toggle(T("show_stalta", lang), value=True)

    st.markdown("<div class='sb-section'>About</div>", unsafe_allow_html=True)
    st.caption("SeismicLens v2.0 · ObsPy · SciPy · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Load / generate data
# ─────────────────────────────────────────────────────────────────────────────
signal_raw = None
fs         = 100.0
metadata   = {}

if data_source == 0:
    signal_raw, fs, metadata = generate_synthetic_quake(
        magnitude=syn_magnitude, depth_km=syn_depth, dist_km=syn_dist,
        noise_level=syn_noise, duration_s=syn_duration,
    )
elif data_source == 1 and uploaded_file:
    try:
        signal_raw, fs, metadata = load_mseed(uploaded_file)
    except Exception as e:
        st.error(f"Error loading MiniSEED: {e}")
elif data_source == 2 and uploaded_file:
    try:
        signal_raw, fs, metadata = load_csv_signal(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
  <h1>SeismicLens</h1>
  <div class='hero-sub'>{T("hero_sub", lang)}</div>
  <div class='hero-badge'>Geophysics · FFT · STA/LTA · Butterworth · MiniSEED · IRIS / INGV</div>
</div>
""", unsafe_allow_html=True)

# ── Quick-start guide ─────────────────────────────────────────────────────────
_GUIDE = {
    "en": {
        "title": "How to use SeismicLens",
        "steps": [
            ("1", "<b>Choose a data source</b> in the sidebar: generate a synthetic earthquake, upload a MiniSEED file from IRIS/INGV, or upload a CSV."),
            ("2", "<b>Configure the Butterworth bandpass filter</b> (low cut, high cut, order). Toggle it on or off to compare filtered vs raw signal."),
            ("3", "<b>Adjust the STA/LTA detector</b> parameters (STA window, LTA window, threshold) to locate the automatic P-wave pick."),
            ("4", "<b>Explore the tabs</b>: Waveform, Spectral Analysis (FFT + PSD), Spectrogram, STA/LTA, Velocity Model, Theory & Math."),
            ("5", "<b>Export</b> the processed signal, FFT spectrum, PSD and STA/LTA ratio as CSV files."),
        ],
    },
    "it": {
        "title": "Come usare SeismicLens",
        "steps": [
            ("1", "<b>Scegli la sorgente dati</b> nella barra laterale: genera un terremoto sintetico, carica un file MiniSEED da IRIS/INGV, o carica un CSV."),
            ("2", "<b>Configura il filtro Butterworth bandpass</b> (taglio basso, taglio alto, ordine). Attivalo o disattivalo per confrontare segnale filtrato e grezzo."),
            ("3", "<b>Regola il detector STA/LTA</b> (finestra STA, finestra LTA, soglia) per individuare automaticamente l'onda P."),
            ("4", "<b>Esplora le schede</b>: Forma d'onda, Analisi spettrale (FFT + PSD), Spettrogramma, STA/LTA, Modello di velocità, Teoria e matematica."),
            ("5", "<b>Esporta</b> il segnale elaborato, lo spettro FFT, la PSD e il rapporto STA/LTA come file CSV."),
        ],
    },
    "fr": {
        "title": "Comment utiliser SeismicLens",
        "steps": [
            ("1", "<b>Choisissez une source de données</b> dans la barre latérale : générez un séisme synthétique, importez un fichier MiniSEED depuis IRIS/INGV, ou importez un CSV."),
            ("2", "<b>Configurez le filtre Butterworth passe-bande</b> (coupure basse, coupure haute, ordre). Activez-le ou désactivez-le pour comparer signal filtré et brut."),
            ("3", "<b>Ajustez le détecteur STA/LTA</b> (fenêtre STA, fenêtre LTA, seuil) pour localiser automatiquement l'onde P."),
            ("4", "<b>Explorez les onglets</b> : Forme d'onde, Analyse spectrale (FFT + PSD), Spectrogramme, STA/LTA, Modèle de vitesse, Théorie et maths."),
            ("5", "<b>Exportez</b> le signal traité, le spectre FFT, la PSD et le rapport STA/LTA en fichiers CSV."),
        ],
    },
    "es": {
        "title": "Cómo usar SeismicLens",
        "steps": [
            ("1", "<b>Elige una fuente de datos</b> en la barra lateral: genera un sismo sintético, carga un archivo MiniSEED de IRIS/INGV, o carga un CSV."),
            ("2", "<b>Configura el filtro Butterworth pasa-banda</b> (corte bajo, corte alto, orden). Actívalo o desactívalo para comparar señal filtrada y bruta."),
            ("3", "<b>Ajusta el detector STA/LTA</b> (ventana STA, ventana LTA, umbral) para localizar automáticamente la onda P."),
            ("4", "<b>Explora las pestañas</b>: Forma de onda, Análisis espectral (FFT + PSD), Espectrograma, STA/LTA, Modelo de velocidad, Teoría y Matemáticas."),
            ("5", "<b>Exporta</b> la señal procesada, el espectro FFT, la PSD y la razón STA/LTA como archivos CSV."),
        ],
    },
    "de": {
        "title": "So benutze SeismicLens",
        "steps": [
            ("1", "<b>Wähle eine Datenquelle</b> in der Seitenleiste: synthetisches Erdbeben generieren, MiniSEED-Datei von IRIS/INGV laden oder CSV hochladen."),
            ("2", "<b>Konfiguriere den Butterworth-Bandpassfilter</b> (untere/obere Grenzfrequenz, Ordnung). Ein-/ausschalten um gefiltertes und Rohsignal zu vergleichen."),
            ("3", "<b>Passe den STA/LTA-Detektor an</b> (STA-Fenster, LTA-Fenster, Schwelle) um den P-Wellen-Einsatz automatisch zu bestimmen."),
            ("4", "<b>Erkunde die Tabs</b>: Wellenform, Spektralanalyse (FFT + PSD), Spektrogramm, STA/LTA, Geschwindigkeitsmodell, Theorie & Mathematik."),
            ("5", "<b>Exportiere</b> das verarbeitete Signal, das FFT-Spektrum, die PSD und das STA/LTA-Verhältnis als CSV-Dateien."),
        ],
    },
}

guide = _GUIDE[lang]
steps_html = "".join(
    f"<div class='guide-step'><div class='guide-num'>{n}</div><div>{txt}</div></div>"
    for n, txt in guide["steps"]
)
st.markdown(
    f"<div class='guide-box'><h4>{guide['title']}</h4>{steps_html}</div>",
    unsafe_allow_html=True,
)

if signal_raw is None:
    st.markdown(f"<div class='warn-box'>{T('no_data', lang)}</div>", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
signal_tapered = taper_signal(signal_raw)
if filter_on:
    try:
        signal_proc = bandpass_filter(signal_tapered, f_low, f_high, fs, order=filter_order)
    except Exception as e:
        st.warning(f"Filter error: {e}")
        signal_proc = signal_tapered
else:
    signal_proc = signal_tapered

N = len(signal_proc)
t = np.linspace(0, N / fs, N)

freqs, amplitudes, phases_deg = compute_fft(signal_proc, fs)
dominant_f, centroid_f, bandwidth = spectral_metrics(freqs, amplitudes)
f_psd, psd = compute_psd(signal_proc, fs)

p_time = detect_p_wave(signal_proc, fs, sta_len, lta_len, threshold)
s_time = metadata.get("s_arrival_s") if data_source == 0 else None

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, unit="", color=""):
    return (f"<div class='metric-card {color}'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}<span class='metric-unit'>{unit}</span></div>"
            f"</div>")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(metric_card(T("m_fs", lang),    f"{fs:.0f}",                          "Hz"),         unsafe_allow_html=True)
c2.markdown(metric_card(T("m_dur", lang),   f"{N/fs:.1f}",                         "s", "green"), unsafe_allow_html=True)
c3.markdown(metric_card(T("m_peak", lang),  f"{np.max(np.abs(signal_proc)):.0f}", "cts", "amber"),unsafe_allow_html=True)
c4.markdown(metric_card(T("m_domf", lang),  f"{dominant_f:.2f}",                  "Hz", "violet"),unsafe_allow_html=True)
c5.markdown(metric_card(T("m_parr", lang),  f"{p_time:.2f}" if p_time else "—",   "s", "coral"), unsafe_allow_html=True)
sp_val = f"{(s_time - p_time):.2f}" if (s_time and p_time) else "—"
c6.markdown(metric_card(T("m_spdelay", lang), sp_val, "s"), unsafe_allow_html=True)

if metadata:
    with st.expander(T("metadata_exp", lang), expanded=False):
        st.json(metadata)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_wave, tab_fft, tab_spec, tab_stalta, tab_model, tab_theory, tab_export = st.tabs([
    T("tab_wave",   lang),
    T("tab_fft",    lang),
    T("tab_spec",   lang),
    T("tab_stalta", lang),
    T("tab_model",  lang),
    T("tab_theory", lang),
    T("tab_export", lang),
])

# ── Waveform ──────────────────────────────────────────────────────────────────
with tab_wave:
    st.markdown(f"<div class='section-title'>{T('tab_wave', lang)}</div>", unsafe_allow_html=True)
    raw_ov = signal_tapered if show_raw and filter_on else None
    filt_label = T("wave_filtered", lang) if filter_on else T("wave_unfiltered", lang)
    fig_wave = make_waveform_fig(t, signal_proc, raw=raw_ov, p_time=p_time, s_time=s_time,
                                 title=f"{T('wave_title', lang)} ({filt_label})")
    st.plotly_chart(fig_wave, use_container_width=True)

    if p_time:
        st.markdown(
            f"<div class='success-box'>{T('wave_p_detected', lang, t=p_time, thr=threshold)}</div>",
            unsafe_allow_html=True,
        )
    if s_time and data_source == 0:
        sp_delay = s_time - (p_time or 0)
        st.markdown(
            f"<div class='info-box'>{T('wave_s_info', lang, s=s_time, sp=sp_delay, d=sp_delay*8)}</div>",
            unsafe_allow_html=True,
        )

    if filter_on:
        eff_order = 2 * filter_order
        with st.expander(T("filter_details_title", lang)):
            st.markdown(f"""
**Parameters:**
- Low cut: **{f_low} Hz** | High cut: **{f_high} Hz**
- Order: **{filter_order}** (effective {eff_order} with zero-phase)
- Nyquist: **{fs/2:.1f} Hz**
- Implementation: `scipy.signal.sosfiltfilt` — Second-Order Sections, forward + backward pass

**Why zero-phase?**
A causal filter delays different frequencies by different amounts, distorting wave arrival times.
`sosfiltfilt` runs the filter forward then backward, cancelling all phase shift exactly.
""")


# ── Spectral Analysis ─────────────────────────────────────────────────────────
with tab_fft:
    st.markdown(f"<div class='section-title'>{T('tab_fft', lang)}</div>", unsafe_allow_html=True)
    cf1, cf2, cf3 = st.columns(3)
    cf1.markdown(metric_card(T("fft_domf", lang),    f"{dominant_f:.3f}", "Hz", "green"),  unsafe_allow_html=True)
    cf2.markdown(metric_card(T("fft_centroid", lang), f"{centroid_f:.3f}", "Hz", "violet"), unsafe_allow_html=True)
    cf3.markdown(metric_card(T("fft_bw", lang),       f"{bandwidth:.3f}",  "Hz", "amber"),  unsafe_allow_html=True)

    st.plotly_chart(make_fft_fig(freqs, amplitudes, dominant_f, centroid_f), use_container_width=True)

    if show_phase:
        st.markdown(f"<div class='section-title'>Phase Spectrum</div>", unsafe_allow_html=True)
        st.plotly_chart(make_phase_fig(freqs, phases_deg), use_container_width=True)
        st.markdown(f"<div class='info-box'>{T('phase_info', lang)}</div>", unsafe_allow_html=True)

    if show_psd_toggle:
        st.markdown(f"<div class='section-title'>Power Spectral Density — Welch</div>", unsafe_allow_html=True)
        st.plotly_chart(make_psd_fig(f_psd, psd), use_container_width=True)
        st.markdown(f"<div class='info-box'>{T('psd_info', lang)}</div>", unsafe_allow_html=True)


# ── Spectrogram ───────────────────────────────────────────────────────────────
with tab_spec:
    if show_spec_toggle:
        st.markdown(f"<div class='section-title'>{T('tab_spec', lang)} — STFT</div>", unsafe_allow_html=True)
        try:
            t_spec, f_spec, Sxx = compute_spectrogram(signal_proc, fs)
            st.plotly_chart(make_spectrogram_fig(t_spec, f_spec, Sxx), use_container_width=True)
            st.markdown(f"<div class='info-box'>{T('spec_info', lang)}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Spectrogram error: {e}")
    else:
        st.info(T("spec_disabled", lang))


# ── STA/LTA ───────────────────────────────────────────────────────────────────
with tab_stalta:
    if show_stalta_tog:
        st.markdown(f"<div class='section-title'>{T('tab_stalta', lang)}</div>", unsafe_allow_html=True)
        try:
            stalta_vals = compute_sta_lta(signal_proc, fs, sta_len, lta_len)
            t_sl = np.linspace(0, len(stalta_vals) / fs, len(stalta_vals))
            st.plotly_chart(make_stalta_fig(t_sl, stalta_vals, threshold), use_container_width=True)

            n_trig = int(np.sum(np.diff((stalta_vals > threshold).astype(int)) > 0))
            if p_time:
                st.markdown(
                    f"<div class='success-box'>{T('stalta_trigger_ok', lang, t=p_time, n=n_trig)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='warn-box'>{T('stalta_no_trigger', lang)}</div>", unsafe_allow_html=True)

            sta_n = int(sta_len * fs)
            lta_n = int(lta_len * fs)
            with st.expander("STA/LTA — algorithm details"):
                st.markdown(f"""
**STA window:** {sta_len} s ({sta_n} samples) &nbsp;|&nbsp;
**LTA window:** {lta_len} s ({lta_n} samples) &nbsp;|&nbsp;
**Threshold:** {threshold}

```
STA(t) = mean( x[t - {sta_n} : t]^2 )
LTA(t) = mean( x[t - {lta_n} : t]^2 )
R(t)   = STA(t) / LTA(t)   ->  trigger when R > {threshold}
```

Implementation: O(N) via prefix sums on squared samples — `cs = cumsum(x^2)`.  
Reference: Allen (1978), *BSSA*.
""")
        except Exception as e:
            st.warning(f"STA/LTA error: {e}")
    else:
        st.info(T("stalta_disabled", lang))


# ── Velocity Model ────────────────────────────────────────────────────────────
with tab_model:
    st.markdown(f"<div class='section-title'>{T('tab_model', lang)}</div>", unsafe_allow_html=True)

    layer_names = T("vel_layers", lang)
    table_html = (
        f"<table class='vel-table'><tr>"
        f"<th>{T('vel_layer', lang)}</th>"
        f"<th>{T('vel_depth', lang)}</th>"
        f"<th>Vp (km/s)</th><th>Vs (km/s)</th>"
        f"<th>Vp/Vs</th><th>Poisson ν</th>"
        f"<th>{T('vel_density', lang)}</th></tr>"
    )
    for (z0, z1, vp, vs, rho), name in zip(CRUSTAL_LAYERS, layer_names):
        ratio = vp / vs
        r2 = ratio ** 2
        nu = (r2 - 2) / (2 * r2 - 2)
        table_html += (
            f"<tr><td>{name}</td><td>{z0}–{z1}</td>"
            f"<td class='vp'>{vp}</td><td class='vs'>{vs}</td>"
            f"<td style='color:#d29922'>{ratio:.3f}</td>"
            f"<td style='color:#bc8cff'>{nu:.3f}</td>"
            f"<td>{rho}</td></tr>"
        )
    table_html += "</table>"
    st.markdown(f"<div class='sl-card'>{table_html}</div>", unsafe_allow_html=True)

    st.plotly_chart(make_velocity_model_fig(layer_names), use_container_width=True)

    st.markdown("""
    <div class='sl-card'>
    <div class='theory-title'>Vp/Vs ratio and Poisson's ratio</div>
    <div class='theory-body'>
    The ratio Vp/Vs is a key diagnostic in seismology. For a Poisson solid (ν = 0.25):
    Vp/Vs = √3 ≈ 1.732. Values above 1.8 can indicate fluid saturation or partial melting.
    </div>
    <div class='math-block'>ν = (Vp² − 2·Vs²) / (2·Vp² − 2·Vs²)

Vp = sqrt[(K + 4G/3) / ρ]    (K = bulk modulus, G = shear modulus, ρ = density)
Vs = sqrt[G / ρ]              (Vs = 0 in fluids since G = 0)</div>
    </div>""", unsafe_allow_html=True)


# ── Theory & Math ─────────────────────────────────────────────────────────────
with tab_theory:
    st.markdown(f"<div class='section-title'>{T('tab_theory', lang)}</div>", unsafe_allow_html=True)

    with st.expander("Discrete Fourier Transform (DFT) & Complex Numbers", expanded=True):
        st.markdown("""
        <div class='theory-card'>
        <div class='theory-title'>Euler's formula and complex exponentials</div>
        <div class='theory-body'>
        The DFT decomposes a discrete signal x[n] (n = 0, …, N−1) into N complex sinusoids.
        Each basis function is a complex exponential rooted in Euler's formula:
        </div>
        <div class='math-block'>e^(jθ) = cos(θ) + j·sin(θ)          (Euler's formula)

X[k] = sum_{n=0}^{N-1}  x[n] · e^(−j·2π·k·n/N)

Each X[k] is a complex number  X[k] ∈ ℂ:
  |X[k]|      →  amplitude at frequency  f_k = k·fs/N  (Hz)
  ∠X[k]       →  phase  =  arg(X[k])  =  atan2(Im, Re)
  X[k].real   →  cosine component  (in-phase)
  X[k].imag   →  sine component    (quadrature)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='theory-card'>
        <div class='theory-title'>Fast Fourier Transform (FFT) — O(N log N)</div>
        <div class='theory-body'>
        The FFT (Cooley & Tukey, 1965) reduces the naive O(N²) DFT to O(N log N) by
        recursively splitting the sum into even and odd indices (decimation-in-time).
        For real-valued signals the spectrum is Hermitian — X[N−k] = conj(X[k]) —
        so only N/2+1 unique bins exist. This is exploited by <code>scipy.fft.rfft</code>.<br><br>
        Hann window reduces spectral leakage from non-periodic signals:
        </div>
        <div class='math-block'>w[n] = 0.5 · (1 − cos(2π·n / (N−1)))     (Hann window)

One-sided amplitude spectrum (after window correction):
  A[k] = (2/N) · |X[k]|     for k = 1 … N/2−1
  A[0] = (1/N) · |X[0]|     (DC component, factor 1 not 2)

Frequency resolution:  Δf = fs / N  (Hz per bin)</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Seismic Wave Physics"):
        st.markdown("""
        <div class='theory-card'>
        <div class='theory-title'>Body waves and elastic moduli</div>
        <div class='theory-body'>
        Seismic waves propagate by elastic deformation. Two body-wave types exist:
        </div>
        <div class='math-block'>P-wave (compressional, fastest):
  Vp = sqrt[(K + 4G/3) / ρ]   →  5.8–8.0 km/s in the crust/mantle

S-wave (shear, slower):
  Vs = sqrt[G / ρ]             →  3.4–4.5 km/s
  Vs = 0 in liquids  (G = 0 → shear modulus vanishes)

Surface waves (Rayleigh, Love) — dispersive, slower than S:
  V_Rayleigh ≈ 0.92 · Vs
  V_Love     ≈ Vs  (frequency-dependent)</div>
        </div>
        <div class='theory-card'>
        <div class='theory-title'>Travel time and Wadati method</div>
        <div class='theory-body'>
        For a local earthquake at focal depth h and epicentral distance d:
        </div>
        <div class='math-block'>Hypocentral distance:   R = sqrt(d² + h²)

Travel times:   t_P = R / Vp
                t_S = R / Vs

Wadati (1933) — epicentral distance from S−P delay:
  d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Butterworth Filter Design"):
        eff = 2 * filter_order
        rolloff = 20 * filter_order
        eff_rolloff = 40 * filter_order
        st.markdown(f"""
        <div class='theory-card'>
        <div class='theory-title'>Transfer function and frequency response</div>
        <div class='theory-body'>
        The Butterworth filter is maximally flat in the passband — no ripple.
        Its poles lie on a circle of radius ω_c in the s-plane.
        Current settings: order {filter_order} → effective order {eff} (zero-phase).
        </div>
        <div class='math-block'>|H(jω)|² = 1 / [1 + (ω/ω_c)^(2·{filter_order})]

Roll-off beyond ω_c:  20·{filter_order} = {rolloff} dB/decade  (one-pass)
With sosfiltfilt (zero-phase, effective order {eff}):
  roll-off = 40·{filter_order} = {eff_rolloff} dB/decade

Band-pass = product of low-pass × high-pass transfer functions.
SOS form (Second-Order Sections) for numerical stability:
  H(z) = product_k  (b0_k + b1_k·z⁻¹ + b2_k·z⁻²)
                   / (1   + a1_k·z⁻¹ + a2_k·z⁻²)</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("STA/LTA Algorithm"):
        st.markdown(f"""
        <div class='theory-card'>
        <div class='theory-title'>Short-Term vs Long-Term Average ratio</div>
        <div class='theory-body'>
        Proposed by Allen (1978). The ratio spikes when a seismic phase arrives because
        short-term energy rises sharply while long-term energy reacts slowly.
        Squaring the samples tracks energy regardless of polarity.
        </div>
        <div class='math-block'>STA(t) = (1/N_sta) · sum_{{k=0}}^{{N_sta-1}}  x²[t−k]
LTA(t) = (1/N_lta) · sum_{{k=0}}^{{N_lta-1}}  x²[t−k]
R(t)   = STA(t) / LTA(t)        trigger when R > {threshold}

Current:  N_sta = {int(sta_len*fs)} samples ({sta_len} s)
          N_lta = {int(lta_len*fs)} samples ({lta_len} s)

O(N) implementation with prefix sums:
  cs     = cumsum(x²)                       (precomputed once)
  STA(t) = (cs[t+1] - cs[t-N_sta+1]) / N_sta</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Spectrogram & Time-Frequency Uncertainty"):
        st.markdown("""
        <div class='theory-card'>
        <div class='theory-title'>Short-Time Fourier Transform (STFT)</div>
        <div class='theory-body'>
        The STFT slides a window w[m] along the signal and computes the DFT of each frame,
        producing a 2D time–frequency representation.
        </div>
        <div class='math-block'>STFT(τ, f) = sum_n  x[n] · w[n−τ] · e^(−j·2π·f·n/fs)

Uncertainty principle (analogous to Heisenberg):
  Δt · Δf  ≥  1/(4π)

Wide window  →  fine frequency resolution, coarse time resolution
Narrow window →  fine time resolution,      coarse frequency resolution

Welch PSD  =  average of |STFT(τ,f)|² over all K frames
           →  variance reduced by factor √K vs single FFT</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Magnitude Scales (Richter & Moment)"):
        st.markdown("""
        <div class='theory-card'>
        <div class='theory-title'>Local magnitude and seismic moment</div>
        <div class='theory-body'>
        Richter (1935) defined local magnitude from the peak ground motion on a
        Wood-Anderson seismograph, corrected for distance:
        </div>
        <div class='math-block'>M_L = log10(A) − log10(A_0(Δ))
  A     = peak amplitude on Wood-Anderson instrument (μm)
  A_0   = empirical distance correction

SeismicLens synthetic amplitude scaling:
  A_peak  ∝  10^(0.8·M − 2.5)

Moment magnitude M_w (Hanks & Kanamori, 1979):
  M_w = (2/3) · log10(M_0) − 10.7
  M_0 = μ · A_fault · D       (N·m)
  μ   = shear modulus, A_fault = rupture area, D = average slip</div>
        </div>""", unsafe_allow_html=True)


# ── Export ────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown(f"<div class='section-title'>{T('tab_export', lang)}</div>", unsafe_allow_html=True)

    ce1, ce2, ce3 = st.columns(3)

    with ce1:
        st.markdown(
            f"<div class='sl-card'><div class='theory-title'>{T('exp_signal_title', lang)}</div>"
            f"<div class='theory-body'>{T('exp_signal_desc', lang)}</div></div>",
            unsafe_allow_html=True,
        )
        buf_sig = io.StringIO()
        pd.DataFrame({"time_s": t, "amplitude_counts": signal_proc}).to_csv(buf_sig, index=False)
        st.download_button(T("dl_signal", lang), buf_sig.getvalue(),
                           file_name="seismiclens_signal.csv", mime="text/csv")

    with ce2:
        st.markdown(
            f"<div class='sl-card'><div class='theory-title'>{T('exp_fft_title', lang)}</div>"
            f"<div class='theory-body'>{T('exp_fft_desc', lang)}</div></div>",
            unsafe_allow_html=True,
        )
        buf_fft = io.StringIO()
        pd.DataFrame({"frequency_hz": freqs, "amplitude": amplitudes,
                      "phase_deg": phases_deg}).to_csv(buf_fft, index=False)
        st.download_button(T("dl_fft", lang), buf_fft.getvalue(),
                           file_name="seismiclens_fft.csv", mime="text/csv")

    with ce3:
        st.markdown(
            f"<div class='sl-card'><div class='theory-title'>{T('exp_psd_title', lang)}</div>"
            f"<div class='theory-body'>{T('exp_psd_desc', lang)}</div></div>",
            unsafe_allow_html=True,
        )
        buf_psd = io.StringIO()
        pd.DataFrame({"frequency_hz": f_psd, "psd_counts2_per_hz": psd,
                      "psd_db": 10 * np.log10(psd + 1e-20)}).to_csv(buf_psd, index=False)
        st.download_button(T("dl_psd", lang), buf_psd.getvalue(),
                           file_name="seismiclens_psd.csv", mime="text/csv")

    try:
        stalta_exp = compute_sta_lta(signal_proc, fs, sta_len, lta_len)
        buf_sl = io.StringIO()
        pd.DataFrame({"time_s": np.linspace(0, len(stalta_exp)/fs, len(stalta_exp)),
                      "stalta_ratio": stalta_exp}).to_csv(buf_sl, index=False)
        ce1.download_button(T("dl_stalta", lang), buf_sl.getvalue(),
                            file_name="seismiclens_stalta.csv", mime="text/csv")
    except Exception:
        pass

    st.markdown(f"<div class='section-title'>{T('preview_title', lang)}</div>", unsafe_allow_html=True)
    df_prev = pd.DataFrame({"time_s": t[:200], "amplitude": signal_proc[:200]})
    st.dataframe(df_prev.style.format({"time_s": "{:.4f}", "amplitude": "{:.2f}"}),
                 height=260, use_container_width=True)

