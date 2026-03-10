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
    # ── Section descriptions ─────────────────────────────────────────────────
    "wave_desc": {
        "en": "The waveform plot shows amplitude (ground motion in counts) vs time. P-waves arrive first (compressional), S-waves second (shear). The Butterworth bandpass filter removes noise outside your chosen frequency band without distorting arrival times.",
        "it": "Il grafico mostra l'ampiezza (moto del suolo in counts) nel tempo. Le onde P arrivano per prime (compressive), le onde S per seconde (di taglio). Il filtro Butterworth bandpass elimina il rumore fuori dalla banda di frequenza scelta senza distorcere i tempi di arrivo.",
        "fr": "Le graphique montre l'amplitude (mouvement du sol en counts) en fonction du temps. Les ondes P arrivent en premier (compression), les ondes S en second (cisaillement). Le filtre Butterworth passe-bande supprime le bruit hors de la bande de fréquence sans distordre les temps d'arrivée.",
        "es": "El gráfico muestra la amplitud (movimiento del suelo en counts) vs tiempo. Las ondas P llegan primero (compresión), las ondas S después (cizallamiento). El filtro Butterworth pasa-banda elimina el ruido fuera de la banda de frecuencia sin distorsionar los tiempos de llegada.",
        "de": "Das Wellenformdiagramm zeigt die Amplitude (Bodenbewegung in Counts) über die Zeit. P-Wellen kommen zuerst an (Kompression), S-Wellen danach (Scherung). Der Butterworth-Bandpassfilter entfernt Rauschen außerhalb des Frequenzbandes ohne Ankunftszeiten zu verzerren.",
    },
    "fft_desc": {
        "en": "The FFT decomposes the signal into its frequency components. Each frequency bin shows how much energy is present at that frequency. The dominant frequency is the peak; the spectral centroid is the energy-weighted mean; bandwidth measures spread.",
        "it": "La FFT scompone il segnale nelle sue componenti in frequenza. Ogni bin di frequenza indica quanta energia è presente a quella frequenza. La frequenza dominante è il picco; il centroide spettrale è la media pesata per l'energia; la larghezza di banda misura la dispersione.",
        "fr": "La FFT décompose le signal en ses composantes fréquentielles. Chaque bin de fréquence indique l'énergie présente à cette fréquence. La fréquence dominante est le pic ; le centroïde spectral est la moyenne pondérée par l'énergie ; la largeur de bande mesure la dispersion.",
        "es": "La FFT descompone la señal en sus componentes de frecuencia. Cada bin muestra cuánta energía hay en esa frecuencia. La frecuencia dominante es el pico; el centroide espectral es la media ponderada por energía; el ancho de banda mide la dispersión.",
        "de": "Die FFT zerlegt das Signal in seine Frequenzkomponenten. Jeder Frequenz-Bin zeigt, wie viel Energie bei dieser Frequenz vorhanden ist. Die dominante Frequenz ist der Peak; der Spektralzentroid ist der energiegewichtete Mittelwert; die Bandbreite misst die Streuung.",
    },
    "spec_desc": {
        "en": "The spectrogram shows how the frequency content changes over time. It is computed via Short-Time Fourier Transform (STFT): the signal is split into short, overlapping windows, the FFT is computed for each, and the results are assembled into a 2D heatmap. Brighter colours = more energy.",
        "it": "Lo spettrogramma mostra come il contenuto in frequenza cambia nel tempo. È calcolato tramite STFT: il segnale è diviso in finestre brevi e sovrapposte, la FFT è calcolata per ciascuna, e i risultati formano una mappa di calore 2D. Colori più chiari = più energia.",
        "fr": "Le spectrogramme montre l'évolution du contenu fréquentiel dans le temps. Calculé via STFT : le signal est découpé en fenêtres courtes chevauchantes, la FFT est calculée pour chacune, et les résultats forment une carte de chaleur 2D. Couleurs plus claires = plus d'énergie.",
        "es": "El espectrograma muestra cómo varía el contenido frecuencial en el tiempo. Se calcula mediante STFT: el signal se divide en ventanas cortas solapadas, se calcula la FFT de cada una, y los resultados forman un mapa de calor 2D. Colores más claros = más energía.",
        "de": "Das Spektrogramm zeigt, wie sich der Frequenzinhalt über die Zeit verändert. Es wird per STFT berechnet: Das Signal wird in kurze, überlappende Fenster unterteilt, für jedes wird die FFT berechnet, und die Ergebnisse bilden eine 2D-Heatmap. Hellere Farben = mehr Energie.",
    },
    "stalta_desc": {
        "en": "STA/LTA (Short-Term Average / Long-Term Average) is the classic automatic seismic phase detector (Allen, 1978). It compares the short-term signal energy to the long-term background energy. When the ratio spikes above the threshold, a seismic phase (usually P-wave) is detected.",
        "it": "STA/LTA è il classico detector automatico di fasi sismiche (Allen, 1978). Confronta l'energia del segnale a breve termine con quella di fondo a lungo termine. Quando il rapporto supera la soglia, viene rilevata una fase sismica (di solito l'onda P).",
        "fr": "STA/LTA est le détecteur automatique classique de phases sismiques (Allen, 1978). Il compare l'énergie du signal à court terme à l'énergie de fond à long terme. Quand le rapport dépasse le seuil, une phase sismique (généralement l'onde P) est détectée.",
        "es": "STA/LTA es el detector automático clásico de fases sísmicas (Allen, 1978). Compara la energía de corto plazo del señal con la energía de fondo a largo plazo. Cuando el cociente supera el umbral, se detecta una fase sísmica (normalmente la onda P).",
        "de": "STA/LTA ist der klassische automatische seismische Phasendetektor (Allen, 1978). Er vergleicht die kurzfristige Signalenergie mit der langfristigen Hintergrundenergie. Wenn das Verhältnis den Schwellenwert überschreitet, wird eine seismische Phase (meist P-Welle) erkannt.",
    },
    "model_desc": {
        "en": "This table shows an IASP91-inspired layered crustal velocity model used for synthetic waveform generation. Each layer is defined by its P-wave velocity (Vp), S-wave velocity (Vs), and density (ρ). These properties depend on mineralogy and pressure.",
        "it": "Questa tabella mostra un modello di velocità crostale a strati ispirato a IASP91, usato per generare i segnali sintetici. Ogni strato è definito dalla velocità dell'onda P (Vp), onda S (Vs) e densità (ρ). Queste proprietà dipendono dalla mineralogia e dalla pressione.",
        "fr": "Ce tableau montre un modèle de vitesse crustale en couches inspiré d'IASP91, utilisé pour générer les formes d'onde synthétiques. Chaque couche est définie par la vitesse des ondes P (Vp), S (Vs) et la densité (ρ). Ces propriétés dépendent de la minéralogie et de la pression.",
        "es": "Esta tabla muestra un modelo de velocidad cortical en capas inspirado en IASP91, usado para generar las formas de onda sintéticas. Cada capa se define por la velocidad de onda P (Vp), onda S (Vs) y densidad (ρ). Estas propiedades dependen de la mineralogía y la presión.",
        "de": "Diese Tabelle zeigt ein IASP91-inspiriertes geschichtetes Krustengeschwindigkeitsmodell für die Synthese von Wellenformen. Jede Schicht ist durch P-Wellengeschwindigkeit (Vp), S-Wellengeschwindigkeit (Vs) und Dichte (ρ) definiert. Diese Eigenschaften hängen von der Mineralogie und dem Druck ab.",
    },
    "filter_why_title": {
        "en": "Why filter seismic signals?",
        "it": "Perché filtrare i segnali sismici?",
        "fr": "Pourquoi filtrer les signaux sismiques ?",
        "es": "¿Por qué filtrar las señales sísmicas?",
        "de": "Warum seismische Signale filtern?",
    },
    "filter_why_body": {
        "en": "Real seismograms contain instrument noise, microseismic noise (ocean waves, 0.1–0.3 Hz), cultural noise (traffic, 1–20 Hz), and electronic noise. A bandpass filter keeps only the frequency band relevant to the earthquake you are studying, improving SNR and making phase picks more accurate.",
        "it": "I sismogrammi reali contengono rumore strumentale, rumore microsiemico (onde oceaniche, 0.1–0.3 Hz), rumore culturale (traffico, 1–20 Hz) e rumore elettronico. Un filtro bandpass mantiene solo la banda di frequenza rilevante per il terremoto studiato, migliorando il SNR e rendendo più precisi i pick delle fasi.",
        "fr": "Les sismogrammes réels contiennent du bruit instrumental, du bruit microsismique (vagues océaniques, 0.1–0.3 Hz), du bruit culturel (trafic, 1–20 Hz) et du bruit électronique. Un filtre passe-bande conserve uniquement la bande de fréquence pertinente pour le séisme étudié, améliorant le SNR et la précision des détections de phases.",
        "es": "Los sismogramas reales contienen ruido instrumental, ruido microsísmico (olas oceánicas, 0.1–0.3 Hz), ruido cultural (tráfico, 1–20 Hz) y ruido electrónico. Un filtro pasa-banda conserva solo la banda de frecuencia relevante para el sismo estudiado, mejorando el SNR y la precisión de los picks de fases.",
        "de": "Echte Seismogramme enthalten Instrumentenrauschen, mikroseismisches Rauschen (Meereswellen, 0,1–0,3 Hz), kulturelles Rauschen (Verkehr, 1–20 Hz) und elektronisches Rauschen. Ein Bandpassfilter behält nur den für das untersuchte Erdbeben relevanten Frequenzbereich, verbessert das SNR und macht Phasenbestimmungen genauer.",
    },
    "export_desc": {
        "en": "Download the processed data as CSV files for further analysis in Python, MATLAB, Excel, or any other tool. Each file contains the full time series or spectrum with all computed quantities.",
        "it": "Scarica i dati elaborati come file CSV per ulteriori analisi in Python, MATLAB, Excel o qualsiasi altro strumento. Ogni file contiene la serie temporale o lo spettro completo con tutte le grandezze calcolate.",
        "fr": "Téléchargez les données traitées en fichiers CSV pour une analyse ultérieure dans Python, MATLAB, Excel ou tout autre outil. Chaque fichier contient la série temporelle ou le spectre complet avec toutes les grandeurs calculées.",
        "es": "Descarga los datos procesados como archivos CSV para análisis adicional en Python, MATLAB, Excel u otra herramienta. Cada archivo contiene la serie temporal o el espectro completo con todas las magnitudes calculadas.",
        "de": "Lade die verarbeiteten Daten als CSV-Dateien für weitere Analysen in Python, MATLAB, Excel oder anderen Tools herunter. Jede Datei enthält die vollständige Zeitreihe oder das Spektrum mit allen berechneten Größen.",
    },
    "syn_desc": {
        "en": "Generate a physically realistic synthetic seismogram using an IASP91-like crustal model. The simulator models P-waves (6–12 Hz), S-waves (2–6 Hz) and surface waves (0.3–1 Hz) with amplitudes scaled by magnitude and distances computed from travel-time equations.",
        "it": "Genera un sismogramma sintetico fisicamente realistico usando un modello crostale simile a IASP91. Il simulatore modella le onde P (6–12 Hz), le onde S (2–6 Hz) e le onde superficiali (0.3–1 Hz) con ampiezze scalate per la magnitudo e distanze calcolate dalle equazioni dei tempi di percorrenza.",
        "fr": "Génère un sismogramme synthétique physiquement réaliste en utilisant un modèle crustal de type IASP91. Le simulateur modélise les ondes P (6–12 Hz), S (2–6 Hz) et de surface (0.3–1 Hz) avec des amplitudes mises à l'échelle par la magnitude et des distances calculées à partir des équations de temps de trajet.",
        "es": "Genera un sismograma sintético físicamente realista usando un modelo cortical tipo IASP91. El simulador modela ondas P (6–12 Hz), ondas S (2–6 Hz) y ondas superficiales (0.3–1 Hz) con amplitudes escaladas por la magnitud y distancias calculadas a partir de ecuaciones de tiempo de viaje.",
        "de": "Erzeugt ein physikalisch realistisches synthetisches Seismogramm unter Verwendung eines IASP91-ähnlichen Krustenmodells. Der Simulator modelliert P-Wellen (6–12 Hz), S-Wellen (2–6 Hz) und Oberflächenwellen (0,3–1 Hz) mit durch die Magnitude skalierten Amplituden und aus Laufzeitgleichungen berechneten Distanzen.",
    },
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

/* ── Tab intro banner ── */
.tab-intro {
    background: linear-gradient(90deg, rgba(88,166,255,0.08) 0%, rgba(63,185,80,0.04) 100%);
    border: 1px solid rgba(88,166,255,0.12);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 13.5px;
    color: rgba(255,255,255,0.62);
    line-height: 1.7;
}
.tab-intro strong { color: #e8edf5; }

/* ── Pill badges ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px 0; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: rgba(255,255,255,0.55);
}
.pill.blue  { background: rgba(88,166,255,0.1);  border-color: rgba(88,166,255,0.25);  color: #58a6ff; }
.pill.green { background: rgba(63,185,80,0.1);   border-color: rgba(63,185,80,0.25);   color: #3fb950; }
.pill.amber { background: rgba(210,153,34,0.1);  border-color: rgba(210,153,34,0.25);  color: #d29922; }
.pill.violet{ background: rgba(188,140,255,0.1); border-color: rgba(188,140,255,0.25); color: #bc8cff; }
.pill.coral { background: rgba(247,129,102,0.1); border-color: rgba(247,129,102,0.25); color: #f78166; }

/* ── Concept cards (two-column grid) ── */
.concept-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0; }
@media (max-width: 768px) { .concept-grid { grid-template-columns: 1fr; } }
.concept-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 16px;
}
.concept-card .cc-icon { font-size: 20px; margin-bottom: 6px; }
.concept-card .cc-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.concept-card .cc-body  { font-size: 12.5px; color: rgba(255,255,255,0.58); line-height: 1.6; }

/* ── Inline formula (smaller, inside prose) ── */
.f-inline {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.3);
    border-radius: 4px; padding: 1px 5px;
    font-size: 12px; color: #3fb950;
}

/* ── Gradient divider ── */
.grad-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(88,166,255,0.3), transparent);
    margin: 22px 0;
}

/* ── Sidebar tip ── */
.sb-tip {
    background: rgba(63,185,80,0.07);
    border-left: 2px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 7px 10px;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    margin: 8px 0 4px 0;
    line-height: 1.5;
}

/* ── Stat highlight row ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.stat-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 8px 14px;
    text-align: center; min-width: 80px;
}
.stat-item .si-val { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; color: #58a6ff; }
.stat-item .si-lbl { font-size: 10px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

/* ── Hero enhanced ── */
.hero-tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.hero-tag {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 3px 9px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: rgba(255,255,255,0.4);
}
.hero-tag.ht-blue   { background: rgba(88,166,255,0.1);  border-color: rgba(88,166,255,0.25);  color: #58a6ff; }
.hero-tag.ht-green  { background: rgba(63,185,80,0.1);   border-color: rgba(63,185,80,0.25);   color: #3fb950; }
.hero-tag.ht-violet { background: rgba(188,140,255,0.1); border-color: rgba(188,140,255,0.25); color: #bc8cff; }
.hero-tag.ht-amber  { background: rgba(210,153,34,0.1);  border-color: rgba(210,153,34,0.25);  color: #d29922; }

/* ── Step card ── */
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 12px 16px;
    margin-bottom: 8px; display: flex; gap: 12px; align-items: flex-start;
    transition: border-color .2s;
}
.step-card:hover { border-color: rgba(88,166,255,0.25); }
.step-icon {
    width: 28px; height: 28px; border-radius: 8px;
    background: rgba(88,166,255,0.15); color: #58a6ff;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-body { font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.55; }
.step-body strong { color: rgba(255,255,255,0.88); }

/* ── Key concept callout ── */
.key-concept {
    background: linear-gradient(135deg, rgba(188,140,255,0.08), rgba(88,166,255,0.06));
    border: 1px solid rgba(188,140,255,0.18);
    border-radius: 12px; padding: 14px 18px; margin: 12px 0;
}
.key-concept .kc-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #bc8cff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.key-concept .kc-body  { font-size: 13px; color: rgba(255,255,255,0.65); line-height: 1.65; }

/* ── Scrollable math box ── */
.math-scroll { overflow-x: auto; }
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
    st.markdown(f"<div class='sb-tip'>{T('syn_desc', lang)}</div>", unsafe_allow_html=True)
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
    _filter_tip = {
        "en": "💡 Typical seismology bands: teleseismic 0.01–2 Hz · regional 0.5–5 Hz · local 1–15 Hz",
        "it": "💡 Bande tipiche: teleseismi 0.01–2 Hz · regionali 0.5–5 Hz · locali 1–15 Hz",
        "fr": "💡 Bandes typiques : téléséismique 0.01–2 Hz · régional 0.5–5 Hz · local 1–15 Hz",
        "es": "💡 Bandas típicas: telesísmico 0.01–2 Hz · regional 0.5–5 Hz · local 1–15 Hz",
        "de": "💡 Typische Bänder: teleseismisch 0.01–2 Hz · regional 0.5–5 Hz · lokal 1–15 Hz",
    }
    st.markdown(f"<div class='sb-tip'>{_filter_tip[lang]}</div>", unsafe_allow_html=True)

    # ── STA/LTA ──────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('stalta_section', lang)}</div>", unsafe_allow_html=True)
    sta_len   = st.slider(T("sta_window", lang), 0.2, 5.0, 0.5, 0.1)
    lta_len   = st.slider(T("lta_window", lang), 5.0, 60.0, 20.0, 1.0)
    threshold = st.slider(T("threshold", lang), 1.0, 15.0, 3.5, 0.5)
    _stalta_tip = {
        "en": "💡 Rule of thumb: LTA ≥ 10× STA. Lower threshold → more triggers. Typical threshold: 3–5.",
        "it": "💡 Regola: LTA ≥ 10× STA. Soglia bassa → più trigger. Soglia tipica: 3–5.",
        "fr": "💡 Règle : LTA ≥ 10× STA. Seuil bas → plus de déclenchements. Seuil typique : 3–5.",
        "es": "💡 Regla: LTA ≥ 10× STA. Umbral bajo → más disparos. Umbral típico: 3–5.",
        "de": "💡 Faustregel: LTA ≥ 10× STA. Niedriger Schwellenwert → mehr Trigger. Typisch: 3–5.",
    }
    st.markdown(f"<div class='sb-tip'>{_stalta_tip[lang]}</div>", unsafe_allow_html=True)

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
  <h1>🌍 SeismicLens</h1>
  <div class='hero-sub'>{T("hero_sub", lang)}</div>
  <div class='hero-tag-row'>
    <span class='hero-tag ht-blue'>🔬 FFT &amp; Spectral Analysis</span>
    <span class='hero-tag ht-green'>🌊 STA/LTA P-wave Picker</span>
    <span class='hero-tag ht-violet'>🎛️ Butterworth Filter</span>
    <span class='hero-tag ht-amber'>📡 MiniSEED · IRIS · INGV</span>
    <span class='hero-tag'>🗺️ IASP91 Velocity Model</span>
    <span class='hero-tag'>📊 CSV Export</span>
    <span class='hero-tag'>🌐 5 Languages</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick-start guide ─────────────────────────────────────────────────────────
_GUIDE = {
    "en": {
        "title": "🚀 How to use SeismicLens",
        "steps": [
            ("1", "📂", "<strong>Choose a data source</strong> in the sidebar: generate a <em>synthetic earthquake</em> (instant, no file needed), upload a real <em>MiniSEED</em> file from IRIS/INGV, or upload a <em>CSV</em> waveform."),
            ("2", "🎛️", "<strong>Configure the Butterworth bandpass filter</strong>: set the low-cut and high-cut frequencies (Hz) and the filter order. Toggle it on/off to compare the filtered signal against the raw noisy waveform."),
            ("3", "🔍", "<strong>Adjust the STA/LTA P-wave detector</strong>: the STA window (short, 0.2–2 s) captures the onset energy; the LTA window (long, 5–60 s) tracks the background. Raise the threshold to suppress false triggers."),
            ("4", "📊", "<strong>Explore the analysis tabs</strong>: <em>Waveform</em> (time domain), <em>Spectral Analysis</em> (FFT + PSD), <em>Spectrogram</em> (time–frequency), <em>STA/LTA</em> (phase detector), <em>Velocity Model</em> (IASP91), <em>Theory &amp; Math</em>."),
            ("5", "💾", "<strong>Export your results</strong> as CSV files — processed signal, FFT spectrum (amplitude + phase), Welch PSD, and STA/LTA ratio — ready for Python, MATLAB, or Excel."),
        ],
    },
    "it": {
        "title": "🚀 Come usare SeismicLens",
        "steps": [
            ("1", "📂", "<strong>Scegli la sorgente dati</strong> nella barra laterale: genera un <em>terremoto sintetico</em> (istantaneo, senza file), carica un file <em>MiniSEED</em> reale da IRIS/INGV, o carica una forma d'onda <em>CSV</em>."),
            ("2", "🎛️", "<strong>Configura il filtro Butterworth bandpass</strong>: imposta le frequenze di taglio basso e alto (Hz) e l'ordine del filtro. Attivalo/disattivalo per confrontare il segnale filtrato con quello grezzo."),
            ("3", "🔍", "<strong>Regola il detector STA/LTA</strong>: la finestra STA (breve, 0.2–2 s) cattura l'energia dell'onset; la finestra LTA (lunga, 5–60 s) traccia il background. Alza la soglia per sopprimere i falsi trigger."),
            ("4", "📊", "<strong>Esplora le schede di analisi</strong>: <em>Forma d'onda</em>, <em>Analisi spettrale</em> (FFT + PSD), <em>Spettrogramma</em>, <em>STA/LTA</em>, <em>Modello di velocità</em>, <em>Teoria e matematica</em>."),
            ("5", "💾", "<strong>Esporta i risultati</strong> come file CSV — segnale elaborato, spettro FFT, PSD Welch, rapporto STA/LTA — pronti per Python, MATLAB o Excel."),
        ],
    },
    "fr": {
        "title": "🚀 Comment utiliser SeismicLens",
        "steps": [
            ("1", "📂", "<strong>Choisissez une source de données</strong> dans la barre latérale : générez un <em>séisme synthétique</em> (instantané, sans fichier), importez un fichier <em>MiniSEED</em> réel depuis IRIS/INGV, ou importez une forme d'onde <em>CSV</em>."),
            ("2", "🎛️", "<strong>Configurez le filtre Butterworth passe-bande</strong> : définissez les fréquences de coupure basse et haute (Hz) et l'ordre du filtre. Activez/désactivez pour comparer signal filtré et brut."),
            ("3", "🔍", "<strong>Ajustez le détecteur STA/LTA</strong> : la fenêtre STA (courte, 0.2–2 s) capture l'énergie d'onset ; la fenêtre LTA (longue, 5–60 s) suit le bruit de fond. Augmentez le seuil pour supprimer les faux déclenchements."),
            ("4", "📊", "<strong>Explorez les onglets d'analyse</strong> : <em>Forme d'onde</em>, <em>Analyse spectrale</em>, <em>Spectrogramme</em>, <em>STA/LTA</em>, <em>Modèle de vitesse</em>, <em>Théorie &amp; Maths</em>."),
            ("5", "💾", "<strong>Exportez vos résultats</strong> en CSV — signal, spectre FFT, PSD Welch, ratio STA/LTA — prêts pour Python, MATLAB ou Excel."),
        ],
    },
    "es": {
        "title": "🚀 Cómo usar SeismicLens",
        "steps": [
            ("1", "📂", "<strong>Elige una fuente de datos</strong> en la barra lateral: genera un <em>sismo sintético</em> (instantáneo, sin archivo), carga un archivo <em>MiniSEED</em> real de IRIS/INGV, o carga una forma de onda <em>CSV</em>."),
            ("2", "🎛️", "<strong>Configura el filtro Butterworth pasa-banda</strong>: establece las frecuencias de corte bajo y alto (Hz) y el orden. Actívalo/desactívalo para comparar señal filtrada y bruta."),
            ("3", "🔍", "<strong>Ajusta el detector STA/LTA</strong>: la ventana STA (corta, 0.2–2 s) captura la energía de onset; la ventana LTA (larga, 5–60 s) rastrea el fondo. Sube el umbral para suprimir falsos disparos."),
            ("4", "📊", "<strong>Explora las pestañas de análisis</strong>: <em>Forma de onda</em>, <em>Análisis espectral</em>, <em>Espectrograma</em>, <em>STA/LTA</em>, <em>Modelo de velocidad</em>, <em>Teoría y Matemáticas</em>."),
            ("5", "💾", "<strong>Exporta tus resultados</strong> como CSV — señal, espectro FFT, PSD Welch, razón STA/LTA — listos para Python, MATLAB o Excel."),
        ],
    },
    "de": {
        "title": "🚀 So verwendest du SeismicLens",
        "steps": [
            ("1", "📂", "<strong>Wähle eine Datenquelle</strong> in der Seitenleiste: Erzeuge ein <em>synthetisches Erdbeben</em> (sofort, keine Datei nötig), lade eine echte <em>MiniSEED</em>-Datei von IRIS/INGV hoch, oder lade eine <em>CSV</em>-Wellenform."),
            ("2", "🎛️", "<strong>Konfiguriere den Butterworth-Bandpassfilter</strong>: Untere und obere Grenzfrequenz (Hz) und Filterordnung einstellen. Ein-/ausschalten um gefiltertes und Rohsignal zu vergleichen."),
            ("3", "🔍", "<strong>Passe den STA/LTA-Detektor an</strong>: Das STA-Fenster (kurz, 0.2–2 s) erfasst die Onset-Energie; das LTA-Fenster (lang, 5–60 s) verfolgt den Hintergrund. Schwelle erhöhen um Fehlauslösungen zu unterdrücken."),
            ("4", "📊", "<strong>Erkunde die Analyse-Tabs</strong>: <em>Wellenform</em>, <em>Spektralanalyse</em>, <em>Spektrogramm</em>, <em>STA/LTA</em>, <em>Geschwindigkeitsmodell</em>, <em>Theorie &amp; Mathematik</em>."),
            ("5", "💾", "<strong>Exportiere deine Ergebnisse</strong> als CSV-Dateien — Signal, FFT-Spektrum, Welch-PSD, STA/LTA-Verhältnis — bereit für Python, MATLAB oder Excel."),
        ],
    },
}

guide = _GUIDE[lang]
steps_html = "".join(
    f"<div class='step-card'><div class='step-icon'>{ico}</div><div class='step-body'>{txt}</div></div>"
    for _, ico, txt in guide["steps"]
)
st.markdown(
    f"<div class='sl-card'><div class='section-title' style='margin-top:0'>{guide['title']}</div>{steps_html}</div>",
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

    # Tab intro
    st.markdown(f"<div class='tab-intro'>{T('wave_desc', lang)}</div>", unsafe_allow_html=True)

    # Active pills
    pills_html = "<div class='pill-row'>"
    pills_html += f"<span class='pill blue'>fs = {fs:.0f} Hz</span>"
    pills_html += f"<span class='pill green'>N = {N} samples</span>"
    if filter_on:
        pills_html += f"<span class='pill violet'>Butterworth {f_low}–{f_high} Hz · order {filter_order}</span>"
    else:
        pills_html += f"<span class='pill amber'>⚠ Filter OFF — raw signal</span>"
    if p_time:
        pills_html += f"<span class='pill coral'>P-arrival @ {p_time:.2f} s</span>"
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    raw_ov = signal_tapered if show_raw and filter_on else None
    filt_label = T("wave_filtered", lang) if filter_on else T("wave_unfiltered", lang)
    fig_wave = make_waveform_fig(t, signal_proc, raw=raw_ov, p_time=p_time, s_time=s_time,
                                 title=f"{T('wave_title', lang)} ({filt_label})")
    st.plotly_chart(fig_wave, use_container_width=True)

    if p_time:
        st.markdown(
            f"<div class='success-box'>✅ {T('wave_p_detected', lang, t=p_time, thr=threshold)}</div>",
            unsafe_allow_html=True,
        )
    if s_time and data_source == 0:
        sp_delay = s_time - (p_time or 0)
        st.markdown(
            f"<div class='info-box'>📐 {T('wave_s_info', lang, s=s_time, sp=sp_delay, d=sp_delay*8)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    # Why filter + filter details
    _fw_col1, _fw_col2 = st.columns(2)
    with _fw_col1:
        st.markdown(f"""
<div class='key-concept'>
<div class='kc-title'>🎛️ {T('filter_why_title', lang)}</div>
<div class='kc-body'>{T('filter_why_body', lang)}</div>
</div>""", unsafe_allow_html=True)

    with _fw_col2:
        if filter_on:
            eff_order = 2 * filter_order
            st.markdown(f"""
<div class='key-concept'>
<div class='kc-title'>⚙️ {T('filter_details_title', lang)}</div>
<div class='kc-body'>
<b>Low cut:</b> {f_low} Hz &nbsp;|&nbsp; <b>High cut:</b> {f_high} Hz &nbsp;|&nbsp; <b>Order:</b> {filter_order} → effective <b>{eff_order}</b> (zero-phase)<br>
<b>Nyquist:</b> {fs/2:.1f} Hz &nbsp;|&nbsp; Roll-off: <span class='f-inline'>{40*filter_order} dB/decade</span><br><br>
<code>sosfiltfilt</code> = forward + backward pass → zero phase distortion,
preserving wave arrival times exactly.
</div>
</div>""", unsafe_allow_html=True)

    if filter_on:
        with st.expander(T("filter_details_title", lang) + " — full math"):
            st.markdown(f"""
**Parameters:**
- Low cut: **{f_low} Hz** | High cut: **{f_high} Hz**
- Order: **{filter_order}** (effective {eff_order} with zero-phase `sosfiltfilt`)
- Nyquist: **{fs/2:.1f} Hz**
- Implementation: `scipy.signal.sosfiltfilt` — Second-Order Sections

**Why Second-Order Sections (SOS)?**
A high-order IIR filter (e.g., order 8) has a long chain of multiply-adds.
In direct form the coefficients have very different magnitudes → numerical overflow/underflow.
SOS splits the filter into cascaded biquad (2nd-order) sections, each stable on its own.

**Why zero-phase?**
A causal filter delays different frequencies by different amounts (group delay ≠ constant),
distorting seismic waveforms and shifting arrival times.
`sosfiltfilt` runs the filter forward then backward: the phase responses cancel exactly,
giving zero phase distortion with double the effective roll-off.
""")
            st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}")


# ── Spectral Analysis ─────────────────────────────────────────────────────────
with tab_fft:
    st.markdown(f"<div class='section-title'>{T('tab_fft', lang)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tab-intro'>{T('fft_desc', lang)}</div>", unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns(3)
    cf1.markdown(metric_card(T("fft_domf", lang),    f"{dominant_f:.3f}", "Hz", "green"),  unsafe_allow_html=True)
    cf2.markdown(metric_card(T("fft_centroid", lang), f"{centroid_f:.3f}", "Hz", "violet"), unsafe_allow_html=True)
    cf3.markdown(metric_card(T("fft_bw", lang),       f"{bandwidth:.3f}",  "Hz", "amber"),  unsafe_allow_html=True)

    # Concept grid
    _fft_concepts = {
        "en": [
            ("📊", "Dominant Frequency", f"The frequency bin with the highest amplitude: <strong>{dominant_f:.3f} Hz</strong>. This is the most energetic frequency in the signal."),
            ("🎯", "Spectral Centroid", f"The energy-weighted mean frequency: <strong>{centroid_f:.3f} Hz</strong>. It represents the 'centre of mass' of the spectrum — higher than dominant_f if energy spreads to high frequencies."),
            ("📏", "Bandwidth (RMS)", f"The RMS spread of energy around the centroid: <strong>{bandwidth:.3f} Hz</strong>. Wide bandwidth = broad-band signal (e.g., impulsive source); narrow = quasi-monochromatic."),
            ("🔢", "Frequency Resolution", f"Each FFT bin covers <strong>Δf = fs/N = {fs/N:.4f} Hz</strong>. To improve resolution, increase the signal duration (more samples N)."),
        ],
        "it": [
            ("📊", "Frequenza dominante", f"Il bin con ampiezza massima: <strong>{dominant_f:.3f} Hz</strong>. È la frequenza più energetica del segnale."),
            ("🎯", "Centroide spettrale", f"La media pesata per l'energia: <strong>{centroid_f:.3f} Hz</strong>. Rappresenta il 'centro di massa' dello spettro."),
            ("📏", "Larghezza di banda (RMS)", f"La dispersione RMS dell'energia attorno al centroide: <strong>{bandwidth:.3f} Hz</strong>. Banda larga = sorgente impulsiva; stretta = quasi-monocromatica."),
            ("🔢", "Risoluzione in frequenza", f"Ogni bin FFT copre <strong>Δf = fs/N = {fs/N:.4f} Hz</strong>. Per migliorare la risoluzione, aumentare la durata del segnale."),
        ],
        "fr": [
            ("📊", "Fréquence dominante", f"Le bin avec l'amplitude maximale : <strong>{dominant_f:.3f} Hz</strong>. C'est la fréquence la plus énergétique du signal."),
            ("🎯", "Centroïde spectral", f"La moyenne pondérée par l'énergie : <strong>{centroid_f:.3f} Hz</strong>. Représente le 'centre de masse' du spectre."),
            ("📏", "Largeur de bande (RMS)", f"La dispersion RMS autour du centroïde : <strong>{bandwidth:.3f} Hz</strong>. Large = source impulsive ; étroite = quasi-monochromatique."),
            ("🔢", "Résolution fréquentielle", f"Chaque bin FFT couvre <strong>Δf = fs/N = {fs/N:.4f} Hz</strong>. Pour améliorer la résolution, augmenter la durée du signal."),
        ],
        "es": [
            ("📊", "Frecuencia dominante", f"El bin con amplitud máxima: <strong>{dominant_f:.3f} Hz</strong>. Es la frecuencia más energética de la señal."),
            ("🎯", "Centroide espectral", f"La media ponderada por energía: <strong>{centroid_f:.3f} Hz</strong>. Representa el 'centro de masa' del espectro."),
            ("📏", "Ancho de banda (RMS)", f"La dispersión RMS alrededor del centroide: <strong>{bandwidth:.3f} Hz</strong>. Ancho = fuente impulsiva; estrecho = casi-monocromático."),
            ("🔢", "Resolución frecuencial", f"Cada bin FFT cubre <strong>Δf = fs/N = {fs/N:.4f} Hz</strong>. Para mejorar la resolución, aumentar la duración de la señal."),
        ],
        "de": [
            ("📊", "Dominante Frequenz", f"Der Bin mit maximaler Amplitude: <strong>{dominant_f:.3f} Hz</strong>. Das ist die energiereichste Frequenz des Signals."),
            ("🎯", "Spektralzentroid", f"Der energiegewichtete Mittelwert: <strong>{centroid_f:.3f} Hz</strong>. Repräsentiert den 'Schwerpunkt' des Spektrums."),
            ("📏", "Bandbreite (RMS)", f"Die RMS-Streuung um den Zentroid: <strong>{bandwidth:.3f} Hz</strong>. Breit = impulsive Quelle; schmal = quasi-monochromatisch."),
            ("🔢", "Frequenzauflösung", f"Jeder FFT-Bin umfasst <strong>Δf = fs/N = {fs/N:.4f} Hz</strong>. Für bessere Auflösung Signaldauer erhöhen."),
        ],
    }
    _concepts = _fft_concepts[lang]
    _cg_html = "<div class='concept-grid'>"
    for ico, title, body in _concepts:
        _cg_html += f"<div class='concept-card'><div class='cc-icon'>{ico}</div><div class='cc-title'>{title}</div><div class='cc-body'>{body}</div></div>"
    _cg_html += "</div>"
    st.markdown(_cg_html, unsafe_allow_html=True)

    st.plotly_chart(make_fft_fig(freqs, amplitudes, dominant_f, centroid_f), use_container_width=True)

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    if show_phase:
        st.markdown(f"<div class='section-title'>Phase Spectrum ∠X(f)</div>", unsafe_allow_html=True)
        st.plotly_chart(make_phase_fig(freqs, phases_deg), use_container_width=True)
        st.markdown(f"<div class='info-box'>ℹ️ {T('phase_info', lang)}</div>", unsafe_allow_html=True)

        with st.expander("📐 Phase — deeper explanation"):
            st.markdown("""
Each FFT coefficient **X(f)** is a **complex number** living in the 2D complex plane:
""")
            st.latex(r"X(f) = \text{Re}[X(f)] + j \cdot \text{Im}[X(f)] = |X(f)| \cdot e^{j\varphi(f)}")
            st.markdown("""
- **|X(f)|** = amplitude (distance from origin in the complex plane)
- **φ(f) = ∠X(f)** = phase angle = `atan2(Im, Re)` ∈ [−π, π]

**What does the phase tell us?**
- **Random noise**: phases are uniformly distributed — no coherent structure.
- **Impulsive signal** (P-wave onset): phases cluster together near the arrival time.
- **Dispersive wave train** (surface waves): phase varies smoothly with frequency (group delay ≠ constant).
- **Zero-phase filter**: after `sosfiltfilt`, the phase spectrum of the filtered signal equals the unfiltered one (phase is unchanged).
""")

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    if show_psd_toggle:
        st.markdown(f"<div class='section-title'>Power Spectral Density (PSD) — Welch method</div>", unsafe_allow_html=True)
        st.plotly_chart(make_psd_fig(f_psd, psd), use_container_width=True)
        st.markdown(f"<div class='info-box'>ℹ️ {T('psd_info', lang)}</div>", unsafe_allow_html=True)

        with st.expander("📈 PSD — deeper explanation"):
            st.markdown("""
**Why not just use the FFT amplitude²?**
A single FFT of a long signal gives a **very noisy** power estimate — each frequency bin has
high variance because it is computed from just one complex number.

**Welch's method** (1967) reduces variance by averaging:
""")
            st.latex(r"S(f) = \frac{1}{K} \sum_{k=1}^{K} \left| \text{FFT}\{x_k[n] \cdot w[n]\} \right|^2")
            st.markdown("""
- Split signal into **K overlapping segments** (50% overlap default)
- Apply **Hann window** to each segment (reduces spectral leakage)
- Compute FFT² of each segment
- **Average** → variance reduced by ~1/K compared to single FFT

**Units:** counts²/Hz (raw) or dB re 1 count²/Hz (log scale, as shown).
**dB conversion:** `PSD_dB = 10 · log₁₀(PSD)`
""")
            st.latex(r"\text{PSD}_{dB}(f) = 10 \cdot \log_{10}\left(S(f)\right)")


# ── Spectrogram ───────────────────────────────────────────────────────────────
with tab_spec:
    if show_spec_toggle:
        st.markdown(f"<div class='section-title'>{T('tab_spec', lang)} — STFT</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tab-intro'>{T('spec_desc', lang)}</div>", unsafe_allow_html=True)
        try:
            t_spec, f_spec, Sxx = compute_spectrogram(signal_proc, fs)
            st.plotly_chart(make_spectrogram_fig(t_spec, f_spec, Sxx), use_container_width=True)

            st.markdown(f"<div class='info-box'>ℹ️ {T('spec_info', lang)}</div>", unsafe_allow_html=True)

            _spec_read = {
                "en": "🔍 <strong>How to read this plot:</strong> The x-axis is time (s), y-axis is frequency (Hz), colour is energy (dB). Look for: <em>vertical bright streaks</em> = impulsive onset (P/S arrival); <em>horizontal bands</em> = monochromatic interference; <em>diagonal stripes</em> = dispersive surface waves.",
                "it": "🔍 <strong>Come leggere questo grafico:</strong> L'asse x è il tempo (s), l'asse y la frequenza (Hz), il colore è l'energia (dB). Cerca: <em>strisce verticali luminose</em> = onset impulsivo (arrivo P/S); <em>bande orizzontali</em> = interferenza monocromatica; <em>strisce diagonali</em> = onde superficiali dispersive.",
                "fr": "🔍 <strong>Comment lire ce graphique :</strong> L'axe x est le temps (s), l'axe y la fréquence (Hz), la couleur est l'énergie (dB). Cherchez : <em>bandes verticales lumineuses</em> = onset impulsif (arrivée P/S) ; <em>bandes horizontales</em> = interférence monochromatique ; <em>bandes diagonales</em> = ondes de surface dispersives.",
                "es": "🔍 <strong>Cómo leer este gráfico:</strong> El eje x es tiempo (s), el eje y frecuencia (Hz), el color es energía (dB). Busca: <em>franjas verticales brillantes</em> = onset impulsivo (llegada P/S); <em>bandas horizontales</em> = interferencia monocromática; <em>franjas diagonales</em> = ondas superficiales dispersivas.",
                "de": "🔍 <strong>Wie man dieses Diagramm liest:</strong> X-Achse = Zeit (s), Y-Achse = Frequenz (Hz), Farbe = Energie (dB). Suche nach: <em>vertikalen hellen Streifen</em> = impulsiver Onset (P/S-Ankunft); <em>horizontalen Bändern</em> = monochromatische Interferenz; <em>diagonalen Streifen</em> = dispersive Oberflächenwellen.",
            }
            st.markdown(f"<div class='warn-box'>{_spec_read[lang]}</div>", unsafe_allow_html=True)

            with st.expander("⚛️ STFT & Heisenberg Uncertainty — detailed explanation"):
                st.markdown("### Short-Time Fourier Transform (STFT)")
                st.latex(r"\text{STFT}(\tau, f) = \int_{-\infty}^{\infty} x(t) \cdot w(t - \tau) \cdot e^{-j 2\pi f t} \, dt")
                st.markdown("""
In discrete form with a Hann window of length M:
- The signal is **windowed** (masked) around time τ, keeping only M samples.
- The **FFT** of those M samples gives the local spectrum at time τ.
- The window slides forward by **hop_size** samples, producing a 2D matrix.

**Time–Frequency Trade-off (Heisenberg–Gabor Uncertainty):**
""")
                st.latex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}")
                st.markdown("""
| Window choice | Time resolution Δt | Frequency resolution Δf |
|---|---|---|
| **Short window** (e.g., 64 samples) | Fine (ms) | Coarse (several Hz) |
| **Long window** (e.g., 1024 samples) | Coarse (hundreds of ms) | Fine (fractions of Hz) |

There is **no way to get both simultaneously** — this is a fundamental limit, not a computational one.

**Welch PSD vs Spectrogram:**
The Welch PSD (shown in Spectral Analysis tab) is the **time-averaged** spectrogram — it collapses the time axis and shows only the mean power per frequency.
""")
        except Exception as e:
            st.warning(f"Spectrogram error: {e}")
    else:
        st.info(T("spec_disabled", lang))


# ── STA/LTA ───────────────────────────────────────────────────────────────────
with tab_stalta:
    if show_stalta_tog:
        st.markdown(f"<div class='section-title'>{T('tab_stalta', lang)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tab-intro'>{T('stalta_desc', lang)}</div>", unsafe_allow_html=True)
        try:
            stalta_vals = compute_sta_lta(signal_proc, fs, sta_len, lta_len)
            t_sl = np.linspace(0, len(stalta_vals) / fs, len(stalta_vals))
            st.plotly_chart(make_stalta_fig(t_sl, stalta_vals, threshold), use_container_width=True)

            n_trig = int(np.sum(np.diff((stalta_vals > threshold).astype(int)) > 0))
            if p_time:
                st.markdown(
                    f"<div class='success-box'>✅ {T('stalta_trigger_ok', lang, t=p_time, n=n_trig)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='warn-box'>⚠️ {T('stalta_no_trigger', lang)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

            # Parameter cards
            sta_n = int(sta_len * fs)
            lta_n = int(lta_len * fs)
            _sl_ratio = lta_len / sta_len
            _sl_color = "green" if _sl_ratio >= 10 else "amber"
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.markdown(metric_card("STA window", f"{sta_len}", "s", "blue"),   unsafe_allow_html=True)
            _pc2.markdown(metric_card("LTA window", f"{lta_len}", "s", "violet"), unsafe_allow_html=True)
            _pc3.markdown(metric_card("LTA / STA",  f"{_sl_ratio:.1f}", "×", _sl_color), unsafe_allow_html=True)
            _pc4.markdown(metric_card("Threshold",  f"{threshold}", "",  "coral"), unsafe_allow_html=True)

            with st.expander("🔬 STA/LTA — algorithm details & tuning guide"):
                st.markdown("### Short-Term Average / Long-Term Average")
                st.latex(r"\text{STA}(t) = \frac{1}{N_{sta}} \sum_{k=0}^{N_{sta}-1} x^2[t-k]")
                st.latex(r"\text{LTA}(t) = \frac{1}{N_{lta}} \sum_{k=0}^{N_{lta}-1} x^2[t-k]")
                st.latex(r"R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)} \quad \xrightarrow{\text{trigger when}} \quad R > \theta")
                st.markdown(f"""
**Current settings:**
- STA = {sta_len} s → **{sta_n} samples** (captures onset energy burst)
- LTA = {lta_len} s → **{lta_n} samples** (tracks background noise floor)
- LTA/STA ratio = **{_sl_ratio:.1f}×** {'✅ good' if _sl_ratio >= 10 else '⚠️ LTA should be ≥10× STA for stable detection'}
- Threshold = **{threshold}** (trigger when R > {threshold})

**O(N) implementation via prefix sums:**
```
cs = cumsum(x²)                      # precomputed once, O(N)
STA(t) = (cs[t+1] - cs[t-N_sta]) / N_sta    # O(1) per sample
LTA(t) = (cs[t+1] - cs[t-N_lta]) / N_lta    # O(1) per sample
```

**Tuning guide:**
| Scenario | STA | LTA | Threshold |
|---|---|---|---|
| Local earthquake (short) | 0.2–0.5 s | 10–20 s | 3–4 |
| Regional earthquake | 0.5–1 s | 20–40 s | 3–5 |
| Teleseismic | 1–2 s | 30–60 s | 2–3 |
| Very noisy data | 0.5 s | 30 s | 5–8 |

Reference: Allen, R.V. (1978), *Automatic earthquake recognition and timing from single traces*. BSSA, 68(5).
""")
        except Exception as e:
            st.warning(f"STA/LTA error: {e}")
    else:
        st.info(T("stalta_disabled", lang))


# ── Velocity Model ────────────────────────────────────────────────────────────
with tab_model:
    st.markdown(f"<div class='section-title'>{T('tab_model', lang)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tab-intro'>{T('model_desc', lang)}</div>", unsafe_allow_html=True)

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

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    # Elastic moduli concept grid
    _vp_concepts = {
        "en": [
            ("🔵", "P-wave velocity (Vp)", "Compressional waves — particles vibrate parallel to propagation direction. Controlled by bulk modulus K (resistance to compression) + shear modulus G + density ρ."),
            ("🟢", "S-wave velocity (Vs)", "Shear waves — particles vibrate perpendicular to propagation. Controlled only by shear modulus G. <strong>Vs = 0 in fluids</strong> because fluids cannot sustain shear stress (G = 0)."),
            ("🟡", "Vp/Vs ratio", "Key diagnostic indicator. Poisson solid: Vp/Vs = √3 ≈ 1.732, ν = 0.25. High Vp/Vs (> 1.8) can indicate fluid saturation, partial melt, or high pore pressure."),
            ("🟣", "Poisson's ratio ν", "Ratio of transverse to longitudinal strain. Range: 0 (incompressible shear) to 0.5 (incompressible liquid). Typical crustal rocks: 0.24–0.28."),
        ],
        "it": [
            ("🔵", "Velocità onda P (Vp)", "Onde compressive — le particelle vibrano parallele alla direzione di propagazione. Controllata dal modulo di bulk K + modulo di taglio G + densità ρ."),
            ("🟢", "Velocità onda S (Vs)", "Onde di taglio — le particelle vibrano perpendicolari alla propagazione. Controllata solo dal modulo G. <strong>Vs = 0 nei fluidi</strong> perché i fluidi non sopportano sforzi di taglio."),
            ("🟡", "Rapporto Vp/Vs", "Indicatore diagnostico chiave. Solido di Poisson: Vp/Vs = √3 ≈ 1.732, ν = 0.25. Valori alti (> 1.8) indicano saturazione di fluidi, fusione parziale o alta pressione dei pori."),
            ("🟣", "Rapporto di Poisson ν", "Rapporto tra deformazione trasversale e longitudinale. Intervallo: 0 (solido incomprimibile in taglio) a 0.5 (liquido incomprimibile). Rocce crostali tipiche: 0.24–0.28."),
        ],
        "fr": [
            ("🔵", "Vitesse onde P (Vp)", "Ondes compressionnelles — les particules vibrent parallèlement à la direction de propagation. Contrôlée par le module de compressibilité K, le module de cisaillement G et la densité ρ."),
            ("🟢", "Vitesse onde S (Vs)", "Ondes de cisaillement — les particules vibrent perpendiculairement. Contrôlée uniquement par G. <strong>Vs = 0 dans les fluides</strong> car ils ne peuvent pas soutenir une contrainte de cisaillement."),
            ("🟡", "Rapport Vp/Vs", "Indicateur diagnostique clé. Solide de Poisson : Vp/Vs = √3 ≈ 1.732, ν = 0.25. Valeurs élevées (> 1.8) indiquent saturation en fluide, fusion partielle ou haute pression des pores."),
            ("🟣", "Coefficient de Poisson ν", "Rapport déformation transversale / longitudinale. Plage : 0 à 0.5. Roches crustales typiques : 0.24–0.28."),
        ],
        "es": [
            ("🔵", "Velocidad onda P (Vp)", "Ondas compresionales — las partículas vibran paralelas a la dirección de propagación. Controlada por módulo de bulk K + módulo de corte G + densidad ρ."),
            ("🟢", "Velocidad onda S (Vs)", "Ondas de corte — las partículas vibran perpendiculares. Controlada solo por G. <strong>Vs = 0 en fluidos</strong> porque no pueden sostener esfuerzos de corte."),
            ("🟡", "Relación Vp/Vs", "Indicador diagnóstico clave. Sólido de Poisson: Vp/Vs = √3 ≈ 1.732, ν = 0.25. Valores altos (> 1.8) indican saturación de fluidos, fusión parcial o alta presión de poros."),
            ("🟣", "Razón de Poisson ν", "Relación deformación transversal / longitudinal. Rango: 0 a 0.5. Rocas corticales típicas: 0.24–0.28."),
        ],
        "de": [
            ("🔵", "P-Wellengeschwindigkeit (Vp)", "Kompressionswellen — Partikel schwingen parallel zur Ausbreitungsrichtung. Gesteuert durch Kompressionsmodul K + Schubmodul G + Dichte ρ."),
            ("🟢", "S-Wellengeschwindigkeit (Vs)", "Scherwellen — Partikel schwingen senkrecht. Nur durch G gesteuert. <strong>Vs = 0 in Flüssigkeiten</strong>, da diese keine Scherspannung aufnehmen können."),
            ("🟡", "Vp/Vs-Verhältnis", "Wichtiger Diagnoseindikator. Poisson-Körper: Vp/Vs = √3 ≈ 1.732, ν = 0.25. Hohe Werte (> 1.8) deuten auf Fluidsättigung, partielle Schmelze oder hohen Porendruck hin."),
            ("🟣", "Poisson-Zahl ν", "Verhältnis von Quer- zu Längsdehnung. Bereich: 0 bis 0.5. Typische Krustengesteine: 0.24–0.28."),
        ],
    }
    _vm_grid = "<div class='concept-grid'>"
    for ico, title, body in _vp_concepts[lang]:
        _vm_grid += f"<div class='concept-card'><div class='cc-icon'>{ico}</div><div class='cc-title'>{title}</div><div class='cc-body'>{body}</div></div>"
    _vm_grid += "</div>"
    st.markdown(_vm_grid, unsafe_allow_html=True)

    with st.expander("⚙️ Elastic wave equations — full derivation"):
        st.latex(r"V_P = \sqrt{\frac{K + \frac{4}{3}G}{\rho}}")
        st.latex(r"V_S = \sqrt{\frac{G}{\rho}}")
        st.latex(r"\nu = \frac{(V_P/V_S)^2 - 2}{2\left[(V_P/V_S)^2 - 1\right]}")
        st.markdown("""
Where:
- **K** = bulk modulus (resistance to uniform compression, Pa)
- **G** = shear modulus (resistance to shear deformation, Pa)
- **ρ** = density (kg/m³)
- **ν** = Poisson's ratio (dimensionless, 0 < ν < 0.5 for stable materials)

**Why does Vp > Vs always?**
Vp involves both K and G (compressional + shear restoring forces), while Vs only involves G.
Since K > 0 always, we have K + 4G/3 > G, hence Vp > Vs for any solid material.

**Fluid indicator:** In fully saturated rock, the pore fluid dramatically increases K (fluid is hard to compress)
while leaving G unchanged → Vp rises, Vs stays the same → Vp/Vs increases above √3.
""")

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
    st.markdown("""
<div class='tab-intro'>
This tab is your <strong>mathematical reference</strong> for every algorithm used in SeismicLens.
Each section explains the concept from first principles, gives the key equations, and links to how
the algorithm is used in the other tabs.
</div>""", unsafe_allow_html=True)

    # ── 1. DFT / FFT ──────────────────────────────────────────────────────────
    with st.expander("📐 1 — Discrete Fourier Transform (DFT) & Fast FFT", expanded=True):
        st.markdown("#### What does the FFT actually do?")
        st.markdown("""
A time-domain signal **x[n]** is a list of N amplitude values sampled at rate fs.
The DFT converts it into N **complex-valued frequency coefficients** X[k].
Each coefficient tells you *how much of a pure sinusoid at frequency f_k = k·fs/N is present*.
""")
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi k n}{N}}, \quad k = 0, 1, \ldots, N-1")
        st.markdown("""
The complex exponential (from **Euler's formula**) is the key building block:
""")
        st.latex(r"e^{j\theta} = \cos(\theta) + j\sin(\theta)")
        st.markdown("""
So each DFT basis function is a **cosine** (real part) + **sine** (imaginary part) at frequency f_k.
The DFT projects the signal onto each basis function (inner product), measuring:
- `Re(X[k])` → how much of the cosine at f_k is in the signal
- `Im(X[k])` → how much of the sine at f_k is in the signal
- `|X[k]|` = √(Re² + Im²) → total **amplitude** at f_k
- `∠X[k]` = atan2(Im, Re) → **phase** at f_k

#### Why is the FFT faster?
The naïve DFT requires **O(N²)** multiplications. The Cooley-Tukey FFT (1965) exploits the periodicity
of e^{-j2πkn/N} to recursively split the sum (decimation-in-time), reducing cost to **O(N log N)**.
For N = 100,000 this is ~10,000× faster.

#### One-sided spectrum for real signals
Real signals have **Hermitian symmetry**: X[N-k] = X[k]*. So only N/2+1 bins are unique.
`scipy.fft.rfft` exploits this and returns only the positive-frequency half.
Normalisation to recover true amplitude:
""")
        st.latex(r"A[k] = \frac{2}{N} |X[k]|, \quad k = 1, \ldots, \frac{N}{2}-1")
        st.markdown("""
**Hann window** is applied before the FFT to reduce **spectral leakage** — the smearing of energy 
from a strong frequency bin into neighbouring bins that occurs when the signal is not an integer 
number of cycles within the window.
""")
        st.latex(r"w[n] = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N-1}\right)\right)")

    # ── 2. Butterworth Filter ──────────────────────────────────────────────────
    with st.expander("🎛️ 2 — Butterworth Bandpass Filter"):
        st.markdown("""
#### Why Butterworth?
The Butterworth filter is **maximally flat in the passband** — it has no ripple (unlike Chebyshev
or elliptic filters). This is important in seismology because we need to preserve the relative
amplitudes of waves inside the passband.

#### Transfer function
In the Laplace (s) domain, the N-th order low-pass Butterworth has:
""")
        st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}")
        st.markdown("""
- At ω = ω_c (corner frequency): |H| = 1/√2 = −3 dB (half-power point)
- Roll-off beyond ω_c: **20n dB/decade** (one-pass)
- The poles lie on a circle of radius ω_c in the s-plane at angles (2k+n-1)π/2n

A **bandpass** is formed by cascading a high-pass (removes below f_low) and a low-pass (removes above f_high):
""")
        eff = 2 * filter_order
        rolloff = 20 * filter_order
        eff_rolloff = 40 * filter_order
        st.latex(r"H_{BP}(s) = H_{HP}(s) \cdot H_{LP}(s)")
        st.markdown(f"""
**Current filter:** order {filter_order} → effective order **{eff}** (zero-phase `sosfiltfilt`)
→ roll-off = **{eff_rolloff} dB/decade** beyond the corner frequencies.

#### Why Second-Order Sections (SOS)?
A high-order IIR filter expressed as a single fraction H(z) = B(z)/A(z) is numerically unstable
for orders ≥ 8 — the large polynomial coefficients cause floating-point overflow.
SOS **factorises** H(z) into a cascade of stable biquad (2nd-order) sections:
""")
        st.latex(r"H(z) = \prod_{k=1}^{n/2} \frac{b_{0k} + b_{1k} z^{-1} + b_{2k} z^{-2}}{1 + a_{1k} z^{-1} + a_{2k} z^{-2}}")

        st.markdown("""
#### Zero-phase filtering
`sosfiltfilt` applies the filter **forward** then **backward** through the signal.
The two passes have opposite phase responses that cancel exactly, giving:
- ✅ **Zero phase distortion** → arrival times preserved exactly
- ✅ Effective order **doubled**: order n → effective 2n, roll-off doubled
- ⚠️ Signal must be long enough to avoid edge artefacts (rule: ≥ 3× filter order / f_low)
""")

    # ── 3. STA/LTA ────────────────────────────────────────────────────────────
    with st.expander("🔍 3 — STA/LTA Seismic Phase Detector"):
        st.markdown("""
#### Intuition
When a seismic wave arrives, the signal energy suddenly **increases sharply**.
STA/LTA exploits this: the short-term window (STA) tracks the instantaneous energy;
the long-term window (LTA) tracks the background noise level.
When the ratio STA/LTA spikes above a threshold, a phase is declared.
""")
        st.latex(r"\text{STA}(t) = \frac{1}{N_{sta}} \sum_{k=0}^{N_{sta}-1} x^2[t-k]")
        st.latex(r"\text{LTA}(t) = \frac{1}{N_{lta}} \sum_{k=0}^{N_{lta}-1} x^2[t-k]")
        st.latex(r"R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}")
        st.markdown(f"""
**Why square the samples?**
Squaring makes the metric invariant to the sign (polarity) of the signal — a negative spike
is just as impulsive as a positive one. It also tracks **energy** rather than amplitude.

**O(N) implementation** via prefix sums (cumulative sum of x²):
```python
cs = np.cumsum(x**2)       # precompute once  →  O(N)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta      # O(1) per sample
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta      # O(1) per sample
```

Without prefix sums, a naïve sliding-window sum would be **O(N × max(N_sta, N_lta))** — orders of
magnitude slower for long LTA windows.

Reference: Allen, R.V. (1978). *Automatic earthquake recognition and timing from single traces*.
Bulletin of the Seismological Society of America, 68(5), 1521–1532.
""")

    # ── 4. Spectrogram / STFT ─────────────────────────────────────────────────
    with st.expander("🎨 4 — Spectrogram & Time-Frequency Analysis"):
        st.markdown("""
#### Why a spectrogram?
The FFT gives the **global** frequency content over the entire signal — it can't tell you *when*
each frequency is present. A spectrogram solves this by computing the FFT locally in time.
""")
        st.latex(r"\text{STFT}(\tau, f) = \sum_{n} x[n] \cdot w[n - \tau] \cdot e^{-j2\pi f n / f_s}")
        st.markdown("""
The spectrogram is the squared magnitude: `|STFT(τ,f)|²`, displayed in dB.

#### Heisenberg-Gabor Uncertainty Principle
There is a **fundamental trade-off** between time and frequency resolution — it cannot be overcome
by any algorithm:
""")
        st.latex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}")
        st.markdown("""
| Window size | Δt (time res.) | Δf (freq. res.) | Best for |
|---|---|---|---|
| Short (32–128 pts) | Fine (ms) | Coarse (Hz) | Impulsive arrivals, onset timing |
| Long (512–2048 pts) | Coarse (100s ms) | Fine (0.1 Hz) | Slow dispersive waves |

In SeismicLens, SciPy's `scipy.signal.spectrogram` uses a Hann window of length `nperseg = min(256, N//4)`
with 75% overlap, balancing time and frequency resolution for typical seismic signals.

#### Reading seismic spectrograms
- **Vertical bright streaks** at early time → P or S wave arrival (broadband, impulsive)
- **Horizontal bright bands** → monochromatic noise (e.g., 50/60 Hz power line)
- **Energy that sweeps from high to low frequency over time** → dispersive surface wave (Love or Rayleigh)
- **Low-frequency stripe throughout** → ocean microseisms (0.1–0.3 Hz)
""")

    # ── 5. Seismic Waves ──────────────────────────────────────────────────────
    with st.expander("🌊 5 — Seismic Wave Physics"):
        st.markdown("#### Types of seismic waves")
        st.markdown("""
Seismic waves are mechanical waves that propagate through the Earth by elastic deformation.
They are classified into **body waves** (travel through the interior) and **surface waves** 
(travel along the Earth's surface).
""")
        st.latex(r"V_P = \sqrt{\frac{K + \frac{4}{3}G}{\rho}} \qquad V_S = \sqrt{\frac{G}{\rho}}")
        st.markdown("""
| Wave type | Polarisation | Speed | Typical freq. (local EQ) | Notes |
|---|---|---|---|---|
| **P (Primary)** | Compressional ‖ | Fastest | 6–12 Hz | Arrives first; feels like a thud |
| **S (Secondary)** | Shear ⊥ | 57–60% of Vp | 2–6 Hz | Arrives after P; main shaking |
| **Rayleigh** | Elliptical (retrograde) | ~92% of Vs | 0.3–1 Hz | Long duration; rolling motion |
| **Love** | Horizontal transverse | ~ Vs | 0.3–1 Hz | Horizontal shaking; surface only |

#### Wadati method for epicentral distance
The S–P time difference is independent of origin time and depends only on distance:
""")
        st.latex(r"t_S - t_P = R \left(\frac{1}{V_S} - \frac{1}{V_P}\right) = R \cdot \frac{V_P - V_S}{V_P V_S}")
        st.latex(r"\Rightarrow \quad d \approx (t_S - t_P) \cdot \frac{V_P \cdot V_S}{V_P - V_S}")
        st.markdown("""
For the IASP91 model (upper crust: Vp = 5.80, Vs = 3.36 km/s):
Δt of **1 second** ≈ **8 km** epicentral distance.
""")

    # ── 6. Magnitude Scales ───────────────────────────────────────────────────
    with st.expander("📏 6 — Earthquake Magnitude Scales"):
        st.markdown("""
#### Richter Local Magnitude (M_L)
Defined by Charles Richter (1935) for Southern California. Uses the peak amplitude on a Wood-Anderson
seismograph, corrected for distance with an empirical attenuation curve:
""")
        st.latex(r"M_L = \log_{10}(A) - \log_{10}(A_0(\Delta))")
        st.markdown("""
Where A is peak amplitude (μm) and A₀(Δ) is the empirical correction for epicentral distance Δ.

**SeismicLens** scales synthetic amplitudes as:
""")
        st.latex(r"A_{peak} \propto 10^{0.8 M - 2.5}")
        st.markdown("""
This approximates the Gutenberg-Richter amplitude-magnitude relation.

#### Moment Magnitude (M_w)
The modern standard (Hanks & Kanamori, 1979), calibrated to be numerically close to M_L:
""")
        st.latex(r"M_w = \frac{2}{3} \log_{10}(M_0) - 10.7")
        st.latex(r"M_0 = \mu \cdot A_{fault} \cdot \bar{D} \quad [\text{N·m}]")
        st.markdown("""
Where:
- **M₀** = seismic moment (Nm) — total energy released at the fault
- **μ** = rigidity / shear modulus of the surrounding rock (~3×10¹⁰ Pa)
- **A_fault** = rupture area (m²)
- **D̄** = average slip on the fault (m)

M_w has **no saturation** at large magnitudes (unlike M_L) and is thus the preferred scale for large events.

| M_w | Energy (joules) | Equivalent |
|---|---|---|
| 2.0 | 10¹¹ J | Small quarry blast |
| 5.0 | 10¹⁴ J | Hiroshima bomb |
| 7.0 | 10¹⁶ J | Large destructive quake |
| 9.0 | 10¹⁸ J | 2011 Tōhoku earthquake |
""")

    # ── 7. Numerical Methods ──────────────────────────────────────────────────
    with st.expander("🖥️ 7 — Numerical Methods & Implementation"):
        st.markdown("""
#### Signal tapering (Tukey window)
Before FFT, the signal is tapered at both ends (5% cosine taper) to force it to smoothly reach zero.
This prevents **spectral leakage** caused by the abrupt truncation of the signal.
""")
        st.latex(r"w_{Tukey}[n] = \begin{cases} \frac{1}{2}\left[1 - \cos\!\left(\frac{2\pi n}{\alpha N}\right)\right] & 0 \le n < \frac{\alpha N}{2} \\ 1 & \frac{\alpha N}{2} \le n < N\left(1-\frac{\alpha}{2}\right) \\ \frac{1}{2}\left[1 - \cos\!\left(\frac{2\pi(N-n)}{\alpha N}\right)\right] & N\left(1-\frac{\alpha}{2}\right) \le n \le N \end{cases}")
        st.markdown("""
Where α = 0.10 (10% total taper — 5% each end).

#### Spectral centroid and bandwidth
The spectral centroid (first moment) and bandwidth (second central moment) are computed on
the one-sided amplitude spectrum A[k]:
""")
        st.latex(r"f_c = \frac{\sum_k f_k \cdot A[k]}{\sum_k A[k]}")
        st.latex(r"\sigma_f = \sqrt{\frac{\sum_k (f_k - f_c)^2 \cdot A[k]}{\sum_k A[k]}}")
        st.markdown("""
These are the frequency-domain analogues of the mean and standard deviation of a probability distribution.

#### Welch PSD normalisation
The Welch PSD is normalised so that Parseval's theorem holds in the frequency domain:
""")
        st.latex(r"\int_{-\infty}^{\infty} S(f)\, df = \text{var}(x) = \frac{1}{N}\sum_{n} x^2[n]")
        st.markdown("""
This means the integral of the PSD equals the signal variance — a useful sanity check.
""")

    # ── References ────────────────────────────────────────────────────────────
    st.markdown("""
<div class='sl-card' style='margin-top:20px;'>
<div class='theory-title'>📚 Key References</div>
<div class='theory-body' style='font-size:12px; line-height:1.8;'>
• Allen, R.V. (1978). Automatic earthquake recognition and timing from single traces. <em>BSSA</em>, 68(5).<br>
• Butterworth, S. (1930). On the theory of filter amplifiers. <em>Wireless Engineer</em>, 7, 536–541.<br>
• Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine calculation of complex Fourier series. <em>Math. Computation</em>, 19, 297–301.<br>
• Hanks, T.C. & Kanamori, H. (1979). A moment magnitude scale. <em>JGR</em>, 84(B5), 2348–2350.<br>
• Kennett, B.L.N. & Engdahl, E.R. (1991). Traveltimes for global earthquake location (IASP91). <em>GJI</em>, 105, 429–465.<br>
• Richter, C.F. (1935). An instrumental earthquake magnitude scale. <em>BSSA</em>, 25(1), 1–32.<br>
• Wadati, K. (1933). On the travel time of earthquake waves. <em>Geophys. Mag.</em>, 7, 101–111.<br>
• Welch, P.D. (1967). The use of Fast Fourier Transform for the estimation of power spectra. <em>IEEE TAES</em>, 15, 70–73.
</div>
</div>""", unsafe_allow_html=True)


# ── Export ────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown(f"<div class='section-title'>{T('tab_export', lang)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tab-intro'>{T('export_desc', lang)}</div>", unsafe_allow_html=True)

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

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    _next_steps = {
        "en": "📌 What to do with the exported data",
        "it": "📌 Cosa fare con i dati esportati",
        "fr": "📌 Que faire avec les données exportées",
        "es": "📌 Qué hacer con los datos exportados",
        "de": "📌 Was mit den exportierten Daten tun",
    }
    st.markdown(f"<div class='section-title'>{_next_steps[lang]}</div>", unsafe_allow_html=True)

    with st.expander("🐍 Python code snippets"):
        st.code("""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load the signal CSV ─────────────────────────────────────────────────────
df = pd.read_csv('seismiclens_signal.csv')
t, amp = df['time_s'].values, df['amplitude_counts'].values

plt.figure(figsize=(12, 3))
plt.plot(t, amp, lw=0.8, color='steelblue')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (counts)')
plt.title('SeismicLens — Filtered Waveform')
plt.tight_layout(); plt.show()

# ── Load the FFT CSV ────────────────────────────────────────────────────────
fft = pd.read_csv('seismiclens_fft.csv')
plt.figure(figsize=(10, 3))
plt.semilogy(fft['frequency_hz'], fft['amplitude'])
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
plt.tight_layout(); plt.show()

# ── Load the PSD CSV ────────────────────────────────────────────────────────
psd = pd.read_csv('seismiclens_psd.csv')
plt.figure(figsize=(10, 3))
plt.plot(psd['frequency_hz'], psd['psd_db'])
plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD (dB)')
plt.title('Welch Power Spectral Density')
plt.tight_layout(); plt.show()

# ── Dominant frequency from FFT ─────────────────────────────────────────────
dominant_f = fft.loc[fft['amplitude'].idxmax(), 'frequency_hz']
print(f'Dominant frequency: {dominant_f:.3f} Hz')
""", language="python")

        st.code("""
# ── Load & plot STA/LTA ─────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

sl = pd.read_csv('seismiclens_stalta.csv')
plt.figure(figsize=(12, 2))
plt.plot(sl['time_s'], sl['stalta_ratio'], color='purple', lw=0.9)
plt.axhline(y=3.5, ls='--', color='red', label='threshold=3.5')
plt.xlabel('Time (s)'); plt.ylabel('STA/LTA')
plt.legend(); plt.tight_layout(); plt.show()

# First trigger time
threshold = 3.5
trigger_mask = sl['stalta_ratio'] > threshold
if trigger_mask.any():
    p_arrival = sl.loc[trigger_mask.idxmax(), 'time_s']
    print(f'P-wave arrival: {p_arrival:.2f} s')
""", language="python")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style='text-align:center; padding: 32px 0 16px 0;'>
  <div style='font-family: JetBrains Mono, monospace; font-size:11px; color:rgba(255,255,255,0.2);
              border-top:1px solid rgba(255,255,255,0.06); padding-top:16px;'>
    SeismicLens v2.1 &nbsp;·&nbsp; ObsPy · SciPy · NumPy · Plotly · Streamlit
    &nbsp;·&nbsp; © 2024–2026
  </div>
</div>""", unsafe_allow_html=True)

