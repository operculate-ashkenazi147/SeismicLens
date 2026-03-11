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
    "filter_toggle":    {"en": "Zero-phase bandpass", "it": "Passa-banda a fase zero", "fr": "Passe-bande zéro-phase", "es": "Pasa-banda fase cero", "de": "Nullphasen-Bandpass"},
    "f_low":            {"en": "Low cut (Hz)", "it": "Taglio basso (Hz)", "fr": "Coupure basse (Hz)", "es": "Corte bajo (Hz)", "de": "Untere Grenzfrequenz (Hz)"},
    "f_high":           {"en": "High cut (Hz)", "it": "Taglio alto (Hz)", "fr": "Coupure haute (Hz)", "es": "Corte alto (Hz)", "de": "Obere Grenzfrequenz (Hz)"},
    "filter_order":     {"en": "Order", "it": "Ordine", "fr": "Ordre", "es": "Orden", "de": "Ordnung"},
    "stalta_section":   {"en": "STA/LTA Detector", "it": "Rilevatore STA/LTA", "fr": "Détecteur STA/LTA", "es": "Detector STA/LTA", "de": "STA/LTA-Detektor"},
    "sta_window":       {"en": "STA window (s)", "it": "Finestra STA (s)", "fr": "Fenêtre STA (s)", "es": "Ventana STA (s)", "de": "STA-Fenster (s)"},
    "lta_window":       {"en": "LTA window (s)", "it": "Finestra LTA (s)", "fr": "Fenêtre LTA (s)", "es": "Ventana LTA (s)", "de": "LTA-Fenster (s)"},
    "threshold":        {"en": "Trigger threshold", "it": "Soglia di trigger", "fr": "Seuil de déclenchement", "es": "Umbral de disparo", "de": "Auslöseschwelle"},
    "display_section":  {"en": "Display Options", "it": "Opzioni di visualizzazione", "fr": "Options d'affichage", "es": "Opciones de visualización", "de": "Anzeigeoptionen"},
    "show_raw":         {"en": "Raw signal overlay", "it": "Sovrapposizione segnale grezzo", "fr": "Superposition signal brut", "es": "Superposición señal bruta", "de": "Rohsignal einblenden"},
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
        "it": "Filtro Butterworth a fase zero — dettagli",
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
        "it": "Il grafico mostra l'ampiezza (moto del suolo in counts) nel tempo. Le onde P arrivano per prime (compressive), le onde S per seconde (di taglio). Il filtro Butterworth passa-banda elimina il rumore fuori dalla banda di frequenza scelta senza distorcere i tempi di arrivo.",
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
        "it": "Lo spettrogramma mostra come il contenuto in frequenza cambia nel tempo. È calcolato tramite STFT: il segnale è diviso in finestre brevi e sovrapposte, la FFT è calcolata per ciascuna, e i risultati formano una mappa di calore 2D. Colori più accesi = più energia.",
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
        "it": "Questa tabella mostra un modello di velocità di crosta a strati ispirato a IASP91, usato per generare i segnali sintetici. Ogni strato è definito dalla velocità dell'onda P (Vp), onda S (Vs) e densità (ρ). Queste proprietà dipendono dalla mineralogia e dalla pressione.",
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
        "it": "I sismogrammi reali contengono rumore strumentale, rumore microsismico (onde oceaniche, 0.1–0.3 Hz), rumore culturale (traffico, 1–20 Hz) e rumore elettronico. Un filtro passa-banda mantiene solo la banda di frequenza rilevante per il terremoto studiato, migliorando il SNR e rendendo più precisi i pick delle fasi.",
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
    "stalta_scenario":  {"en": "Scenario", "it": "Scenario", "fr": "Scenario", "es": "Escenario", "de": "Szenario"},
    "stalta_sc1":       {"en": "Local earthquake", "it": "Terremoto locale", "fr": "Seisme local", "es": "Sismo local", "de": "Lokales Erdbeben"},
    "stalta_sc2":       {"en": "Regional earthquake", "it": "Terremoto regionale", "fr": "Seisme regional", "es": "Sismo regional", "de": "Regionales Erdbeben"},
    "stalta_sc3":       {"en": "Teleseismic", "it": "Telesismico", "fr": "Telesismique", "es": "Telesismico", "de": "Teleseismisch"},
    "stalta_sc4":       {"en": "Very noisy data", "it": "Dati molto rumorosi", "fr": "Donnees tres bruyantes", "es": "Datos muy ruidosos", "de": "Sehr verrauschte Daten"},
    "stalta_samples":   {"en": "samples", "it": "campioni", "fr": "echantillons", "es": "muestras", "de": "Abtastwerte"},
    "theory_header_1":  {"en": "1 — Discrete Fourier Transform (DFT)", "it": "1 — Trasformata di Fourier discreta (DFT)", "fr": "1 — Transformee de Fourier discrete (DFT)", "es": "1 — Transformada de Fourier discreta (DFT)", "de": "1 — Diskrete Fourier-Transformation (DFT)"},
    "theory_header_2":  {"en": "2 — Butterworth Filter Design", "it": "2 — Progettazione del filtro Butterworth", "fr": "2 — Conception du filtre Butterworth", "es": "2 — Diseno del filtro Butterworth", "de": "2 — Butterworth-Filterentwurf"},
    "theory_header_3":  {"en": "3 — Power Spectral Density (PSD)", "it": "3 — Densità spettrale di potenza (PSD)", "fr": "3 — Densite spectrale de puissance (DSP)", "es": "3 — Densidad espectral de potencia (PSD)", "de": "3 — Leistungsspektraldichte (PSD)"},
    "theory_header_4":  {"en": "4 — Short-Time Fourier Transform (STFT)", "it": "4 — Trasformata di Fourier a tempo breve (STFT)", "fr": "4 — Transformee de Fourier a court terme (STFT)", "es": "4 — Transformada de Fourier de tiempo reducido (STFT)", "de": "4 — Kurzzeit-Fourier-Transformation (STFT)"},
    "theory_header_5":  {"en": "5 — Seismic Wave Physics", "it": "5 — Fisica delle onde sismiche", "fr": "5 — Physique des ondes sismiques", "es": "5 — Fisica de las ondas sismicas", "de": "5 — Seismische Wellenphysik"},
    "theory_header_6":  {"en": "6 — Earthquake Magnitude Scales", "it": "6 — Scale di magnitudo dei terremoti", "fr": "6 — Echelles de magnitude des seismes", "es": "6 — Escalas de magnitud de terremotos", "de": "6 — Erdbeben-Magnitudenskalen"},
    "theory_header_7":  {"en": "7 — Numerical Methods & Implementation", "it": "7 — Metodi numerici e implementazione", "fr": "7 — Methodes numeriques et mise en oeuvre", "es": "7 — Metodos numericos e implementacion", "de": "7 — Numerische Methoden & Implementierung"},
    "theory_waves_title": {"en": "Types of seismic waves", "it": "Tipi di onde sismiche", "fr": "Types d'ondes sismiques", "es": "Tipos de ondas sismicas", "de": "Arten von seismischen Wellen"},
    "theory_wadati_title": {"en": "Wadati method for epicentral distance", "it": "Metodo di Wadati per la distanza epicentrale", "fr": "Methode de Wadati pour la distance epicentrale", "es": "Metodo de Wadati para la distancia epicentral", "de": "Wadati-Methode fur Epizentraldistanz"},
    "theory_richter_title": {"en": "Richter Local Magnitude (M_L)", "it": "Magnitudo locale Richter (M_L)", "fr": "Magnitude locale de Richter (M_L)", "es": "Magnitud local de Richter (M_L)", "de": "Richter-Lokalmagnitude (M_L)"},
    "theory_moment_title": {"en": "Moment Magnitude (M_w)", "it": "Magnitudo di momento (M_w)", "fr": "Magnitude de moment (M_w)", "es": "Magnitud de momento (M_w)", "de": "Momenten-Magnitude (M_w)"},
    "theory_taper_title": {"en": "Signal tapering (Tukey window)", "it": "Tapering del segnale (finestra di Tukey)", "fr": "Fenetrage du signal (fenetre de Tukey)", "es": "Enventanado de senal (ventana de Tukey)", "de": "Signal-Tapering (Tukey-Fenster)"},
    "theory_centroid_title": {"en": "Spectral centroid and bandwidth", "it": "Centroide spettrale e larghezza di banda", "fr": "Centroide spectral et largeur de bande", "es": "Centroide espectral y ancho de banda", "de": "Spektraler Schwerpunkt und Bandbreite"},
    "theory_psd_norm_title": {"en": "Welch PSD normalisation", "it": "Normalizzazione della PSD di Welch", "fr": "Normalisation de la DSP de Welch", "es": "Normalizacion de la PSD de Welch", "de": "Welch-PSD-Normalisierung"},
    "theory_ref_title": {"en": "Key References", "it": "Riferimenti chiave", "fr": "References cles", "es": "Referencias clave", "de": "Wichtige Referenzen"},
    "theory_read_spec_title": {"en": "Reading seismic spectrograms", "it": "Leggere gli spettrogrammi sismici", "fr": "Lire les spectrogrammes sismiques", "es": "Leer espectrogramas sismicos", "de": "Seismische Spektrogramme lesen"},
    "theory_read_spec_1": {"en": "Vertical bright streaks at early time -> P or S wave arrival (broadband, impulsive)", "it": "Strisce verticali luminose all'inizio -> arrivo onde P o S (banda larga, impulsivo)", "fr": "Bandes verticales lumineuses au debut -> arrivee d'ondes P ou S (large bande, impulsif)", "es": "Franjas verticales brillantes al principio -> llegada de ondas P o S (banda ancha, impulsiva)", "de": "Vertikale helle Streifen zu Beginn -> P- oder S-Wellenankunft (breitbandig, impulsiv)"},
    "theory_read_spec_2": {"en": "Horizontal bright bands -> monochromatic noise (e.g., 50/60 Hz power line)", "it": "Bande orizzontali luminose -> rumore monocromatico (es. linea elettrica 50/60 Hz)", "fr": "Bandes horizontales lumineuses -> bruit monochromatique (ex. ligne electrique 50/60 Hz)", "es": "Bandas horizontales brillantes -> ruido monocromatico (ej. linea electrica 50/60 Hz)", "de": "Horizontale helle Bander -> monochromatisches Rauschen (z. B. 50/60 Hz Stromleitung)"},
    "theory_read_spec_3": {"en": "Energy that sweeps from high to low frequency over time -> dispersive surface wave (Love or Rayleigh)", "it": "Energia che scorre da alta a bassa frequenza nel tempo -> onda superficiale dispersiva (Love o Rayleigh)", "fr": "Energie qui balaie des hautes vers les basses frequences -> onde de surface dispersive (Love ou Rayleigh)", "es": "Energia que recorre de alta a baja frecuencia en el tiempo -> onda superficial dispersiva (Love o Rayleigh)", "de": "Energie, die im Laufe der Zeit von hohen zu niedrigen Frequenzen sweep -> dispersive Oberflachenwelle (Love oder Rayleigh)"},
    "theory_read_spec_4": {"en": "Low-frequency stripe throughout -> ocean microseisms (0.1-0.3 Hz)", "it": "Striscia a bassa frequenza costante -> microsismicità oceanica (0.1-0.3 Hz)", "fr": "Bande basse frequence constante -> microseismes oceaniques (0.1-0.3 Hz)", "es": "Franja de baja frecuencia constante -> microsismos oceanicos (0.1-0.3 Hz)", "de": "Konstanter niederfrequenter Streifen -> ozeanische Mikroseismik (0,1-0,3 Hz)"},
    "syn_desc": {
        "en": "Generate a physically realistic synthetic seismogram using an IASP91-like crustal model. The simulator models P-waves (6-12 Hz), S-waves (2-6 Hz) and surface waves (0.3-1 Hz) with amplitudes scaled by magnitude and distances from travel-time equations.",
        "it": "Genera un sismogramma sintetico fisicamente realistico usando un modello di crosta simile a IASP91. Il simulatore modella le onde P (6-12 Hz), le onde S (2-6 Hz) e le onde superficiali (0.3-1 Hz) con ampiezze scalate per la magnitudo e distanze calcolate dalle equazioni dei tempi di percorrenza.",
        "fr": "Genere un sismogramme synthetique physiquement realiste en utilisant un modele crustal de type IASP91. Le simulateur modelise les ondes P (6-12 Hz), S (2-6 Hz) et de surface (0.3-1 Hz) avec des amplitudes mises a l'echelle par la magnitude.",
        "es": "Genera un sismograma sintetico fisicamente realista usando un modelo cortical tipo IASP91. El simulador modela ondas P (6-12 Hz), ondas S (2-6 Hz) y ondas superficiales (0.3-1 Hz) con amplitudes escaladas por la magnitud.",
        "de": "Erzeugt ein physikalisch realistisches synthetisches Seismogramm unter Verwendung eines IASP91-ahnlichen Krustenmodells. Der Simulator modelliert P-Wellen (6-12 Hz), S-Wellen (2-6 Hz) und Oberflachenwellen (0.3-1 Hz).",
    },
    "about_title":      {"en": "About", "it": "Informazioni", "fr": "A propos", "es": "Acerca de", "de": "Uber"},
    "next_steps_title": {"en": "Next steps for your analysis", "it": "Prossimi passi per la tua analisi", "fr": "Prochaines etapes de votre analyse", "es": "Proximos pasos para su analisis", "de": "Nachste Schritte fur deine Analyse"},
    "error_mseed":      {"en": "Error loading MiniSEED", "it": "Errore nel caricamento del MiniSEED", "fr": "Erreur lors du chargement du MiniSEED", "es": "Error al cargar MiniSEED", "de": "Fehler beim Laden von MiniSEED"},
    "error_csv":        {"en": "Error loading CSV", "it": "Errore nel caricamento del CSV", "fr": "Erreur lors du chargement du CSV", "es": "Error al cargar CSV", "de": "Fehler beim Laden von CSV"},
    "error_filter":     {"en": "Filter error", "it": "Errore del filtro", "fr": "Erreur de filtrage", "es": "Error de filtro", "de": "Filterfehler"},
    "error_stalta":     {"en": "STA/LTA error", "it": "Errore STA/LTA", "fr": "Erreur STA/LTA", "es": "Error STA/LTA", "de": "STA/LTA-Fehler"},
    "theme_toggle":     {"en": "☀ Light theme", "it": "☀ Tema chiaro", "fr": "☀ Thème clair", "es": "☀ Tema claro", "de": "☀ Helles Design"},
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
    page_icon="[S]",
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
# Light theme CSS (injected conditionally)
# ─────────────────────────────────────────────────────────────────────────────
_LIGHT_CSS = """
<style>
/* ════════════════════════ LIGHT THEME OVERRIDES ════════════════════════ */
.stApp {
    background: linear-gradient(145deg, #f0f4ff 0%, #ffffff 40%, #eaf3ff 100%) !important;
    color: #1a2030 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(240,244,255,0.98) !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}

/* Generic text */
html, body, [class*="css"] { color: #1a2030 !important; }

/* Cards */
.sl-card {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.sl-card:hover { border-color: rgba(30,100,220,0.35) !important; }

.metric-card {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.metric-card:hover {
    background: rgba(30,100,220,0.06) !important;
    border-color: rgba(30,100,220,0.3) !important;
}
.metric-label { color: rgba(0,0,0,0.4) !important; }
.metric-value { color: #1a64dc !important; }
.metric-unit  { color: rgba(0,0,0,0.35) !important; }
.metric-card.green  .metric-value { color: #1a8a3a !important; }
.metric-card.amber  .metric-value { color: #a06000 !important; }
.metric-card.violet .metric-value { color: #7030b0 !important; }
.metric-card.coral  .metric-value { color: #c03020 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg,
        rgba(30,100,220,0.10) 0%,
        rgba(20,140,50,0.05) 50%,
        rgba(110,60,200,0.07) 100%) !important;
    border: 1px solid rgba(30,100,220,0.18) !important;
    box-shadow: 0 4px 20px rgba(30,100,220,0.08);
}
.hero::before { background: radial-gradient(circle, rgba(30,100,220,0.12) 0%, transparent 70%) !important; }
.hero h1      { color: #1a2030 !important; }
.hero-sub     { color: rgba(0,0,0,0.5) !important; }
.hero-badge   { background: rgba(30,100,220,0.1) !important; border-color: rgba(30,100,220,0.25) !important; color: #1a64dc !important; }

.hero-tag     { background: rgba(0,0,0,0.04) !important; border-color: rgba(0,0,0,0.09) !important; color: rgba(0,0,0,0.45) !important; }
.hero-tag.ht-blue   { background: rgba(30,100,220,0.08)  !important; border-color: rgba(30,100,220,0.2)  !important; color: #1a64dc  !important; }
.hero-tag.ht-green  { background: rgba(20,140,50,0.08)   !important; border-color: rgba(20,140,50,0.2)   !important; color: #1a8a3a  !important; }
.hero-tag.ht-violet { background: rgba(110,60,200,0.08)  !important; border-color: rgba(110,60,200,0.2)  !important; color: #7030b0  !important; }
.hero-tag.ht-amber  { background: rgba(160,96,0,0.08)    !important; border-color: rgba(160,96,0,0.2)    !important; color: #a06000  !important; }

/* Section title */
.section-title { color: rgba(0,0,0,0.35) !important; border-bottom: 1px solid rgba(0,0,0,0.07) !important; }

/* Info / warn / success boxes */
.info-box    { background: rgba(30,100,220,0.06) !important;  border-left-color: #1a64dc !important; color: rgba(0,0,0,0.65) !important; }
.warn-box    { background: rgba(160,96,0,0.07)   !important;  border-left-color: #a06000 !important; color: rgba(0,0,0,0.65) !important; }
.success-box { background: rgba(20,140,50,0.07)  !important;  border-left-color: #1a8a3a !important; color: rgba(0,0,0,0.65) !important; }

/* Math block */
.math-block {
    background: rgba(240,244,255,0.9) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #1a8a3a !important;
}

/* Theory cards */
.theory-card  { background: rgba(255,255,255,0.8) !important; border: 1px solid rgba(0,0,0,0.07) !important; }
.theory-title { color: #a06000 !important; }
.theory-body  { color: rgba(0,0,0,0.62) !important; }

/* Velocity table */
.vel-table th { color: rgba(0,0,0,0.38) !important; }
.vel-table td { background: rgba(255,255,255,0.7) !important; color: rgba(0,0,0,0.62) !important; border-color: rgba(0,0,0,0.06) !important; }
.vp { color: #1a64dc !important; }
.vs { color: #1a8a3a !important; }

/* Sidebar section headers */
.sb-section { color: rgba(0,0,0,0.3) !important; border-top: 1px solid rgba(0,0,0,0.07) !important; }
.sb-tip     { background: rgba(20,140,50,0.06) !important; color: rgba(0,0,0,0.5) !important; }

/* Label override */
label { color: rgba(0,0,0,0.6) !important; }

/* Plotly border */
.js-plotly-plot { border: 1px solid rgba(0,0,0,0.07) !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(30,100,220,0.09) !important;
    border: 1px solid rgba(30,100,220,0.25) !important;
    color: #1a64dc !important;
}
.stDownloadButton > button:hover { background: rgba(30,100,220,0.18) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(0,0,0,0.04) !important; border: 1px solid rgba(0,0,0,0.07) !important; }
.stTabs [data-baseweb="tab"]      { color: rgba(0,0,0,0.45) !important; }
.stTabs [aria-selected="true"]    { background: rgba(30,100,220,0.12) !important; color: #1a64dc !important; }

/* Tab intro */
.tab-intro { background: linear-gradient(90deg, rgba(30,100,220,0.07) 0%, rgba(20,140,50,0.04) 100%) !important; border-color: rgba(30,100,220,0.12) !important; color: rgba(0,0,0,0.62) !important; }
.tab-intro strong { color: #1a2030 !important; }

/* Pills */
.pill        { background: rgba(0,0,0,0.04) !important; border-color: rgba(0,0,0,0.1) !important; color: rgba(0,0,0,0.5) !important; }
.pill.blue   { background: rgba(30,100,220,0.08)  !important; border-color: rgba(30,100,220,0.22)  !important; color: #1a64dc  !important; }
.pill.green  { background: rgba(20,140,50,0.08)   !important; border-color: rgba(20,140,50,0.22)   !important; color: #1a8a3a  !important; }
.pill.amber  { background: rgba(160,96,0,0.08)    !important; border-color: rgba(160,96,0,0.22)    !important; color: #a06000  !important; }
.pill.violet { background: rgba(110,60,200,0.08)  !important; border-color: rgba(110,60,200,0.22)  !important; color: #7030b0  !important; }
.pill.coral  { background: rgba(192,48,32,0.08)   !important; border-color: rgba(192,48,32,0.22)   !important; color: #c03020  !important; }

/* Concept cards */
.concept-card       { background: rgba(255,255,255,0.8) !important; border-color: rgba(0,0,0,0.07) !important; }
.concept-card .cc-title { color: rgba(0,0,0,0.35) !important; }
.concept-card .cc-body  { color: rgba(0,0,0,0.6)  !important; }

/* Inline formula */
.f-inline { background: rgba(230,240,255,0.9) !important; color: #1a8a3a !important; }

/* Gradient divider */
.grad-div { background: linear-gradient(90deg, transparent, rgba(30,100,220,0.28), transparent) !important; }

/* Step cards */
.step-card        { background: rgba(255,255,255,0.75) !important; border-color: rgba(0,0,0,0.07) !important; }
.step-card:hover  { border-color: rgba(30,100,220,0.25) !important; }
.step-icon        { background: rgba(30,100,220,0.12) !important; color: #1a64dc !important; }
.step-body        { color: rgba(0,0,0,0.6) !important; }
.step-body strong { color: rgba(0,0,0,0.85) !important; }

/* Key concept callout */
.key-concept       { background: linear-gradient(135deg, rgba(110,60,200,0.07), rgba(30,100,220,0.05)) !important; border-color: rgba(110,60,200,0.16) !important; }
.key-concept .kc-title { color: #7030b0 !important; }
.key-concept .kc-body  { color: rgba(0,0,0,0.62) !important; }

/* Stat row */
.stat-item          { background: rgba(255,255,255,0.8) !important; border-color: rgba(0,0,0,0.07) !important; }
.stat-item .si-val  { color: #1a64dc !important; }
.stat-item .si-lbl  { color: rgba(0,0,0,0.35) !important; }

/* Guide box */
.guide-box    { background: rgba(30,100,220,0.05) !important; border-color: rgba(30,100,220,0.14) !important; }
.guide-box h4 { color: #1a64dc !important; }
.guide-step   { color: rgba(0,0,0,0.6) !important; }
.guide-num    { background: rgba(30,100,220,0.18) !important; color: #1a64dc !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── Theme-aware palette ───────────────────────────────────────────────────────
def _is_light():
    return st.session_state.get("light_theme", False)

def _plot_base():
    if _is_light():
        return dict(
            paper_bgcolor="rgba(255,255,255,0.0)",
            plot_bgcolor="rgba(240,244,255,0.55)",
            font=dict(family="JetBrains Mono, monospace", color="rgba(0,0,0,0.45)", size=11),
            margin=dict(l=56, r=20, t=44, b=44),
            xaxis=dict(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)",
                       linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)",
                       linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=10)),
        )
    return dict(
        paper_bgcolor="rgba(13,17,23,0.0)",
        plot_bgcolor="rgba(0,0,0,0.25)",
        font=dict(family="JetBrains Mono, monospace", color="rgba(255,255,255,0.5)", size=11),
        margin=dict(l=56, r=20, t=44, b=44),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10)),
    )

# Colour aliases (dark / light)
def _c(dark, light):
    return light if _is_light() else dark

def _title(text):
    title_color = "rgba(0,0,0,0.75)" if _is_light() else "rgba(255,255,255,0.8)"
    return dict(text=text, font=dict(color=title_color, size=13), x=0.0, xanchor="left")


def make_waveform_fig(t, signal, raw=None, p_time=None, s_time=None, title="Waveform"):
    fig = go.Figure()
    if raw is not None:
        raw_color = "rgba(0,0,0,0.15)" if _is_light() else "rgba(255,255,255,0.15)"
        fig.add_trace(go.Scatter(x=t, y=raw, mode="lines",
                                 line=dict(color=raw_color, width=0.9),
                                 name="Raw", opacity=0.6))
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines",
                             line=dict(color=_c("#58a6ff", "#1a64dc"), width=1.4), name="Signal"))
    if p_time is not None:
        p_col = _c("#f78166", "#c03020")
        fig.add_vline(x=p_time, line_width=2, line_dash="dash", line_color=p_col,
                      annotation_text="  P", annotation_font_color=p_col, annotation_font_size=12)
    if s_time is not None:
        s_col = _c("#3fb950", "#1a8a3a")
        fig.add_vline(x=s_time, line_width=2, line_dash="dash", line_color=s_col,
                      annotation_text="  S", annotation_font_color=s_col, annotation_font_size=12)
    legend_fc = "rgba(0,0,0,0.45)" if _is_light() else "rgba(255,255,255,0.5)"
    fig.update_layout(**_plot_base(), title=_title(title),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (counts)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  font=dict(size=10, color=legend_fc)))
    return fig


def make_fft_fig(freqs, amplitudes, dominant_f, centroid_f=None):
    line_col = _c("#3fb950", "#1a8a3a")
    fill_col = _c("rgba(63,185,80,0.07)", "rgba(20,140,50,0.08)")
    dom_col  = _c("#d29922", "#a06000")
    cen_col  = _c("#bc8cff", "#7030b0")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=amplitudes, mode="lines", fill="tozeroy",
                             line=dict(color=line_col, width=1.6),
                             fillcolor=fill_col, name="|A(f)|"))
    fig.add_vline(x=dominant_f, line_width=1.8, line_dash="dot", line_color=dom_col,
                  annotation_text=f"  Peak {dominant_f:.2f} Hz",
                  annotation_font_color=dom_col, annotation_font_size=11)
    if centroid_f is not None:
        fig.add_vline(x=centroid_f, line_width=1.4, line_dash="dash", line_color=cen_col,
                      annotation_text=f"  Centroid {centroid_f:.2f} Hz",
                      annotation_font_color=cen_col, annotation_font_size=11,
                      annotation_position="bottom right")
    fig.update_layout(**_plot_base(), title=_title("FFT — Amplitude Spectrum"),
                      xaxis_title="Frequency (Hz)", yaxis_title="|A(f)|  (counts)")
    return fig

def make_phase_fig(freqs, phases_deg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=phases_deg, mode="lines",
                             line=dict(color=_c("#bc8cff", "#7030b0"), width=1.2), name="Phase"))
    fig.update_layout(**_plot_base(), title=_title("FFT — Phase Spectrum  ∠X(f)"),
                      xaxis_title="Frequency (Hz)", yaxis_title="Phase (deg)")
    return fig


def make_psd_fig(f, psd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=10 * np.log10(psd + 1e-20), mode="lines", fill="tozeroy",
                             line=dict(color=_c("#d29922", "#a06000"), width=1.4),
                             fillcolor=_c("rgba(210,153,34,0.06)", "rgba(160,96,0,0.07)"), name="PSD"))
    fig.update_layout(**_plot_base(), title=_title("Welch PSD  S(f)"),
                      xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB re 1 count²/Hz)")
    return fig


def make_spectrogram_fig(t, f, Sxx):
    z = 10 * np.log10(Sxx + 1e-20)
    cb_fc = "rgba(0,0,0,0.45)" if _is_light() else "rgba(255,255,255,0.4)"
    colorscale = "Viridis" if _is_light() else "Inferno"
    fig = go.Figure(go.Heatmap(x=t, y=f, z=z, colorscale=colorscale,
                               colorbar=dict(title="dB", tickfont=dict(color=cb_fc, size=10)),
                               zmin=np.percentile(z, 5), zmax=np.percentile(z, 99)))
    fig.update_layout(**_plot_base(), title=_title("Spectrogram (STFT)"),
                      xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    return fig


def make_stalta_fig(t, stalta, threshold):
    sta_col  = _c("#bc8cff", "#7030b0")
    thr_col  = _c("#f78166", "#c03020")
    fill_col = _c("rgba(247,129,102,0.12)", "rgba(192,48,32,0.10)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=stalta, mode="lines",
                             line=dict(color=sta_col, width=1.4), name="STA/LTA"))
    trigger_mask = stalta > threshold
    if trigger_mask.any():
        y_fill = np.where(trigger_mask, stalta, np.nan)
        fig.add_trace(go.Scatter(x=t, y=y_fill, mode="lines", fill="tozeroy",
                                 line=dict(color="rgba(0,0,0,0)", width=0),
                                 fillcolor=fill_col,
                                 name="Triggered", showlegend=False))
    fig.add_hline(y=threshold, line_width=1.6, line_dash="dash", line_color=thr_col,
                  annotation_text=f"  Threshold = {threshold}",
                  annotation_font_color=thr_col, annotation_font_size=11)
    fig.update_layout(**_plot_base(), title=_title("STA/LTA Characteristic Function"),
                      xaxis_title="Time (s)", yaxis_title="STA/LTA ratio")
    return fig


def make_velocity_model_fig(layer_names):
    vp_vals = [l[2] for l in CRUSTAL_LAYERS]
    vs_vals = [l[3] for l in CRUSTAL_LAYERS]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=layer_names, x=vp_vals, orientation="h", name="Vp (km/s)",
                         marker_color=_c("#58a6ff", "#1a64dc"),
                         text=[f"{v} km/s" for v in vp_vals], textposition="auto"))
    fig.add_trace(go.Bar(y=layer_names, x=vs_vals, orientation="h", name="Vs (km/s)",
                         marker_color=_c("#3fb950", "#1a8a3a"),
                         text=[f"{v} km/s" for v in vs_vals], textposition="auto"))
    legend_fc = "rgba(0,0,0,0.45)" if _is_light() else "rgba(255,255,255,0.5)"
    fig.update_layout(**_plot_base(), title=_title("Crustal Velocity Model (IASP91-like)"),
                      xaxis_title="Velocity (km/s)", barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  font=dict(color=legend_fc)))
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

    light_theme = st.toggle(T("theme_toggle", lang), value=False, key="light_theme")

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
        "en": "Typical seismology bands: teleseismic 0.01-2 Hz · regional 0.5-5 Hz · local 1-15 Hz",
        "it": "Bande tipiche: teleseismi 0.01-2 Hz · regionali 0.5-5 Hz · locali 1-15 Hz",
        "fr": "Bandes typiques : teleseismique 0.01-2 Hz · regional 0.5-5 Hz · local 1-15 Hz",
        "es": "Bandas tipicas: telesismico 0.01-2 Hz · regional 0.5-5 Hz · local 1-15 Hz",
        "de": "Typische Bander: teleseismisch 0.01-2 Hz · regional 0.5-5 Hz · lokal 1-15 Hz",
    }
    st.markdown(f"<div class='sb-tip'>{_filter_tip[lang]}</div>", unsafe_allow_html=True)

    # ── STA/LTA ──────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('stalta_section', lang)}</div>", unsafe_allow_html=True)
    sta_len   = st.slider(T("sta_window", lang), 0.2, 5.0, 0.5, 0.1)
    lta_len   = st.slider(T("lta_window", lang), 5.0, 60.0, 20.0, 1.0)
    threshold = st.slider(T("threshold", lang), 1.0, 15.0, 3.5, 0.5)
    _stalta_tip = {
        "en": "Rule of thumb: LTA >= 10x STA. Lower threshold -> more triggers. Typical threshold: 3-5.",
        "it": "Regola: LTA >= 10x STA. Soglia bassa -> piu trigger. Soglia tipica: 3-5.",
        "fr": "Regle : LTA >= 10x STA. Seuil bas -> plus de declenchements. Seuil typique : 3-5.",
        "es": "Regla: LTA >= 10x STA. Umbral bajo -> mas disparos. Umbral tipico: 3-5.",
        "de": "Faustregel: LTA >= 10x STA. Niedriger Schwellenwert -> mehr Trigger. Typisch: 3-5.",
    }
    st.markdown(f"<div class='sb-tip'>{_stalta_tip[lang]}</div>", unsafe_allow_html=True)

    # ── Display ──────────────────────────────────────────────────────────────
    st.markdown(f"<div class='sb-section'>{T('display_section', lang)}</div>", unsafe_allow_html=True)
    show_raw         = st.toggle(T("show_raw", lang), value=False)
    show_phase       = st.toggle(T("show_phase", lang), value=False)
    show_psd_toggle  = st.toggle(T("show_psd", lang), value=True)
    show_spec_toggle = st.toggle(T("show_spec", lang), value=True)
    show_stalta_tog  = st.toggle(T("show_stalta", lang), value=True)

    st.markdown(f"<div class='sb-section'>{T('about_title', lang)}</div>", unsafe_allow_html=True)
    st.caption("SeismicLens v2.0 · ObsPy · SciPy · Streamlit")


# Inject light theme CSS if toggle is active
if light_theme:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)


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
        st.error(f"{T('error_mseed', lang)}: {e}")
elif data_source == 2 and uploaded_file:
    try:
        signal_raw, fs, metadata = load_csv_signal(uploaded_file)
    except Exception as e:
        st.error(f"{T('error_csv', lang)}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
  <h1>SeismicLens</h1>
  <div class='hero-sub'>{T("hero_sub", lang)}</div>
  <div class='hero-tag-row'>
    <span class='hero-tag ht-blue'>FFT &amp; {T("tab_fft", lang)}</span>
    <span class='hero-tag ht-green'>STA/LTA P-wave Picker</span>
    <span class='hero-tag ht-violet'>Butterworth Filter</span>
    <span class='hero-tag ht-amber'>MiniSEED · IRIS · INGV</span>
    <span class='hero-tag'>IASP91 Velocity Model</span>
    <span class='hero-tag'>CSV Export</span>
    <span class='hero-tag'>5 {T("language", lang)}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick-start guide ─────────────────────────────────────────────────────────
_GUIDE = {
    "en": {
        "title": "How to use SeismicLens",
        "steps": [
            ("1", "1", "<strong>Choose a data source</strong> in the sidebar: generate a <em>synthetic earthquake</em> (instant, no file needed), upload a real <em>MiniSEED</em> file from IRIS/INGV, or upload a <em>CSV</em> waveform."),
            ("2", "2", "<strong>Configure the Butterworth bandpass filter</strong>: set the low-cut and high-cut frequencies (Hz) and the filter order. Toggle it on/off to compare the filtered signal against the raw noisy waveform."),
            ("3", "3", "<strong>Adjust the STA/LTA P-wave detector</strong>: the STA window (short, 0.2-2 s) captures the onset energy; the LTA window (long, 5-60 s) tracks the background. Raise the threshold to suppress false triggers."),
            ("4", "4", "<strong>Explore the analysis tabs</strong>: <em>Waveform</em> (time domain), <em>Spectral Analysis</em> (FFT + PSD), <em>Spectrogram</em> (time-frequency), <em>STA/LTA</em> (phase detector), <em>Velocity Model</em> (IASP91), <em>Theory &amp; Math</em>."),
            ("5", "5", "<strong>Export your results</strong> as CSV files - processed signal, FFT spectrum (amplitude + phase), Welch PSD, and STA/LTA ratio - ready for Python, MATLAB, or Excel."),
        ],
    },
    "it": {
        "title": "Come usare SeismicLens",
        "steps": [
            ("1", "1", "<strong>Scegli la sorgente dati</strong> nella barra laterale: genera un <em>terremoto sintetico</em> (istantaneo, senza file), carica un file <em>MiniSEED</em> reale da IRIS/INGV, o carica una forma d'onda <em>CSV</em>."),
            ("2", "2", "<strong>Configura il filtro Butterworth bandpass</strong>: imposta le frequenze di taglio basso e alto (Hz) e l'ordine del filtro. Attivalo/disattivalo per confrontare il segnale filtrato con quello grezzo."),
            ("3", "3", "<strong>Regola il detector STA/LTA</strong>: la finestra STA (breve, 0.2-2 s) cattura l'energia dell'onset; la finestra LTA (lunga, 5-60 s) traccia il background. Alza la soglia per sopprimere i falsi trigger."),
            ("4", "4", "<strong>Esplora le schede di analisi</strong>: <em>Forma d'onda</em>, <em>Analisi spettrale</em> (FFT + PSD), <em>Spettrogramma</em>, <em>STA/LTA</em>, <em>Modello di velocita</em>, <em>Teoria e matematica</em>."),
            ("5", "5", "<strong>Esporta i risultati</strong> come file CSV - segnale elaborato, spettro FFT, PSD Welch, rapporto STA/LTA - pronti per Python, MATLAB o Excel."),
        ],
    },
    "fr": {
        "title": "Comment utiliser SeismicLens",
        "steps": [
            ("1", "1", "<strong>Choisissez une source de donnees</strong> dans la barre laterale : generez un <em>seisme synthetique</em> (instantane, sans fichier), importez un fichier <em>MiniSEED</em> reel depuis IRIS/INGV, ou importez une forme d'onde <em>CSV</em>."),
            ("2", "2", "<strong>Configurez le filtre Butterworth passe-bande</strong> : definissez les frequences de coupure basse et haute (Hz) et l'ordre du filtre. Activez/desactivez pour comparer signal filtre et brut."),
            ("3", "3", "<strong>Ajustez le detecteur STA/LTA</strong> : la fenetre STA (courte, 0.2-2 s) capture l'energie d'onset ; la fenetre LTA (longue, 5-60 s) suit le bruit de fond. Augmentez le seuil pour supprimer les faux declenchements."),
            ("4", "4", "<strong>Explorez les onglets d'analyse</strong> : <em>Forme d'onde</em>, <em>Analyse spectrale</em>, <em>Spectrogramme</em>, <em>STA/LTA</em>, <em>Modele de vitesse</em>, <em>Theorie &amp; Maths</em>."),
            ("5", "5", "<strong>Exportez vos resultats</strong> en CSV - signal, spectre FFT, PSD Welch, ratio STA/LTA - prets pour Python, MATLAB ou Excel."),
        ],
    },
    "es": {
        "title": "Como usar SeismicLens",
        "steps": [
            ("1", "1", "<strong>Elige una fuente de datos</strong> en la barra lateral: genera un <em>sismo sintetico</em> (instantaneo, sin archivo), carga un archivo <em>MiniSEED</em> real de IRIS/INGV, o carga una forma de onda <em>CSV</em>."),
            ("2", "2", "<strong>Configura el filtro Butterworth pasa-banda</strong>: establece las frecuencias de corte bajo y alto (Hz) y el orden. Activalo/desactivalo para comparar senal filtrada y bruta."),
            ("3", "3", "<strong>Ajusta el detector STA/LTA</strong>: la ventana STA (corta, 0.2-2 s) captura la energia de onset; la ventana LTA (larga, 5-60 s) rastrea el fondo. Sube el umbral para suprimir falsos disparos."),
            ("4", "4", "<strong>Explora las pestanas de analisis</strong>: <em>Forma de onda</em>, <em>Analisis espectral</em>, <em>Espectrograma</em>, <em>STA/LTA</em>, <em>Modelo de velocidad</em>, <em>Teoria y Matematicas</em>."),
            ("5", "5", "<strong>Exporta tus resultados</strong> como CSV - senal, espectro FFT, PSD Welch, razon STA/LTA - listos para Python, MATLAB o Excel."),
        ],
    },
    "de": {
        "title": "So verwendest du SeismicLens",
        "steps": [
            ("1", "1", "<strong>Wahle eine Datenquelle</strong> in der Seitenleiste: Erzeuge ein <em>synthetisches Erdbeben</em> (sofort, keine Datei notig), lade eine echte <em>MiniSEED</em>-Datei von IRIS/INGV hoch, oder lade eine <em>CSV</em>-Wellenform."),
            ("2", "2", "<strong>Konfiguriere den Butterworth-Bandpassfilter</strong>: Untere und obere Grenzfrequenz (Hz) und Filterordnung einstellen. Ein-/ausschalten um gefiltertes und Rohsignal zu vergleichen."),
            ("3", "3", "<strong>Passe den STA/LTA-Detektor an</strong>: Das STA-Fenster (kurz, 0.2-2 s) erfasst die Onset-Energie; das LTA-Fenster (lang, 5-60 s) verfolgt den Hintergrund. Schwelle erhohen um Fehlauslosungen zu unterdruecken."),
            ("4", "4", "<strong>Erkunde die Analyse-Tabs</strong>: <em>Wellenform</em>, <em>Spektralanalyse</em>, <em>Spektrogramm</em>, <em>STA/LTA</em>, <em>Geschwindigkeitsmodell</em>, <em>Theorie &amp; Mathematik</em>."),
            ("5", "5", "<strong>Exportiere deine Ergebnisse</strong> als CSV-Dateien - Signal, FFT-Spektrum, Welch-PSD, STA/LTA-Verhaltnis - bereit fur Python, MATLAB oder Excel."),
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
        st.warning(f"{T('error_filter', lang)}: {e}")
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
    _pill_filter_off = {
        "en": "Filter OFF — raw signal",
        "it": "Filtro DISATTIVO — segnale grezzo",
        "fr": "Filtre DESACTIVE — signal brut",
        "es": "Filtro DESACTIVADO — senal bruta",
        "de": "Filter AUS — Rohsignal",
    }
    _pill_p_arrival = {
        "en": f"P-arrival @ {p_time:.2f} s" if p_time else "",
        "it": f"Arrivo P @ {p_time:.2f} s" if p_time else "",
        "fr": f"Arrivee P @ {p_time:.2f} s" if p_time else "",
        "es": f"Llegada P @ {p_time:.2f} s" if p_time else "",
        "de": f"P-Ankunft @ {p_time:.2f} s" if p_time else "",
    }
    _pill_samples = {
        "en": f"N = {N} samples",
        "it": f"N = {N} campioni",
        "fr": f"N = {N} echantillons",
        "es": f"N = {N} muestras",
        "de": f"N = {N} Abtastwerte",
    }
    pills_html = "<div class='pill-row'>"
    pills_html += f"<span class='pill blue'>fs = {fs:.0f} Hz</span>"
    pills_html += f"<span class='pill green'>{_pill_samples[lang]}</span>"
    if filter_on:
        pills_html += f"<span class='pill violet'>Butterworth {f_low}–{f_high} Hz · order {filter_order}</span>"
    else:
        pills_html += f"<span class='pill amber'>{_pill_filter_off[lang]}</span>"
    if p_time:
        pills_html += f"<span class='pill coral'>{_pill_p_arrival[lang]}</span>"
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

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

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    # Why filter + filter details
    _fw_col1, _fw_col2 = st.columns(2)
    with _fw_col1:
        st.markdown(f"""
<div class='key-concept'>
<div class='kc-title'>{T('filter_why_title', lang)}</div>
<div class='kc-body'>{T('filter_why_body', lang)}</div>
</div>""", unsafe_allow_html=True)

    _filter_low_cut = {"en": "Low cut", "it": "Taglio basso", "fr": "Coupure basse", "es": "Corte bajo", "de": "Untere Grenzfrequenz"}
    _filter_high_cut = {"en": "High cut", "it": "Taglio alto", "fr": "Coupure haute", "es": "Corte alto", "de": "Obere Grenzfrequenz"}
    _filter_order_lbl = {"en": "Order", "it": "Ordine", "fr": "Ordre", "es": "Orden", "de": "Ordnung"}
    _filter_eff = {"en": "effective", "it": "effettivo", "fr": "effectif", "es": "efectivo", "de": "effektiv"}
    _filter_nyquist = {"en": "Nyquist", "it": "Nyquist", "fr": "Nyquist", "es": "Nyquist", "de": "Nyquist"}
    _filter_rolloff = {"en": "Roll-off", "it": "Roll-off", "fr": "Roll-off", "es": "Roll-off", "de": "Roll-off"}
    _filter_zero_phase_desc = {
        "en": "forward + backward pass — zero phase distortion, preserving wave arrival times exactly.",
        "it": "passaggio avanti + indietro — distorsione di fase zero, preserva esattamente i tempi di arrivo.",
        "fr": "passage avant + arriere — distorsion de phase zero, preserve exactement les temps d'arrivee.",
        "es": "paso adelante + atras — distorsion de fase cero, preserva exactamente los tiempos de llegada.",
        "de": "Vorwarts- + Ruckwartsdurchlauf — keine Phasenverzerrung, Ankunftszeiten bleiben exakt erhalten.",
    }

    with _fw_col2:
        if filter_on:
            eff_order = 2 * filter_order
            st.markdown(f"""
<div class='key-concept'>
<div class='kc-title'>{T('filter_details_title', lang)}</div>
<div class='kc-body'>
<b>{_filter_low_cut[lang]}:</b> {f_low} Hz &nbsp;|&nbsp; <b>{_filter_high_cut[lang]}:</b> {f_high} Hz &nbsp;|&nbsp; <b>{_filter_order_lbl[lang]}:</b> {filter_order} — {_filter_eff[lang]} <b>{eff_order}</b> (zero-phase)<br>
<b>{_filter_nyquist[lang]}:</b> {fs/2:.1f} Hz &nbsp;|&nbsp; {_filter_rolloff[lang]}: <span class='f-inline'>{40*filter_order} dB/decade</span><br><br>
<code>sosfiltfilt</code> = {_filter_zero_phase_desc[lang]}
</div>
</div>""", unsafe_allow_html=True)

    if filter_on:
        _filt_exp_title = {
            "en": "Filter details — full mathematics",
            "it": "Dettagli filtro — matematica completa",
            "fr": "Details du filtre — mathematiques completes",
            "es": "Detalles del filtro — matematicas completas",
            "de": "Filterdetails — vollstandige Mathematik",
        }
        _filt_params_lbl = {
            "en": "Parameters",
            "it": "Parametri",
            "fr": "Parametres",
            "es": "Parametros",
            "de": "Parameter",
        }
        _filt_why_sos = {
            "en": "**Why Second-Order Sections (SOS)?** A high-order IIR filter (e.g., order 8) has a long chain of multiply-adds. In direct form the coefficients have very different magnitudes, causing numerical overflow/underflow. SOS splits the filter into cascaded biquad (2nd-order) sections, each stable on its own.",
            "it": "**Perche le Sezioni del Secondo Ordine (SOS)?** Un filtro IIR di ordine elevato (es. ordine 8) ha una lunga catena di moltiplicazioni-addizioni. Nella forma diretta i coefficienti hanno grandezze molto diverse, causando overflow/underflow numerico. Le SOS suddividono il filtro in sezioni biquad (2° ordine) a cascata, ognuna stabile di per se.",
            "fr": "**Pourquoi les Sections du Second Ordre (SOS) ?** Un filtre IIR d'ordre eleve (ex. ordre 8) comporte une longue chaine de multiplications-additions. Sous forme directe, les coefficients ont des magnitudes tres differentes, causant des depassements numeriques. Les SOS divisent le filtre en sections biquad (2e ordre) en cascade, chacune stable.",
            "es": "**Por que Secciones de Segundo Orden (SOS)?** Un filtro IIR de orden elevado (ej. orden 8) tiene una larga cadena de multiplicaciones-sumas. En forma directa los coeficientes tienen magnitudes muy diferentes, causando desbordamiento numerico. Las SOS dividen el filtro en secciones biquad (2o orden) en cascada, cada una estable.",
            "de": "**Warum Abschnitte zweiter Ordnung (SOS)?** Ein IIR-Filter hoher Ordnung (z.B. Ordnung 8) hat eine lange Kette von Multiplikations-Additionen. In direkter Form haben die Koeffizienten sehr unterschiedliche Grossenordnungen, was zu numerischem Uber-/Unterlauf fuhrt. SOS zerlegt den Filter in kaskadierende Biquad-Abschnitte (2. Ordnung), die jeweils fur sich stabil sind.",
        }
        _filt_why_zero = {
            "en": "**Why zero-phase?** A causal filter delays different frequencies by different amounts (group delay is not constant), distorting seismic waveforms and shifting arrival times. `sosfiltfilt` runs the filter forward then backward: the phase responses cancel exactly, giving zero phase distortion with double the effective roll-off.",
            "it": "**Perche zero-phase?** Un filtro causale ritarda le diverse frequenze di quantita diverse (ritardo di gruppo non costante), distorcendo le forme d'onda sismiche e spostando i tempi di arrivo. `sosfiltfilt` applica il filtro in avanti e poi all'indietro: le risposte in fase si annullano esattamente, dando distorsione di fase zero con roll-off effettivo doppio.",
            "fr": "**Pourquoi zero-phase ?** Un filtre causal retarde les differentes frequences de quantites differentes (retard de groupe non constant), distordant les formes d'onde sismiques. `sosfiltfilt` applique le filtre en avant puis en arriere : les reponses en phase s'annulent exactement, donnant une distorsion de phase nulle avec un roll-off double.",
            "es": "**Por que fase cero?** Un filtro causal retrasa diferentes frecuencias en cantidades distintas (retardo de grupo no constante), distorsionando las formas de onda sismicas. `sosfiltfilt` aplica el filtro hacia adelante y luego hacia atras: las respuestas de fase se cancelan exactamente, dando distorsion de fase cero con roll-off doble.",
            "de": "**Warum Nullphasen-Filterung?** Ein kausales Filter verzogert verschiedene Frequenzen unterschiedlich stark (nicht konstante Gruppenlaufzeit), was seismische Wellenformen verzerrt und Ankunftszeiten verschiebt. `sosfiltfilt` wendet den Filter vorwarts dann ruckwarts an: Die Phasenantworten heben sich exakt auf, was zu keiner Phasenverzerrung bei doppeltem effektivem Roll-off fuhrt.",
        }
        with st.expander(f"{T('filter_details_title', lang)} — {_filt_params_lbl[lang]}"):
            st.markdown(f"""
**{_filt_params_lbl[lang]}:**
- {_filter_low_cut[lang]}: **{f_low} Hz** | {_filter_high_cut[lang]}: **{f_high} Hz**
- {_filter_order_lbl[lang]}: **{filter_order}** ({_filter_eff[lang]} {eff_order} con zero-phase `sosfiltfilt`)
- {_filter_nyquist[lang]}: **{fs/2:.1f} Hz**
- Implementation: `scipy.signal.sosfiltfilt` — Second-Order Sections

{_filt_why_sos[lang]}

{_filt_why_zero[lang]}
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
            ("[A]", "Dominant Frequency", f"The frequency bin with the highest amplitude: <strong>{dominant_f:.3f} Hz</strong>. This is the most energetic frequency in the signal."),
            ("[C]", "Spectral Centroid", f"The energy-weighted mean frequency: <strong>{centroid_f:.3f} Hz</strong>. It represents the 'centre of mass' of the spectrum — higher than dominant_f if energy spreads to high frequencies."),
            ("[B]", "Bandwidth (RMS)", f"The RMS spread of energy around the centroid: <strong>{bandwidth:.3f} Hz</strong>. Wide bandwidth = broad-band signal (e.g., impulsive source); narrow = quasi-monochromatic."),
            ("[f]", "Frequency Resolution", f"Each FFT bin covers <strong>df = fs/N = {fs/N:.4f} Hz</strong>. To improve resolution, increase the signal duration (more samples N)."),
        ],
        "it": [
            ("[A]", "Frequenza dominante", f"Il bin con ampiezza massima: <strong>{dominant_f:.3f} Hz</strong>. E' la frequenza piu energetica del segnale."),
            ("[C]", "Centroide spettrale", f"La media pesata per l'energia: <strong>{centroid_f:.3f} Hz</strong>. Rappresenta il 'centro di massa' dello spettro."),
            ("[B]", "Larghezza di banda (RMS)", f"La dispersione RMS dell'energia attorno al centroide: <strong>{bandwidth:.3f} Hz</strong>. Banda larga = sorgente impulsiva; stretta = quasi-monocromatica."),
            ("[f]", "Risoluzione in frequenza", f"Ogni bin FFT copre <strong>df = fs/N = {fs/N:.4f} Hz</strong>. Per migliorare la risoluzione, aumentare la durata del segnale."),
        ],
        "fr": [
            ("[A]", "Frequence dominante", f"Le bin avec l'amplitude maximale : <strong>{dominant_f:.3f} Hz</strong>. C'est la frequence la plus energetique du signal."),
            ("[C]", "Centroide spectral", f"La moyenne ponderee par l'energie : <strong>{centroid_f:.3f} Hz</strong>. Represente le 'centre de masse' du spectre."),
            ("[B]", "Largeur de bande (RMS)", f"La dispersion RMS autour du centroide : <strong>{bandwidth:.3f} Hz</strong>. Large = source impulsive ; etroite = quasi-monochromatique."),
            ("[f]", "Resolution frequentielle", f"Chaque bin FFT couvre <strong>df = fs/N = {fs/N:.4f} Hz</strong>. Pour ameliorer la resolution, augmenter la duree du signal."),
        ],
        "es": [
            ("[A]", "Frecuencia dominante", f"El bin con amplitud maxima: <strong>{dominant_f:.3f} Hz</strong>. Es la frecuencia mas energetica de la senal."),
            ("[C]", "Centroide espectral", f"La media ponderada por energia: <strong>{centroid_f:.3f} Hz</strong>. Representa el 'centro de masa' del espectro."),
            ("[B]", "Ancho de banda (RMS)", f"La dispersion RMS alrededor del centroide: <strong>{bandwidth:.3f} Hz</strong>. Ancho = fuente impulsiva; estrecho = casi-monocromatico."),
            ("[f]", "Resolucion frecuencial", f"Cada bin FFT cubre <strong>df = fs/N = {fs/N:.4f} Hz</strong>. Para mejorar la resolucion, aumentar la duracion de la senal."),
        ],
        "de": [
            ("[A]", "Dominante Frequenz", f"Der Bin mit maximaler Amplitude: <strong>{dominant_f:.3f} Hz</strong>. Das ist die energiereichste Frequenz des Signals."),
            ("[C]", "Spektralzentroid", f"Der energiegewichtete Mittelwert: <strong>{centroid_f:.3f} Hz</strong>. Reprasentiert den 'Schwerpunkt' des Spektrums."),
            ("[B]", "Bandbreite (RMS)", f"Die RMS-Streuung um den Zentroid: <strong>{bandwidth:.3f} Hz</strong>. Breit = impulsive Quelle; schmal = quasi-monochromatisch."),
            ("[f]", "Frequenzauflosung", f"Jeder FFT-Bin umfasst <strong>df = fs/N = {fs/N:.4f} Hz</strong>. Fur bessere Auflosung Signaldauer erhohen."),
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
        _phase_title = {
            "en": "Phase Spectrum X(f)",
            "it": "Spettro di fase X(f)",
            "fr": "Spectre de phase X(f)",
            "es": "Espectro de fase X(f)",
            "de": "Phasenspektrum X(f)",
        }
        _phase_exp_title = {
            "en": "Phase — detailed explanation",
            "it": "Fase — spiegazione dettagliata",
            "fr": "Phase — explication detaillee",
            "es": "Fase — explicacion detallada",
            "de": "Phase — detaillierte Erklarung",
        }
        _phase_exp_body = {
            "en": """
Each FFT coefficient **X(f)** is a **complex number** in the 2D complex plane:

- **|X(f)|** = amplitude (distance from origin)
- **phi(f) = angle X(f)** = phase angle = `atan2(Im, Re)` in [-pi, pi]

**What does the phase tell us?**
- **Random noise**: phases are uniformly distributed — no coherent structure.
- **Impulsive signal** (P-wave onset): phases cluster together near the arrival time.
- **Dispersive wave train** (surface waves): phase varies smoothly with frequency (group delay not constant).
- **Zero-phase filter**: after `sosfiltfilt`, the phase spectrum of the filtered signal equals the unfiltered one.
""",
            "it": """
Ogni coefficiente FFT **X(f)** e' un **numero complesso** nel piano complesso 2D:

- **|X(f)|** = ampiezza (distanza dall'origine)
- **phi(f) = angolo X(f)** = angolo di fase = `atan2(Im, Re)` in [-pi, pi]

**Cosa ci dice la fase?**
- **Rumore casuale**: le fasi sono distribuite uniformemente — nessuna struttura coerente.
- **Segnale impulsivo** (onset onda P): le fasi si raggruppano vicino al tempo di arrivo.
- **Treno d'onde dispersivo** (onde superficiali): la fase varia in modo continuo con la frequenza (ritardo di gruppo non costante).
- **Filtro zero-phase**: dopo `sosfiltfilt`, lo spettro di fase del segnale filtrato e' uguale a quello non filtrato.
""",
            "fr": """
Chaque coefficient FFT **X(f)** est un **nombre complexe** dans le plan complexe 2D:

- **|X(f)|** = amplitude (distance depuis l'origine)
- **phi(f) = angle X(f)** = angle de phase = `atan2(Im, Re)` dans [-pi, pi]

**Que nous dit la phase?**
- **Bruit aleatoire**: les phases sont uniformement distribuees — aucune structure coherente.
- **Signal impulsif** (onset onde P): les phases se regroupent pres du temps d'arrivee.
- **Train d'ondes dispersif** (ondes de surface): la phase varie en douceur avec la frequence (retard de groupe non constant).
- **Filtre zero-phase**: apres `sosfiltfilt`, le spectre de phase du signal filtre est egal au signal non filtre.
""",
            "es": """
Cada coeficiente FFT **X(f)** es un **numero complejo** en el plano complejo 2D:

- **|X(f)|** = amplitud (distancia desde el origen)
- **phi(f) = angulo X(f)** = angulo de fase = `atan2(Im, Re)` en [-pi, pi]

**Que nos dice la fase?**
- **Ruido aleatorio**: las fases estan distribuidas uniformemente — sin estructura coherente.
- **Senal impulsiva** (onset onda P): las fases se agrupan cerca del tiempo de llegada.
- **Tren de ondas dispersivo** (ondas superficiales): la fase varia suavemente con la frecuencia.
- **Filtro fase cero**: tras `sosfiltfilt`, el espectro de fase del senal filtrada es igual al no filtrado.
""",
            "de": """
Jeder FFT-Koeffizient **X(f)** ist eine **komplexe Zahl** in der 2D-komplexen Ebene:

- **|X(f)|** = Amplitude (Abstand vom Ursprung)
- **phi(f) = Winkel X(f)** = Phasenwinkel = `atan2(Im, Re)` in [-pi, pi]

**Was sagt uns die Phase?**
- **Zufallsrauschen**: Phasen sind gleichmaessig verteilt — keine koharente Struktur.
- **Impulsives Signal** (P-Wellen-Onset): Phasen clustern sich nahe der Ankunftszeit.
- **Dispersiver Wellenzug** (Oberflachenwellen): Phase andert sich gleichmaessig mit der Frequenz.
- **Nullphasen-Filter**: Nach `sosfiltfilt` ist das Phasenspektrum des gefilterten Signals gleich dem ungefilterten.
""",
        }
        st.markdown(f"<div class='section-title'>{_phase_title[lang]}</div>", unsafe_allow_html=True)
        st.plotly_chart(make_phase_fig(freqs, phases_deg), use_container_width=True)
        st.markdown(f"<div class='info-box'>{T('phase_info', lang)}</div>", unsafe_allow_html=True)

        with st.expander(_phase_exp_title[lang]):
            st.latex(r"X(f) = \text{Re}[X(f)] + j \cdot \text{Im}[X(f)] = |X(f)| \cdot e^{j\varphi(f)}")
            st.markdown(_phase_exp_body[lang])

    st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

    if show_psd_toggle:
        _psd_title = {
            "en": "Power Spectral Density (PSD) — Welch method",
            "it": "Densita spettrale di potenza (PSD) — metodo di Welch",
            "fr": "Densite spectrale de puissance (PSD) — methode de Welch",
            "es": "Densidad espectral de potencia (PSD) — metodo de Welch",
            "de": "Leistungsspektraldichte (PSD) — Welch-Methode",
        }
        _psd_exp_title = {
            "en": "PSD — detailed explanation",
            "it": "PSD — spiegazione dettagliata",
            "fr": "PSD — explication detaillee",
            "es": "PSD — explicacion detallada",
            "de": "PSD — detaillierte Erklarung",
        }
        _psd_exp_body = {
            "en": """
**Why not just use the FFT amplitude squared?**
A single FFT of a long signal gives a **very noisy** power estimate — each frequency bin has
high variance because it is computed from just one complex number.

**Welch's method** (1967) reduces variance by averaging:

- Split signal into **K overlapping segments** (50% overlap default)
- Apply **Hann window** to each segment (reduces spectral leakage)
- Compute FFT squared of each segment
- **Average** — variance reduced by ~1/K compared to single FFT

**Units:** counts squared per Hz (raw) or dB re 1 count squared per Hz (log scale, as shown).
""",
            "it": """
**Perche non usare semplicemente l'ampiezza FFT al quadrato?**
Una singola FFT di un segnale lungo produce una stima di potenza **molto rumorosa** — ogni bin di frequenza
ha alta varianza perche e' calcolato da un solo numero complesso.

**Il metodo di Welch** (1967) riduce la varianza per media:

- Dividi il segnale in **K segmenti sovrapposti** (50% overlap di default)
- Applica la **finestra di Hann** a ciascun segmento (riduce la perdita spettrale)
- Calcola la FFT al quadrato di ciascun segmento
- **Media** — varianza ridotta di ~1/K rispetto a una singola FFT

**Unita':** counts^2/Hz (grezzi) o dB re 1 count^2/Hz (scala logaritmica, come mostrato).
""",
            "fr": """
**Pourquoi ne pas simplement utiliser l'amplitude FFT au carre?**
Une seule FFT d'un long signal donne une estimation de puissance **tres bruitee** — chaque bin de frequence
a une variance elevee car il est calcule a partir d'un seul nombre complexe.

**La methode de Welch** (1967) reduit la variance par moyennage:

- Diviser le signal en **K segments chevauchants** (50% de recouvrement par defaut)
- Appliquer la **fenetre de Hann** a chaque segment (reduit la fuite spectrale)
- Calculer la FFT au carre de chaque segment
- **Moyenner** — variance reduite d'environ 1/K par rapport a une seule FFT

**Unites:** counts^2/Hz (bruts) ou dB re 1 count^2/Hz (echelle logarithmique).
""",
            "es": """
**Por que no usar simplemente la amplitud FFT al cuadrado?**
Una sola FFT de una senal larga da una estimacion de potencia **muy ruidosa** — cada bin de frecuencia
tiene alta varianza porque se calcula a partir de un solo numero complejo.

**El metodo de Welch** (1967) reduce la varianza por promediado:

- Dividir la senal en **K segmentos solapados** (50% de solapamiento por defecto)
- Aplicar la **ventana de Hann** a cada segmento (reduce la fuga espectral)
- Calcular la FFT al cuadrado de cada segmento
- **Promediar** — varianza reducida en ~1/K respecto a una sola FFT

**Unidades:** counts^2/Hz (brutos) o dB re 1 count^2/Hz (escala logaritmica).
""",
            "de": """
**Warum nicht einfach die quadrierte FFT-Amplitude verwenden?**
Eine einzige FFT eines langen Signals liefert eine **sehr verrauschte** Leistungsschatzung — jeder
Frequenz-Bin hat hohe Varianz, da er aus nur einer komplexen Zahl berechnet wird.

**Welchs Methode** (1967) reduziert die Varianz durch Mittelung:

- Signal in **K uberlappende Segmente** aufteilen (standardmaessig 50% Uberlappung)
- **Hann-Fenster** auf jedes Segment anwenden (reduziert Spektralausblutung)
- FFT-Quadrat jedes Segments berechnen
- **Mitteln** — Varianz um ~1/K gegenuber einer einzelnen FFT reduziert

**Einheiten:** counts^2/Hz (roh) oder dB re 1 count^2/Hz (logarithmische Skala).
""",
        }
        st.markdown(f"<div class='section-title'>{_psd_title[lang]}</div>", unsafe_allow_html=True)
        st.plotly_chart(make_psd_fig(f_psd, psd), use_container_width=True)
        st.markdown(f"<div class='info-box'>{T('psd_info', lang)}</div>", unsafe_allow_html=True)

        with st.expander(_psd_exp_title[lang]):
            st.markdown(_psd_exp_body[lang])
            st.latex(r"S(f) = \frac{1}{K} \sum_{k=1}^{K} \left| \text{FFT}\{x_k[n] \cdot w[n]\} \right|^2")
            st.latex(r"\text{PSD}_{dB}(f) = 10 \cdot \log_{10}\left(S(f)\right)")


# ── Spectrogram ───────────────────────────────────────────────────────────────
with tab_spec:
    if show_spec_toggle:
        st.markdown(f"<div class='section-title'>{T('tab_spec', lang)} — STFT</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tab-intro'>{T('spec_desc', lang)}</div>", unsafe_allow_html=True)
        try:
            t_spec, f_spec, Sxx = compute_spectrogram(signal_proc, fs)
            st.plotly_chart(make_spectrogram_fig(t_spec, f_spec, Sxx), use_container_width=True)

            st.markdown(f"<div class='info-box'>[i] {T('spec_info', lang)}</div>", unsafe_allow_html=True)

            _spec_read = {
                "en": "<strong>How to read this plot:</strong> The x-axis is time (s), y-axis is frequency (Hz), colour is energy (dB). Look for: <em>vertical bright streaks</em> = impulsive onset (P/S arrival); <em>horizontal bands</em> = monochromatic interference; <em>diagonal stripes</em> = dispersive surface waves.",
                "it": "<strong>Come leggere questo grafico:</strong> L'asse x e' il tempo (s), l'asse y la frequenza (Hz), il colore e' l'energia (dB). Cerca: <em>strisce verticali luminose</em> = onset impulsivo (arrivo P/S); <em>bande orizzontali</em> = interferenza monocromatica; <em>strisce diagonali</em> = onde superficiali dispersive.",
                "fr": "<strong>Comment lire ce graphique :</strong> L'axe x est le temps (s), l'axe y la frequence (Hz), la couleur est l'energie (dB). Cherchez : <em>bandes verticales lumineuses</em> = onset impulsif (arrivee P/S) ; <em>bandes horizontales</em> = interference monochromatique ; <em>bandes diagonales</em> = ondes de surface dispersives.",
                "es": "<strong>Como leer este grafico:</strong> El eje x es tiempo (s), el eje y frecuencia (Hz), el color es energia (dB). Busca: <em>franjas verticales brillantes</em> = onset impulsivo (llegada P/S); <em>bandas horizontales</em> = interferencia monocromatica; <em>franjas diagonales</em> = ondas superficiales dispersivas.",
                "de": "<strong>So liest man dieses Diagramm:</strong> X-Achse = Zeit (s), Y-Achse = Frequenz (Hz), Farbe = Energie (dB). Suche nach: <em>vertikalen hellen Streifen</em> = impulsiver Onset (P/S-Ankunft); <em>horizontalen Bandern</em> = monochromatische Interferenz; <em>diagonalen Streifen</em> = dispersive Oberflachenwellen.",
            }
            st.markdown(f"<div class='warn-box'>{_spec_read[lang]}</div>", unsafe_allow_html=True)

            _stft_exp_title = {
                "en": "STFT and Heisenberg Uncertainty — detailed explanation",
                "it": "STFT e principio di indeterminazione di Heisenberg — spiegazione dettagliata",
                "fr": "STFT et incertitude d'Heisenberg — explication detaillee",
                "es": "STFT e incertidumbre de Heisenberg — explicacion detallada",
                "de": "STFT und Heisenbergsche Unbestimmtheitsrelation — detaillierte Erklarung",
            }
            _stft_body = {
                "en": """
In discrete form with a Hann window of length M:
- The signal is **windowed** around time tau, keeping only M samples.
- The **FFT** of those M samples gives the local spectrum at time tau.
- The window slides forward by **hop_size** samples, producing a 2D matrix.

**Time-Frequency Trade-off (Heisenberg-Gabor Uncertainty):**

| Window choice | Time resolution | Frequency resolution |
|---|---|---|
| **Short window** (64 samples) | Fine (ms) | Coarse (several Hz) |
| **Long window** (1024 samples) | Coarse (100s ms) | Fine (fractions of Hz) |

There is **no way to get both simultaneously** — this is a fundamental physical limit, not a computational one.

**Welch PSD vs Spectrogram:** The Welch PSD is the **time-averaged** spectrogram — it collapses the time axis and shows only the mean power per frequency.
""",
                "it": """
Nella forma discreta con finestra di Hann di lunghezza M:
- Il segnale viene **moltiplicato per la finestra** attorno al tempo tau, mantenendo solo M campioni.
- La **FFT** di questi M campioni da' lo spettro locale al tempo tau.
- La finestra scorre in avanti di **hop_size** campioni, producendo una matrice 2D.

**Compromesso tempo-frequenza (Principio di indeterminazione di Heisenberg-Gabor):**

| Scelta della finestra | Risoluzione temporale | Risoluzione frequenziale |
|---|---|---|
| **Finestra corta** (64 campioni) | Fine (ms) | Grossolana (diversi Hz) |
| **Finestra lunga** (1024 campioni) | Grossolana (centinaia di ms) | Fine (frazioni di Hz) |

**Non e' possibile ottenere entrambe simultaneamente** — questo e' un limite fisico fondamentale, non computazionale.

**PSD Welch vs Spettrogramma:** La PSD Welch e' lo spettrogramma **mediato nel tempo** — collassa l'asse temporale e mostra solo la potenza media per frequenza.
""",
                "fr": """
Sous forme discrete avec une fenetre de Hann de longueur M:
- Le signal est **fenetree** autour du temps tau, conservant seulement M echantillons.
- La **FFT** de ces M echantillons donne le spectre local au temps tau.
- La fenetre avance de **hop_size** echantillons, produisant une matrice 2D.

**Compromis temps-frequence (Incertitude d'Heisenberg-Gabor):**

| Choix de la fenetre | Resolution temporelle | Resolution frequentielle |
|---|---|---|
| **Fenetre courte** (64 ech.) | Fine (ms) | Grossiere (plusieurs Hz) |
| **Fenetre longue** (1024 ech.) | Grossiere (centaines de ms) | Fine (fractions de Hz) |

**Il est impossible d'avoir les deux simultanement** — c'est une limite physique fondamentale, pas computationnelle.

**PSD Welch vs Spectrogramme:** La PSD Welch est le spectrogramme **moyenne en temps** — elle efface l'axe temporel et montre uniquement la puissance moyenne par frequence.
""",
                "es": """
En forma discreta con una ventana de Hann de longitud M:
- La senal se **enventana** alrededor del tiempo tau, conservando solo M muestras.
- La **FFT** de estas M muestras da el espectro local en el tiempo tau.
- La ventana avanza **hop_size** muestras, produciendo una matriz 2D.

**Compromiso tiempo-frecuencia (Incertidumbre de Heisenberg-Gabor):**

| Eleccion de ventana | Resolucion temporal | Resolucion frecuencial |
|---|---|---|
| **Ventana corta** (64 muestras) | Fina (ms) | Gruesa (varios Hz) |
| **Ventana larga** (1024 muestras) | Gruesa (cientos de ms) | Fina (fracciones de Hz) |

**No es posible obtener ambas simultaneamente** — este es un limite fisico fundamental, no computacional.

**PSD Welch vs Espectrograma:** La PSD Welch es el espectrograma **promediado en tiempo** — colapsa el eje temporal y muestra solo la potencia media por frecuencia.
""",
                "de": """
In diskreter Form mit einem Hann-Fenster der Lange M:
- Das Signal wird um die Zeit tau **gefenstert**, wobei nur M Samples beibehalten werden.
- Die **FFT** dieser M Samples ergibt das lokale Spektrum zum Zeitpunkt tau.
- Das Fenster verschiebt sich um **hop_size** Samples nach vorne und erzeugt eine 2D-Matrix.

**Zeit-Frequenz-Kompromiss (Heisenberg-Gabor-Unbestimmtheitsrelation):**

| Fensterwahl | Zeitauflosung | Frequenzauflosung |
|---|---|---|
| **Kurzes Fenster** (64 Samples) | Fein (ms) | Grob (mehrere Hz) |
| **Langes Fenster** (1024 Samples) | Grob (100er ms) | Fein (Bruchteile von Hz) |

**Es ist nicht moglich, beides gleichzeitig zu erreichen** — dies ist eine grundlegende physikalische Grenze, keine computationale.

**Welch-PSD vs Spektrogramm:** Die Welch-PSD ist das **zeitlich gemittelte** Spektrogramm — sie bricht die Zeitachse zusammen und zeigt nur die mittlere Leistung pro Frequenz.
""",
            }
            with st.expander(_stft_exp_title[lang]):
                st.latex(r"\text{STFT}(\tau, f) = \int_{-\infty}^{\infty} x(t) \cdot w(t - \tau) \cdot e^{-j 2\pi f t} \, dt")
                st.markdown(_stft_body[lang])
                st.latex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}")
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
                    f"<div class='success-box'>{T('stalta_trigger_ok', lang, t=p_time, n=n_trig)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='warn-box'>{T('stalta_no_trigger', lang)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='grad-div'></div>", unsafe_allow_html=True)

            # Parameter cards
            sta_n = int(sta_len * fs)
            lta_n = int(lta_len * fs)
            _sl_ratio = lta_len / sta_len
            _sl_color = "green" if _sl_ratio >= 10 else "amber"
            _sl_lta_sta = {"en": "LTA / STA", "it": "LTA / STA", "fr": "LTA / STA", "es": "LTA / STA", "de": "LTA / STA"}
            _sl_threshold_lbl = {"en": "Threshold", "it": "Soglia", "fr": "Seuil", "es": "Umbral", "de": "Schwelle"}
            _sl_window_lbl = {"en": "Window", "it": "Finestra", "fr": "Fenetre", "es": "Ventana", "de": "Fenster"}
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.markdown(metric_card(f"STA {_sl_window_lbl[lang]}", f"{sta_len}", "s", "blue"),   unsafe_allow_html=True)
            _pc2.markdown(metric_card(f"LTA {_sl_window_lbl[lang]}", f"{lta_len}", "s", "violet"), unsafe_allow_html=True)
            _pc3.markdown(metric_card(_sl_lta_sta[lang], f"{_sl_ratio:.1f}", "x", _sl_color),      unsafe_allow_html=True)
            _pc4.markdown(metric_card(_sl_threshold_lbl[lang], f"{threshold}", "",  "coral"),       unsafe_allow_html=True)

            _stalta_det_title = {
                "en": "STA/LTA — algorithm details and tuning guide",
                "it": "STA/LTA — dettagli dell'algoritmo e guida alla regolazione",
                "fr": "STA/LTA — details de l'algorithme et guide de reglage",
                "es": "STA/LTA — detalles del algoritmo y guia de ajuste",
                "de": "STA/LTA — Algorithmusdetails und Einstellungsleitfaden",
            }
            _stalta_good = {
                "en": "good (LTA/STA ratio is adequate)",
                "it": "buono (rapporto LTA/STA adeguato)",
                "fr": "bon (rapport LTA/STA adequat)",
                "es": "bueno (relacion LTA/STA adecuada)",
                "de": "gut (LTA/STA-Verhaltnis angemessen)",
            }
            _stalta_warn_ratio = {
                "en": "LTA should be at least 10x STA for stable detection",
                "it": "LTA dovrebbe essere almeno 10x STA per una rilevazione stabile",
                "fr": "LTA doit etre au moins 10x STA pour une detection stable",
                "es": "LTA debe ser al menos 10x STA para una deteccion estable",
                "de": "LTA sollte mindestens 10x STA sein fur stabile Erkennung",
            }
            _stalta_tuning_header = {
                "en": "Tuning guide",
                "it": "Guida alla regolazione",
                "fr": "Guide de reglage",
                "es": "Guia de ajuste",
                "de": "Einstellungsleitfaden",
            }
            _stalta_scenario = {
                "en": "Scenario",
                "it": "Scenario",
                "fr": "Scenario",
                "es": "Escenario",
                "de": "Szenario",
            }
            _stalta_why_sq = {
                "en": "**Why square the samples?** Squaring makes the metric invariant to the sign (polarity) of the signal — a negative spike is just as impulsive as a positive one. It also tracks **energy** rather than amplitude.",
                "it": "**Perche elevare i campioni al quadrato?** Il quadrato rende la metrica invariante al segno (polarita') del segnale — un picco negativo e' impulsivo quanto uno positivo. Misura inoltre l'**energia** anziche' l'ampiezza.",
                "fr": "**Pourquoi elever les echantillons au carre?** Le carre rend la metrique invariante au signe (polarite) du signal — un pic negatif est aussi impulsif qu'un pic positif. Cela mesure egalement l'**energie** plutot que l'amplitude.",
                "es": "**Por que elevar al cuadrado las muestras?** El cuadrado hace la metrica invariante al signo (polaridad) de la senal — un pico negativo es tan impulsivo como uno positivo. Tambien rastrea la **energia** en lugar de la amplitud.",
                "de": "**Warum werden die Samples quadriert?** Durch das Quadrieren wird die Metrik unempfindlich gegenuber dem Vorzeichen (Polaritat) des Signals — ein negativer Spike ist genauso impulsiv wie ein positiver. Es wird auch **Energie** statt Amplitude verfolgt.",
            }
            _stalta_prefix_sum = {
                "en": "**O(N) implementation via prefix sums:**",
                "it": "**Implementazione O(N) tramite somme prefisse:**",
                "fr": "**Implementation O(N) par sommes de prefixes:**",
                "es": "**Implementacion O(N) mediante sumas de prefijos:**",
                "de": "**O(N)-Implementierung uber Prafix-Summen:**",
            }
            _stalta_params_lbl = {
                "en": "Current settings",
                "it": "Impostazioni correnti",
                "fr": "Parametres actuels",
                "es": "Configuracion actual",
                "de": "Aktuelle Einstellungen",
            }
            _samples_word = {"en": "samples", "it": "campioni", "fr": "echantillons", "es": "muestras", "de": "Abtastwerte"}
            _loc_eq = {"en": "Local earthquake", "it": "Terremoto locale", "fr": "Seisme local", "es": "Sismo local", "de": "Lokales Erdbeben"}
            _reg_eq = {"en": "Regional earthquake", "it": "Terremoto regionale", "fr": "Seisme regional", "es": "Sismo regional", "de": "Regionales Erdbeben"}
            _tele_eq = {"en": "Teleseismic", "it": "Telesismico", "fr": "Telesismique", "es": "Telesismico", "de": "Teleseismisch"}
            _noisy = {"en": "Very noisy data", "it": "Dati molto rumorosi", "fr": "Donnees tres bruyantes", "es": "Datos muy ruidosos", "de": "Sehr verrauschte Daten"}
            with st.expander(_stalta_det_title[lang]):
                st.latex(r"\text{STA}(t) = \frac{1}{N_{sta}} \sum_{k=0}^{N_{sta}-1} x^2[t-k]")
                st.latex(r"\text{LTA}(t) = \frac{1}{N_{lta}} \sum_{k=0}^{N_{lta}-1} x^2[t-k]")
                st.latex(r"R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}")
                _ratio_status = _stalta_good[lang] if _sl_ratio >= 10 else _stalta_warn_ratio[lang]
                _sw = _samples_word[lang]
                st.markdown(f"""
**{_stalta_params_lbl[lang]}:**
- STA = {sta_len} s — **{sta_n} {_sw}**
- LTA = {lta_len} s — **{lta_n} {_sw}**
- LTA/STA = **{_sl_ratio:.1f}x** — {_ratio_status}
- {_sl_threshold_lbl[lang]} = **{threshold}**

{_stalta_why_sq[lang]}

{_stalta_prefix_sum[lang]}
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```

**{_stalta_tuning_header[lang]}:**

| {_stalta_scenario[lang]} | STA | LTA | {_sl_threshold_lbl[lang]} |
|---|---|---|---|
| {_loc_eq[lang]} | 0.2-0.5 s | 10-20 s | 3-4 |
| {_reg_eq[lang]} | 0.5-1 s | 20-40 s | 3-5 |
| {_tele_eq[lang]} | 1-2 s | 30-60 s | 2-3 |
| {_noisy[lang]} | 0.5 s | 30 s | 5-8 |

Allen, R.V. (1978). *Automatic earthquake recognition and timing from single traces*. BSSA, 68(5).
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
            ("[P]", "P-wave velocity (Vp)", "Compressional waves — particles vibrate parallel to propagation direction. Controlled by bulk modulus K (resistance to compression) + shear modulus G + density rho."),
            ("[S]", "S-wave velocity (Vs)", "Shear waves — particles vibrate perpendicular to propagation. Controlled only by shear modulus G. <strong>Vs = 0 in fluids</strong> because fluids cannot sustain shear stress (G = 0)."),
            ("[r]", "Vp/Vs ratio", "Key diagnostic indicator. Poisson solid: Vp/Vs = sqrt(3) ~ 1.732, nu = 0.25. High Vp/Vs (> 1.8) can indicate fluid saturation, partial melt, or high pore pressure."),
            ("[v]", "Poisson's ratio nu", "Ratio of transverse to longitudinal strain. Range: 0 (incompressible shear) to 0.5 (incompressible liquid). Typical crustal rocks: 0.24-0.28."),
        ],
        "it": [
            ("[P]", "Velocita' onda P (Vp)", "Onde compressive — le particelle vibrano parallele alla direzione di propagazione. Controllata dal modulo di bulk K + modulo di taglio G + densita' rho."),
            ("[S]", "Velocita' onda S (Vs)", "Onde di taglio — le particelle vibrano perpendicolari alla propagazione. Controllata solo dal modulo G. <strong>Vs = 0 nei fluidi</strong> perche' i fluidi non sopportano sforzi di taglio."),
            ("[r]", "Rapporto Vp/Vs", "Indicatore diagnostico chiave. Solido di Poisson: Vp/Vs = sqrt(3) ~ 1.732, nu = 0.25. Valori alti (> 1.8) indicano saturazione di fluidi, fusione parziale o alta pressione dei pori."),
            ("[v]", "Rapporto di Poisson nu", "Rapporto tra deformazione trasversale e longitudinale. Intervallo: 0 (solido incomprimibile in taglio) a 0.5 (liquido incomprimibile). Rocce crostali tipiche: 0.24-0.28."),
        ],
        "fr": [
            ("[P]", "Vitesse onde P (Vp)", "Ondes compressionnelles — les particules vibrent parallelement a la direction de propagation. Controlee par le module de compressibilite K, le module de cisaillement G et la densite rho."),
            ("[S]", "Vitesse onde S (Vs)", "Ondes de cisaillement — les particules vibrent perpendiculairement. Controlee uniquement par G. <strong>Vs = 0 dans les fluides</strong> car ils ne peuvent pas soutenir une contrainte de cisaillement."),
            ("[r]", "Rapport Vp/Vs", "Indicateur diagnostique cle. Solide de Poisson : Vp/Vs = sqrt(3) ~ 1.732, nu = 0.25. Valeurs elevees (> 1.8) indiquent saturation en fluide, fusion partielle ou haute pression des pores."),
            ("[v]", "Coefficient de Poisson nu", "Rapport deformation transversale / longitudinale. Plage : 0 a 0.5. Roches crustales typiques : 0.24-0.28."),
        ],
        "es": [
            ("[P]", "Velocidad onda P (Vp)", "Ondas compresionales — las particulas vibran paralelas a la direccion de propagacion. Controlada por modulo de bulk K + modulo de corte G + densidad rho."),
            ("[S]", "Velocidad onda S (Vs)", "Ondas de corte — las particulas vibran perpendiculares. Controlada solo por G. <strong>Vs = 0 en fluidos</strong> porque no pueden sostener esfuerzos de corte."),
            ("[r]", "Relacion Vp/Vs", "Indicador diagnostico clave. Solido de Poisson: Vp/Vs = sqrt(3) ~ 1.732, nu = 0.25. Valores altos (> 1.8) indican saturacion de fluidos, fusion parcial o alta presion de poros."),
            ("[v]", "Razon de Poisson nu", "Relacion deformacion transversal / longitudinal. Rango: 0 a 0.5. Rocas corticales tipicas: 0.24-0.28."),
        ],
        "de": [
            ("[P]", "P-Wellengeschwindigkeit (Vp)", "Kompressionswellen — Partikel schwingen parallel zur Ausbreitungsrichtung. Gesteuert durch Kompressionsmodul K + Schubmodul G + Dichte rho."),
            ("[S]", "S-Wellengeschwindigkeit (Vs)", "Scherwellen — Partikel schwingen senkrecht. Nur durch G gesteuert. <strong>Vs = 0 in Flussigkeiten</strong>, da diese keine Scherspannung aufnehmen konnen."),
            ("[r]", "Vp/Vs-Verhaltnis", "Wichtiger Diagnoseindikator. Poisson-Korper: Vp/Vs = sqrt(3) ~ 1.732, nu = 0.25. Hohe Werte (> 1.8) deuten auf Fluidsattigung, partielle Schmelze oder hohen Porendruck hin."),
            ("[v]", "Poisson-Zahl nu", "Verhaltnis von Quer- zu Langsdehnung. Bereich: 0 bis 0.5. Typische Krustengesteine: 0.24-0.28."),
        ],
    }
    _vm_grid = "<div class='concept-grid'>"
    for ico, title, body in _vp_concepts[lang]:
        _vm_grid += f"<div class='concept-card'><div class='cc-icon'>{ico}</div><div class='cc-title'>{title}</div><div class='cc-body'>{body}</div></div>"
    _vm_grid += "</div>"
    st.markdown(_vm_grid, unsafe_allow_html=True)

    _elastic_exp_title = {
        "en": "Elastic wave equations — full derivation",
        "it": "Equazioni delle onde elastiche — derivazione completa",
        "fr": "Équations des ondes élastiques — dérivation complète",
        "es": "Ecuaciones de ondas elásticas — derivación completa",
        "de": "Elastische Wellengleichungen — vollständige Herleitung",
    }
    _elastic_body = {
        "en": """
Where:
- **K** = bulk modulus (resistance to uniform compression, Pa)
- **G** = shear modulus (resistance to shear deformation, Pa)
- **rho** = density (kg/m3)
- **nu** = Poisson's ratio (dimensionless, 0 < nu < 0.5 for stable materials)

**Why does Vp > Vs always?**
Vp involves both K and G (compressional + shear restoring forces), while Vs only involves G.
Since K > 0 always, we have K + 4G/3 > G, hence Vp > Vs for any solid material.

**Fluid indicator:** In fully saturated rock, the pore fluid dramatically increases K (fluid is hard to compress)
while leaving G unchanged. This means Vp rises, Vs stays the same, and Vp/Vs increases above sqrt(3).
""",
        "it": """
Dove:
- **K** = modulo di bulk (resistenza alla compressione uniforme, Pa)
- **G** = modulo di taglio (resistenza alla deformazione di taglio, Pa)
- **rho** = densita' (kg/m3)
- **nu** = rapporto di Poisson (adimensionale, 0 < nu < 0.5 per materiali stabili)

**Perche' Vp > Vs sempre?**
Vp coinvolge sia K che G (forze di ripristino compressive + taglio), mentre Vs coinvolge solo G.
Poiche' K > 0 sempre, abbiamo K + 4G/3 > G, quindi Vp > Vs per qualsiasi materiale solido.

**Indicatore di fluido:** In roccia completamente satura, il fluido nei pori aumenta drasticamente K
(il fluido e' difficile da comprimere) lasciando G invariato. Cio' significa che Vp aumenta, Vs rimane uguale e Vp/Vs supera sqrt(3).
""",
        "fr": """
Ou:
- **K** = module de compressibilite (resistance a la compression uniforme, Pa)
- **G** = module de cisaillement (resistance a la deformation de cisaillement, Pa)
- **rho** = densite (kg/m3)
- **nu** = coefficient de Poisson (adimensionnel, 0 < nu < 0.5 pour les materiaux stables)

**Pourquoi Vp > Vs toujours?**
Vp implique K et G (forces de rappel compressionnelles + cisaillement), tandis que Vs n'implique que G.
Comme K > 0 toujours, on a K + 4G/3 > G, donc Vp > Vs pour tout materiau solide.

**Indicateur de fluide:** Dans une roche completement saturee, le fluide interstitiel augmente considerablement K
(le fluide est difficile a comprimer) en laissant G inchange, ce qui fait monter Vp/Vs au-dela de sqrt(3).
""",
        "es": """
Donde:
- **K** = modulo de compresion volumetrica (resistencia a la compresion uniforme, Pa)
- **G** = modulo de corte (resistencia a la deformacion de corte, Pa)
- **rho** = densidad (kg/m3)
- **nu** = razon de Poisson (adimensional, 0 < nu < 0.5 para materiales estables)

**Por que Vp > Vs siempre?**
Vp involucra tanto K como G (fuerzas restauradoras compresional + corte), mientras que Vs solo involucra G.
Dado que K > 0 siempre, tenemos K + 4G/3 > G, por lo tanto Vp > Vs para cualquier material solido.

**Indicador de fluido:** En roca completamente saturada, el fluido de los poros aumenta dramaticamente K
(el fluido es dificil de comprimir) sin cambiar G, por lo que Vp/Vs sube por encima de sqrt(3).
""",
        "de": """
Wobei:
- **K** = Kompressionsmodul (Widerstand gegen gleichmaessige Kompression, Pa)
- **G** = Schubmodul (Widerstand gegen Scherverformung, Pa)
- **rho** = Dichte (kg/m3)
- **nu** = Poisson-Zahl (dimensionslos, 0 < nu < 0.5 fur stabile Materialien)

**Warum ist Vp > Vs immer?**
Vp beinhaltet sowohl K als auch G (Kompressions- + Scherruckstellkrafte), wahrend Vs nur G beinhaltet.
Da K > 0 immer gilt, folgt K + 4G/3 > G und damit Vp > Vs fur jedes feste Material.

**Fluidindikator:** In vollstandig gesattigtem Gestein erhoht das Porenfluid K drastisch
(Fluid ist schwer komprimierbar), wahrend G unverandert bleibt. Dadurch steigt Vp/Vs uber sqrt(3).
""",
    }
    with st.expander(_elastic_exp_title[lang]):
        st.latex(r"V_P = \sqrt{\frac{K + \frac{4}{3}G}{\rho}}")
        st.latex(r"V_S = \sqrt{\frac{G}{\rho}}")
        st.latex(r"\nu = \frac{(V_P/V_S)^2 - 2}{2\left[(V_P/V_S)^2 - 1\right]}")
        st.markdown(_elastic_body[lang])

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
    _theory_intro = {
        "en": "This tab is your <strong>mathematical reference</strong> for every algorithm used in SeismicLens. Each section explains the concept from first principles, gives the key equations, and links to how the algorithm is used in the other tabs.",
        "it": "Questa sezione è il tuo <strong>riferimento matematico</strong> per ogni algoritmo usato in SeismicLens. Ogni sezione spiega il concetto dai principi fondamentali, fornisce le equazioni chiave e indica come l'algoritmo viene usato nelle altre schede.",
        "fr": "Cet onglet est votre <strong>référence mathématique</strong> pour chaque algorithme utilisé dans SeismicLens. Chaque section explique le concept depuis les premiers principes, donne les équations clés et indique comment l'algorithme est utilisé dans les autres onglets.",
        "es": "Esta pestaña es tu <strong>referencia matemática</strong> para cada algoritmo usado en SeismicLens. Cada sección explica el concepto desde los primeros principios, proporciona las ecuaciones clave e indica cómo se usa el algoritmo en las otras pestañas.",
        "de": "Dieser Tab ist Ihre <strong>mathematische Referenz</strong> für jeden in SeismicLens verwendeten Algorithmus. Jeder Abschnitt erklärt das Konzept von den Grundprinzipien, gibt die Schlüsselgleichungen an und zeigt, wie der Algorithmus in den anderen Tabs verwendet wird.",
    }
    st.markdown(f"<div class='tab-intro'>{_theory_intro[lang]}</div>", unsafe_allow_html=True)

    _exp1_title = {
        "en": "1 — Discrete Fourier Transform (DFT) and Fast FFT",
        "it": "1 — Trasformata di Fourier Discreta (DFT) e FFT veloce",
        "fr": "1 — Transformée de Fourier Discrète (DFT) et FFT rapide",
        "es": "1 — Transformada de Fourier Discreta (DFT) y FFT rápida",
        "de": "1 — Diskrete Fourier-Transformation (DFT) und schnelle FFT",
    }
    _exp1_body = {
        "en": """
#### What does the FFT actually do?
A time-domain signal **x[n]** is a list of N amplitude values sampled at rate fs.
The DFT converts it into N **complex-valued frequency coefficients** X[k].
Each coefficient tells you *how much of a pure sinusoid at frequency f_k = k·fs/N is present*.

The complex exponential (from **Euler's formula**) is the key building block:

So each DFT basis function is a **cosine** (real part) + **sine** (imaginary part) at frequency f_k.
The DFT projects the signal onto each basis function (inner product), measuring:
- `Re(X[k])` — how much of the cosine at f_k is in the signal
- `Im(X[k])` — how much of the sine at f_k is in the signal
- `|X[k]|` = sqrt(Re^2 + Im^2) — total **amplitude** at f_k
- angle X[k] = atan2(Im, Re) — **phase** at f_k

#### Why is the FFT faster?
The naive DFT requires **O(N^2)** multiplications. The Cooley-Tukey FFT (1965) exploits the periodicity
of e^(-j2pi*k*n/N) to recursively split the sum (decimation-in-time), reducing cost to **O(N log N)**.
For N = 100,000 this is ~10,000x faster.

#### One-sided spectrum for real signals
Real signals have **Hermitian symmetry**: X[N-k] = conj(X[k]). So only N/2+1 bins are unique.
`scipy.fft.rfft` exploits this and returns only the positive-frequency half.
Normalisation to recover true amplitude: A[k] = (2/N)|X[k]|, k = 1...N/2-1

**Hann window** is applied before the FFT to reduce **spectral leakage** — the smearing of energy
from a strong frequency bin into neighbouring bins that occurs when the signal is not an integer
number of cycles within the window.
""",
        "it": """
#### Cosa fa realmente la FFT?
Un segnale nel dominio del tempo **x[n]** è una lista di N valori di ampiezza campionati alla frequenza fs.
La DFT lo converte in N **coefficienti di frequenza a valori complessi** X[k].
Ogni coefficiente indica *quanta sinusoide pura alla frequenza f_k = k·fs/N è presente*.

L'esponenziale complesso (dalla **formula di Eulero**) è il mattone fondamentale:

Quindi ogni funzione base DFT è un **coseno** (parte reale) + **seno** (parte immaginaria) alla frequenza f_k.
La DFT proietta il segnale su ogni funzione base (prodotto interno), misurando:
- `Re(X[k])` — quanto coseno alla frequenza f_k è presente nel segnale
- `Im(X[k])` — quanto seno alla frequenza f_k è presente nel segnale
- `|X[k]|` = sqrt(Re² + Im²) — **ampiezza** totale alla frequenza f_k
- angolo X[k] = atan2(Im, Re) — **fase** alla frequenza f_k

#### Perché la FFT è più veloce?
La DFT ingenua richiede **O(N²)** moltiplicazioni. La FFT di Cooley-Tukey (1965) sfrutta la periodicità
di e^(-j2π·k·n/N) per suddividere ricorsivamente la somma, riducendo il costo a **O(N log N)**.
Per N = 100.000 questo è circa 10.000× più veloce.

#### Spettro monolaterale per segnali reali
I segnali reali hanno **simmetria Hermitiana**: X[N-k] = conj(X[k]). Quindi solo N/2+1 bin sono unici.
`scipy.fft.rfft` sfrutta questo e restituisce solo la metà a frequenza positiva.
Normalizzazione per recuperare l'ampiezza reale: A[k] = (2/N)|X[k]|, k = 1…N/2-1

La **finestra di Hann** viene applicata prima della FFT per ridurre la **perdita spettrale** — la dispersione
dell'energia da un bin forte ai bin vicini, che si verifica quando il segnale non contiene un numero intero
di cicli all'interno della finestra.
""",
        "fr": """
#### Que fait réellement la FFT ?
Un signal dans le domaine temporel **x[n]** est une liste de N valeurs d'amplitude échantillonnées à la fréquence fs.
La DFT le convertit en N **coefficients de fréquence à valeurs complexes** X[k].
Chaque coefficient indique *quelle quantité d'une sinusoïde pure à la fréquence f_k = k·fs/N est présente*.

L'exponentielle complexe (de la **formule d'Euler**) est le composant clé :

Ainsi chaque fonction de base DFT est un **cosinus** (partie réelle) + **sinus** (partie imaginaire) à la fréquence f_k.
La DFT projette le signal sur chaque fonction de base, mesurant :
- `Re(X[k])` — quelle part du cosinus à f_k est dans le signal
- `Im(X[k])` — quelle part du sinus à f_k est dans le signal
- `|X[k]|` = sqrt(Re² + Im²) — **amplitude** totale à f_k
- angle X[k] = atan2(Im, Re) — **phase** à f_k

#### Pourquoi la FFT est-elle plus rapide ?
La DFT naïve nécessite **O(N²)** multiplications. La FFT de Cooley-Tukey (1965) exploite la périodicité
de e^(-j2π·k·n/N) pour diviser récursivement la somme, réduisant le coût à **O(N log N)**.
Pour N = 100 000, c'est environ 10 000× plus rapide.

La **fenêtre de Hann** est appliquée avant la FFT pour réduire la **fuite spectrale**.
""",
        "es": """
#### ¿Qué hace realmente la FFT?
Una señal en el dominio del tiempo **x[n]** es una lista de N valores de amplitud muestreados a la tasa fs.
La DFT la convierte en N **coeficientes de frecuencia de valores complejos** X[k].
Cada coeficiente indica *cuánta sinusoide pura a la frecuencia f_k = k·fs/N está presente*.

- `Re(X[k])` — cuánto del coseno en f_k está en la señal
- `Im(X[k])` — cuánto del seno en f_k está en la señal
- `|X[k]|` = sqrt(Re² + Im²) — **amplitud** total en f_k
- ángulo X[k] = atan2(Im, Re) — **fase** en f_k

#### ¿Por qué la FFT es más rápida?
La DFT ingenua requiere **O(N²)** multiplicaciones. La FFT de Cooley-Tukey (1965) reduce el coste a **O(N log N)**.
Para N = 100.000 esto es ~10.000× más rápido.

La **ventana de Hann** se aplica antes de la FFT para reducir la **fuga espectral**.
""",
        "de": """
#### Was macht die FFT tatsächlich?
Ein zeitbereichiges Signal **x[n]** ist eine Liste von N Amplitudenwerten, die mit der Rate fs abgetastet wurden.
Die DFT konvertiert es in N **komplexwertige Frequenzkoeffizienten** X[k].
Jeder Koeffizient gibt an, *wie viel einer reinen Sinuswelle bei der Frequenz f_k = k·fs/N vorhanden ist*.

- `Re(X[k])` — wie viel des Kosinus bei f_k im Signal enthalten ist
- `Im(X[k])` — wie viel des Sinus bei f_k im Signal enthalten ist
- `|X[k]|` = sqrt(Re² + Im²) — **Amplitude** bei f_k
- Winkel X[k] = atan2(Im, Re) — **Phase** bei f_k

#### Warum ist die FFT schneller?
Die naive DFT benötigt **O(N²)** Multiplikationen. Die Cooley-Tukey-FFT (1965) reduziert den Aufwand auf **O(N log N)**.
Für N = 100.000 ist das etwa 10.000× schneller.

Das **Hann-Fenster** wird vor der FFT angewendet, um **Spektralleckage** zu reduzieren.
""",
    }

    # ── 1. DFT / FFT ──────────────────────────────────────────────────────────
    with st.expander(_exp1_title[lang], expanded=True):
        st.markdown(_exp1_body[lang])
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi k n}{N}}, \quad k = 0, 1, \ldots, N-1")
        st.latex(r"e^{j\theta} = \cos(\theta) + j\sin(\theta)")
        st.latex(r"A[k] = \frac{2}{N} |X[k]|, \quad k = 1, \ldots, \frac{N}{2}-1")
        st.latex(r"w[n] = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N-1}\right)\right)")

    # ── 2. Butterworth Filter ──────────────────────────────────────────────────
    _exp2b_title = {
        "en": "2 — Butterworth Bandpass Filter",
        "it": "2 — Filtro Butterworth passa-banda",
        "fr": "2 — Filtre Butterworth passe-bande",
        "es": "2 — Filtro Butterworth pasa-banda",
        "de": "2 — Butterworth-Bandpassfilter",
    }
    _exp2b_intro = {
        "en": "The Butterworth filter is **maximally flat in the passband** — no ripple. A bandpass is formed by cascading a high-pass and a low-pass. SOS (Second-Order Sections) ensures numerical stability for high orders. `sosfiltfilt` runs the filter forward then backward: zero phase distortion, arrival times preserved exactly.",
        "it": "Il filtro Butterworth è **massimamente piatto nella banda passante** — nessuna ondulazione. Un filtro passa-banda si ottiene cascadando un passa-alto e un passa-basso. Le SOS (Sezioni del Secondo Ordine) garantiscono la stabilità numerica per ordini elevati. `sosfiltfilt` applica il filtro in avanti poi all'indietro: distorsione di fase zero, tempi di arrivo preservati esattamente.",
        "fr": "Le filtre Butterworth est **maximalement plat dans la bande passante** — pas d'ondulation. Un passe-bande est formé en cascadant un passe-haut et un passe-bas. Les SOS garantissent la stabilité numérique. `sosfiltfilt` applique le filtre en aller puis en retour : distorsion de phase nulle.",
        "es": "El filtro Butterworth es **maximalmente plano en la banda de paso** — sin ondulaciones. Un pasa-banda se forma cascadeando un pasa-alto y un pasa-bajo. Las SOS garantizan la estabilidad numérica. `sosfiltfilt` aplica el filtro hacia adelante y atrás: distorsión de fase cero.",
        "de": "Das Butterworth-Filter ist **maximal flach im Durchlassbereich** — keine Welligkeit. Ein Bandpass wird durch Kaskadierung eines Hochpass- und Tiefpassfilters gebildet. SOS sichert numerische Stabilität. `sosfiltfilt` läuft vorwärts und rückwärts: keine Phasenverzerrung.",
    }
    _exp2b_current = {
        "en": f"Current filter: order {filter_order}, effective order **{2*filter_order}** (zero-phase), roll-off = **{40*filter_order} dB/decade**.",
        "it": f"Filtro attuale: ordine {filter_order}, ordine effettivo **{2*filter_order}** (zero-phase), roll-off = **{40*filter_order} dB/decade**.",
        "fr": f"Filtre actuel : ordre {filter_order}, ordre effectif **{2*filter_order}** (zero-phase), roll-off = **{40*filter_order} dB/decennie**.",
        "es": f"Filtro actual: orden {filter_order}, orden efectivo **{2*filter_order}** (fase cero), roll-off = **{40*filter_order} dB/decada**.",
        "de": f"Aktueller Filter: Ordnung {filter_order}, effektive Ordnung **{2*filter_order}** (Nullphase), Roll-off = **{40*filter_order} dB/Dekade**.",
    }
    with st.expander(_exp2b_title[lang]):
        st.markdown(_exp2b_intro[lang])
        st.markdown(_exp2b_current[lang])
        st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}")
        st.latex(r"H_{BP}(s) = H_{HP}(s) \cdot H_{LP}(s)")
        st.latex(r"H(z) = \prod_{k=1}^{n/2} \frac{b_{0k} + b_{1k} z^{-1} + b_{2k} z^{-2}}{1 + a_{1k} z^{-1} + a_{2k} z^{-2}}")


    # ── 3. STA/LTA ────────────────────────────────────────────────────────────
    _exp3_title = {
        "en": "3 — STA/LTA Seismic Phase Detector",
        "it": "3 — Rilevatore di fasi sismiche STA/LTA",
        "fr": "3 — Détecteur de phases sismiques STA/LTA",
        "es": "3 — Detector de fases sísmicas STA/LTA",
        "de": "3 — STA/LTA seismischer Phasendetektor",
    }
    _exp3_body = {
        "en": f"""
#### Intuition
When a seismic wave arrives, the signal energy suddenly **increases sharply**.
STA/LTA exploits this: the short-term window (STA) tracks the instantaneous energy;
the long-term window (LTA) tracks the background noise level.
When the ratio STA/LTA spikes above a threshold, a phase is declared.

**Why square the samples?**
Squaring makes the metric invariant to the sign (polarity) of the signal — a negative spike
is just as impulsive as a positive one. It also tracks **energy** rather than amplitude.

**O(N) implementation** via prefix sums:
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```
Without prefix sums, a naive sliding-window sum would be O(N x max(N_sta, N_lta)) — orders of magnitude slower.

Reference: Allen, R.V. (1978). *Automatic earthquake recognition and timing from single traces*. BSSA, 68(5).
""",
        "it": f"""
#### Intuizione
Quando arriva un'onda sismica, l'energia del segnale **aumenta bruscamente**.
Lo STA/LTA sfrutta questo: la finestra a breve termine (STA) traccia l'energia istantanea;
la finestra a lungo termine (LTA) traccia il livello di rumore di fondo.
Quando il rapporto STA/LTA supera una soglia, viene dichiarata una fase.

**Perche' elevare al quadrato i campioni?**
Il quadrato rende la metrica invariante al segno (polarita') del segnale — un picco negativo
e' impulsivo quanto uno positivo. Misura inoltre l'**energia** anziche' l'ampiezza.

**Implementazione O(N)** tramite prefix sums:
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```
Senza prefix sums, una somma a finestra scorrevole sarebbe O(N x max(N_sta, N_lta)) — ordini di grandezza piu' lento.

Riferimento: Allen, R.V. (1978). *Automatic earthquake recognition and timing from single traces*. BSSA, 68(5).
""",
        "fr": f"""
#### Intuition
Quand une onde sismique arrive, l'energie du signal **augmente brusquement**.
Le STA/LTA exploite cela : la fenetre a court terme (STA) suit l'energie instantanee ;
la fenetre a long terme (LTA) suit le niveau de bruit de fond.
Quand le rapport STA/LTA depasse un seuil, une phase est declaree.

**Pourquoi elever au carre?** Le carre rend la metrique invariante au signe (polarite) du signal.
Il mesure l'**energie** plutot que l'amplitude.

**Implementation O(N)** par sommes de prefixes:
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```
Reference: Allen, R.V. (1978). BSSA, 68(5).
""",
        "es": f"""
#### Intuicion
Cuando llega una onda sismica, la energia de la senal **aumenta bruscamente**.
El STA/LTA aprovecha esto: la ventana a corto plazo (STA) rastrea la energia instantanea;
la ventana a largo plazo (LTA) rastrea el nivel de ruido de fondo.
Cuando el cociente STA/LTA supera un umbral, se declara una fase.

**Por que elevar al cuadrado?** El cuadrado hace la metrica invariante al signo (polaridad) de la senal.
Mide la **energia** en lugar de la amplitud.

**Implementacion O(N)** mediante sumas de prefijos:
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```
Referencia: Allen, R.V. (1978). BSSA, 68(5).
""",
        "de": f"""
#### Intuition
Wenn eine seismische Welle ankommt, steigt die Signalenergie **schlagartig an**.
STA/LTA nutzt dies: Das kurzfristige Fenster (STA) verfolgt die momentane Energie;
das langfristige Fenster (LTA) verfolgt den Hintergrundrauschpegel.
Wenn das Verhaltnis STA/LTA einen Schwellenwert uberschreitet, wird eine Phase deklariert.

**Warum quadrieren?** Das Quadrat macht die Metrik unabhangig vom Vorzeichen (Polaritat).
Es misst **Energie** statt Amplitude.

**O(N)-Implementierung** uber Prafix-Summen:
```python
cs = np.cumsum(x**2)
STA[t] = (cs[t] - cs[t - N_sta]) / N_sta
LTA[t] = (cs[t] - cs[t - N_lta]) / N_lta
```
Referenz: Allen, R.V. (1978). BSSA, 68(5).
""",
    }
    with st.expander(_exp3_title[lang]):
        st.latex(r"\text{STA}(t) = \frac{1}{N_{sta}} \sum_{k=0}^{N_{sta}-1} x^2[t-k]")
        st.latex(r"\text{LTA}(t) = \frac{1}{N_{lta}} \sum_{k=0}^{N_{lta}-1} x^2[t-k]")
        st.latex(r"R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}")
        st.markdown(_exp3_body[lang])

    # ── 4. Spectrogram / STFT ─────────────────────────────────────────────────
    _exp4_title = {
        "en": "4 — Spectrogram and Time-Frequency Analysis",
        "it": "4 — Spettrogramma e analisi tempo-frequenza",
        "fr": "4 — Spectrogramme et analyse temps-fréquence",
        "es": "4 — Espectrograma y análisis tiempo-frecuencia",
        "de": "4 — Spektrogramm und Zeit-Frequenz-Analyse",
    }
    _exp4_body = {
        "en": """
#### Why a spectrogram?
The FFT gives the **global** frequency content over the entire signal — it cannot tell you *when*
each frequency is present. A spectrogram solves this by computing the FFT locally in time.

The spectrogram is the squared magnitude: |STFT(tau,f)|^2, displayed in dB.

#### Heisenberg-Gabor Uncertainty Principle
There is a **fundamental trade-off** between time and frequency resolution:

| Window size | Time res. | Freq. res. | Best for |
|---|---|---|---|
| Short (32-128 pts) | Fine (ms) | Coarse (Hz) | Impulsive arrivals |
| Long (512-2048 pts) | Coarse (100s ms) | Fine (0.1 Hz) | Dispersive waves |

In SeismicLens, `scipy.signal.spectrogram` uses a Hann window with 75% overlap.

#### Reading seismic spectrograms
- **Vertical bright streaks** at early time — P or S wave arrival (broadband, impulsive)
- **Horizontal bright bands** — monochromatic noise (e.g., 50/60 Hz power line)
- **Energy sweeping from high to low frequency** — dispersive surface wave (Love or Rayleigh)
- **Low-frequency stripe throughout** — ocean microseisms (0.1-0.3 Hz)
""",
        "it": """
#### Perche' uno spettrogramma?
La FFT fornisce il contenuto in frequenza **globale** dell'intero segnale — non puo' dire *quando*
ogni frequenza e' presente. Lo spettrogramma risolve questo calcolando la FFT localmente nel tempo.

Lo spettrogramma e' il quadrato del modulo: |STFT(tau,f)|^2, visualizzato in dB.

#### Principio di indeterminazione di Heisenberg-Gabor
C'e' un **compromesso fondamentale** tra risoluzione temporale e frequenziale:

| Dimensione finestra | Ris. temporale | Ris. frequenziale | Ideale per |
|---|---|---|---|
| Corta (32-128 camp.) | Fine (ms) | Grossolana (Hz) | Onset impulsivi |
| Lunga (512-2048 camp.) | Grossolana (100s ms) | Fine (0.1 Hz) | Onde dispersive |

In SeismicLens, `scipy.signal.spectrogram` usa una finestra di Hann con 75% di sovrapposizione.

#### Leggere gli spettrogrammi sismici
- **Strisce verticali luminose** all'inizio — arrivo onda P o S (broadband, impulsivo)
- **Bande orizzontali luminose** — rumore monocromatico (es. rete elettrica 50/60 Hz)
- **Energia che scende da alta a bassa frequenza nel tempo** — onda superficiale dispersiva (Love o Rayleigh)
- **Striscia a bassa frequenza per tutta la durata** — microseismi oceanici (0.1-0.3 Hz)
""",
        "fr": """
#### Pourquoi un spectrogramme?
La FFT donne le contenu frequentiel **global** sur l'ensemble du signal — elle ne peut pas dire *quand*
chaque frequence est presente. Le spectrogramme resout cela en calculant la FFT localement dans le temps.

Le spectrogramme est le carre du module : |STFT(tau,f)|^2, affiche en dB.

#### Principe d'incertitude de Heisenberg-Gabor
Il y a un **compromis fondamental** entre resolution temporelle et frequentielle.

In SeismicLens, `scipy.signal.spectrogram` utilise une fenetre de Hann avec 75% de recouvrement.

#### Lecture des spectrogrammes sismiques
- **Bandes verticales lumineuses** — arrivee onde P ou S (impulsif)
- **Bandes horizontales** — bruit monochromatique (reseau electrique 50/60 Hz)
- **Energie qui descend de haute a basse frequence** — onde de surface dispersive
- **Bande basse frequence continue** — microseismes oceaniques (0.1-0.3 Hz)
""",
        "es": """
#### Por que un espectrograma?
La FFT da el contenido frecuencial **global** sobre toda la senal — no puede decir *cuando*
esta presente cada frecuencia. El espectrograma soluciona esto calculando la FFT localmente en el tiempo.

El espectrograma es el cuadrado del modulo: |STFT(tau,f)|^2, mostrado en dB.

#### Principio de incertidumbre de Heisenberg-Gabor
Existe un **compromiso fundamental** entre resolucion temporal y frecuencial.

En SeismicLens, `scipy.signal.spectrogram` usa una ventana de Hann con 75% de solapamiento.

#### Lectura de espectrogramas sismicos
- **Franjas verticales brillantes** — llegada onda P o S (impulsivo)
- **Bandas horizontales** — ruido monocromatico (red electrica 50/60 Hz)
- **Energia que baja de alta a baja frecuencia** — onda superficial dispersiva
- **Franja de baja frecuencia continua** — microseismos oceanicos (0.1-0.3 Hz)
""",
        "de": """
#### Warum ein Spektrogramm?
Die FFT gibt den **globalen** Frequenzinhalt uber das gesamte Signal — sie kann nicht sagen, *wann*
jede Frequenz vorhanden ist. Das Spektrogramm lost dies durch lokale FFT-Berechnung in der Zeit.

Das Spektrogramm ist der quadrierte Betrag: |STFT(tau,f)|^2, angezeigt in dB.

#### Heisenberg-Gabor Unbestimmtheitsrelation
Es gibt einen **fundamentalen Kompromiss** zwischen Zeit- und Frequenzauflosung.

In SeismicLens verwendet `scipy.signal.spectrogram` ein Hann-Fenster mit 75% Uberlappung.

#### Lesen seismischer Spektrogramme
- **Vertikale helle Streifen** — P- oder S-Wellen-Ankunft (impulsiv)
- **Horizontale Bander** — monochromatisches Rauschen (Stromnetz 50/60 Hz)
- **Energie, die von hoch nach niedrig fallt** — dispersive Oberflachenwelle
- **Durchgehender Niedrigfrequenz-Streifen** — ozeanische Mikroseismik (0.1-0.3 Hz)
""",
    }
    with st.expander(_exp4_title[lang]):
        st.latex(r"\text{STFT}(\tau, f) = \sum_{n} x[n] \cdot w[n - \tau] \cdot e^{-j2\pi f n / f_s}")
        st.latex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}")
        st.markdown(_exp4_body[lang])

    # ── 5. Seismic Waves ──────────────────────────────────────────────────────
    _exp5_title = {
        "en": "5 — Seismic Wave Physics",
        "it": "5 — Fisica delle onde sismiche",
        "fr": "5 — Physique des ondes sismiques",
        "es": "5 — Física de las ondas sísmicas",
        "de": "5 — Physik der seismischen Wellen",
    }
    _exp5_body = {
        "en": """
Seismic waves are mechanical waves that propagate through the Earth by elastic deformation.
They are classified into **body waves** (travel through the interior) and **surface waves**
(travel along the Earth's surface).

| Wave type | Polarisation | Speed | Typical freq. (local EQ) | Notes |
|---|---|---|---|---|
| **P (Primary)** | Compressional, parallel | Fastest | 6-12 Hz | Arrives first |
| **S (Secondary)** | Shear, perpendicular | ~57-60% of Vp | 2-6 Hz | Main shaking |
| **Rayleigh** | Elliptical (retrograde) | ~92% of Vs | 0.3-1 Hz | Rolling motion |
| **Love** | Horizontal transverse | ~Vs | 0.3-1 Hz | Horizontal only |

#### Wadati method for epicentral distance
The S-P time difference is independent of origin time and depends only on distance:

For the IASP91 model (upper crust: Vp = 5.80, Vs = 3.36 km/s): 1 second of S-P delay ~ 8 km epicentral distance.
""",
        "it": """
Le onde sismiche sono onde meccaniche che si propagano attraverso la Terra per deformazione elastica.
Si classificano in **onde di volume** (viaggiano attraverso l'interno) e **onde superficiali**
(viaggiano lungo la superficie terrestre).

| Tipo di onda | Polarizzazione | Velocita' | Freq. tipica (sisma locale) | Note |
|---|---|---|---|---|
| **P (Primaria)** | Compressiva, parallela | Massima | 6-12 Hz | Arriva per prima |
| **S (Secondaria)** | Taglio, perpendicolare | ~57-60% di Vp | 2-6 Hz | Scuotimento principale |
| **Rayleigh** | Ellittica (retrograda) | ~92% di Vs | 0.3-1 Hz | Moto ondeggiante |
| **Love** | Trasversale orizzontale | ~Vs | 0.3-1 Hz | Solo orizzontale |

#### Metodo di Wadati per la distanza epicentrale
La differenza di tempi S-P e' indipendente dall'origine e dipende solo dalla distanza:

Per il modello IASP91 (crosta superiore: Vp = 5.80, Vs = 3.36 km/s): 1 secondo di ritardo S-P ~ 8 km di distanza epicentrale.
""",
        "fr": """
Les ondes sismiques sont des ondes mecaniques qui se propagent dans la Terre par deformation elastique.
Elles se classent en **ondes de volume** (voyagent dans l'interieur) et **ondes de surface**.

| Type d'onde | Polarisation | Vitesse | Freq. typique (seisme local) | Notes |
|---|---|---|---|---|
| **P (Primaire)** | Compressionnelle, parallele | Maximale | 6-12 Hz | Arrive en premier |
| **S (Secondaire)** | Cisaillement, perpendiculaire | ~57-60% de Vp | 2-6 Hz | Secousse principale |
| **Rayleigh** | Elliptique (retrograde) | ~92% de Vs | 0.3-1 Hz | Mouvement roulant |
| **Love** | Transversale horizontale | ~Vs | 0.3-1 Hz | Horizontale uniquement |

#### Methode de Wadati
Pour le modele IASP91: 1 seconde de retard S-P ~ 8 km de distance epicentrale.
""",
        "es": """
Las ondas sismicas son ondas mecanicas que se propagan por la Tierra por deformacion elastica.
Se clasifican en **ondas de cuerpo** (viajan por el interior) y **ondas superficiales**.

| Tipo de onda | Polarizacion | Velocidad | Freq. tipica (sismo local) | Notas |
|---|---|---|---|---|
| **P (Primaria)** | Compresional, paralela | Maxima | 6-12 Hz | Llega primero |
| **S (Secundaria)** | Corte, perpendicular | ~57-60% de Vp | 2-6 Hz | Sacudida principal |
| **Rayleigh** | Eliptica (retrograda) | ~92% de Vs | 0.3-1 Hz | Movimiento ondulante |
| **Love** | Transversal horizontal | ~Vs | 0.3-1 Hz | Solo horizontal |

#### Metodo de Wadati
Para el modelo IASP91: 1 segundo de retardo S-P ~ 8 km de distancia epicentral.
""",
        "de": """
Seismische Wellen sind mechanische Wellen, die sich durch elastische Verformung durch die Erde ausbreiten.
Sie werden in **Raumwellen** (reisen durch das Innere) und **Oberflachenwellen** eingeteilt.

| Wellentyp | Polarisation | Geschwindigkeit | Typische Freq. (Nahbeben) | Hinweise |
|---|---|---|---|---|
| **P (Primarwelle)** | Kompression, parallel | Schnellste | 6-12 Hz | Kommt zuerst an |
| **S (Sekundarwelle)** | Scherung, senkrecht | ~57-60% von Vp | 2-6 Hz | Haupterschutterung |
| **Rayleigh** | Elliptisch (retrograd) | ~92% von Vs | 0.3-1 Hz | Rollende Bewegung |
| **Love** | Horizontal transversal | ~Vs | 0.3-1 Hz | Nur horizontal |

#### Wadati-Methode
Fur das IASP91-Modell: 1 Sekunde S-P-Verzogerung ~ 8 km Epizentraldistanz.
""",
    }
    with st.expander(_exp5_title[lang]):
        st.latex(r"V_P = \sqrt{\frac{K + \frac{4}{3}G}{\rho}} \qquad V_S = \sqrt{\frac{G}{\rho}}")
        st.markdown(_exp5_body[lang])
        st.latex(r"t_S - t_P = R \left(\frac{1}{V_S} - \frac{1}{V_P}\right)")
        st.latex(r"d \approx (t_S - t_P) \cdot \frac{V_P \cdot V_S}{V_P - V_S}")

    # ── 6. Magnitude Scales ───────────────────────────────────────────────────
    _exp6_title = {
        "en": "6 — Earthquake Magnitude Scales",
        "it": "6 — Scale di magnitudo dei terremoti",
        "fr": "6 — Échelles de magnitude des séismes",
        "es": "6 — Escalas de magnitud de terremotos",
        "de": "6 — Erdbeben-Magnituden-Skalen",
    }
    _exp6_richter = {
        "en": "#### Richter Local Magnitude (M_L)\nDefined by Charles Richter (1935) for Southern California. Uses peak amplitude on a Wood-Anderson seismograph, corrected for distance. **SeismicLens** scales synthetic amplitudes as:",
        "it": "#### Magnitudo locale di Richter (M_L)\nDefinita da Charles Richter (1935) per la California meridionale. Usa l'ampiezza di picco su un sismografo Wood-Anderson, corretta per la distanza. **SeismicLens** scala le ampiezze sintetiche come:",
        "fr": "#### Magnitude locale de Richter (M_L)\nDefinie par Charles Richter (1935) pour le sud de la Californie. Utilise l'amplitude maximale sur un sismographe Wood-Anderson, corrigee pour la distance. **SeismicLens** met a l'echelle les amplitudes synthetiques comme :",
        "es": "#### Magnitud local de Richter (M_L)\nDefinida por Charles Richter (1935) para el sur de California. Usa la amplitud pico en un sismografo Wood-Anderson, corregida por distancia. **SeismicLens** escala las amplitudes sinteticas como:",
        "de": "#### Richter-Lokalmagnitude (M_L)\nVon Charles Richter (1935) fur Sudkalifornien definiert. Verwendet die Spitzenamplitude an einem Wood-Anderson-Seismographen. **SeismicLens** skaliert synthetische Amplituden wie:",
    }
    _exp6_mw = {
        "en": "#### Moment Magnitude (M_w)\nThe modern standard (Hanks & Kanamori, 1979). M_w has **no saturation** at large magnitudes and is the preferred scale for large events.\n\nWhere M0 = seismic moment (Nm), mu = rigidity (~3e10 Pa), A_fault = rupture area (m2), D_bar = average slip (m).\n\n| M_w | Energy (J) | Equivalent |\n|---|---|---|\n| 2.0 | 1e11 | Small quarry blast |\n| 5.0 | 1e14 | Hiroshima bomb |\n| 7.0 | 1e16 | Large destructive quake |\n| 9.0 | 1e18 | 2011 Tohoku earthquake |",
        "it": "#### Magnitudo del momento (M_w)\nLo standard moderno (Hanks & Kanamori, 1979). M_w **non satura** alle grandi magnitude ed e' la scala preferita per i grandi terremoti.\n\nDove M0 = momento sismico (Nm), mu = rigidita' (~3e10 Pa), A_fault = area di rottura (m2), D_bar = scorrimento medio (m).\n\n| M_w | Energia (J) | Equivalente |\n|---|---|---|\n| 2.0 | 1e11 | Piccola esplosione in cava |\n| 5.0 | 1e14 | Bomba di Hiroshima |\n| 7.0 | 1e16 | Grande terremoto distruttivo |\n| 9.0 | 1e18 | Terremoto Tohoku 2011 |",
        "fr": "#### Magnitude du moment (M_w)\nLe standard moderne (Hanks & Kanamori, 1979). M_w n'a **pas de saturation** aux grandes magnitudes.\n\n| M_w | Energie (J) | Equivalent |\n|---|---|---|\n| 2.0 | 1e11 | Petite explosion en carriere |\n| 5.0 | 1e14 | Bombe d'Hiroshima |\n| 7.0 | 1e16 | Grand seisme destructeur |\n| 9.0 | 1e18 | Seisme Tohoku 2011 |",
        "es": "#### Magnitud del momento (M_w)\nEl estandar moderno (Hanks & Kanamori, 1979). M_w no tiene **saturacion** en grandes magnitudes.\n\n| M_w | Energia (J) | Equivalente |\n|---|---|---|\n| 2.0 | 1e11 | Explosion en cantera |\n| 5.0 | 1e14 | Bomba de Hiroshima |\n| 7.0 | 1e16 | Gran terremoto destructivo |\n| 9.0 | 1e18 | Terremoto Tohoku 2011 |",
        "de": "#### Momenten-Magnitude (M_w)\nDer moderne Standard (Hanks & Kanamori, 1979). M_w hat **keine Sattigung** bei grossen Magnituden.\n\n| M_w | Energie (J) | Aquivalent |\n|---|---|---|\n| 2.0 | 1e11 | Kleiner Steinbruchsprengung |\n| 5.0 | 1e14 | Hiroshima-Bombe |\n| 7.0 | 1e16 | Grosses Zerstorungsbeben |\n| 9.0 | 1e18 | Tohoku-Erdbeben 2011 |",
    }
    with st.expander(_exp6_title[lang]):
        st.markdown(_exp6_richter[lang])
        st.latex(r"M_L = \log_{10}(A) - \log_{10}(A_0(\Delta))")
        st.latex(r"A_{peak} \propto 10^{0.8 M - 2.5}")
        st.markdown(_exp6_mw[lang])
        st.latex(r"M_w = \frac{2}{3} \log_{10}(M_0) - 10.7")
        st.latex(r"M_0 = \mu \cdot A_{fault} \cdot \bar{D}")

    # ── 7. Numerical Methods ──────────────────────────────────────────────────
    _exp7_title = {
        "en": "7 — Numerical Methods and Implementation",
        "it": "7 — Metodi numerici e implementazione",
        "fr": "7 — Méthodes numériques et implémentation",
        "es": "7 — Métodos numéricos e implementación",
        "de": "7 — Numerische Methoden und Implementierung",
    }
    _exp7_body = {
        "en": """
#### Signal tapering (Tukey window)
Before FFT, the signal is tapered at both ends (5% cosine taper per side) to smoothly reach zero.
This prevents **spectral leakage** caused by the abrupt truncation of the signal.
Where alpha = 0.10 (10% total taper — 5% each end).

#### Spectral centroid and bandwidth
The spectral centroid (first moment) and bandwidth (second central moment) are computed on
the one-sided amplitude spectrum A[k]. These are the frequency-domain analogues of the mean
and standard deviation of a probability distribution.

#### Welch PSD normalisation
The Welch PSD is normalised so that Parseval's theorem holds:
the integral of the PSD equals the signal variance — a useful sanity check.
""",
        "it": """
#### Tapering del segnale (finestra di Tukey)
Prima della FFT, il segnale viene smussato alle estremita' (5% di cosinusoide per lato) per raggiungere
dolcemente lo zero. Questo previene la **perdita spettrale** causata dalla brusca troncatura del segnale.
Dove alpha = 0.10 (10% di tapering totale — 5% per ogni estremita').

#### Centroide spettrale e larghezza di banda
Il centroide spettrale (primo momento) e la larghezza di banda (secondo momento centrale) vengono calcolati
sullo spettro di ampiezza monolaterale A[k]. Sono gli analoghi nel dominio della frequenza della media
e della deviazione standard di una distribuzione di probabilita'.

#### Normalizzazione della PSD Welch
La PSD Welch e' normalizzata in modo che il teorema di Parseval sia verificato:
l'integrale della PSD equivale alla varianza del segnale — un'utile verifica di coerenza.
""",
        "fr": """
#### Fenestrage du signal (fenetre de Tukey)
Avant la FFT, le signal est effile aux deux extremites (5% de cosinusoide par cote) pour atteindre
doucement zero. Cela evite la **fuite spectrale** causee par la troncature abrupte du signal.
Ou alpha = 0.10 (10% d'effilement total).

#### Centroide spectral et largeur de bande
Le centroide spectral (premier moment) et la largeur de bande (second moment central) sont calcules
sur le spectre d'amplitude monolateral A[k]. Ce sont les analogues frequentiels de la moyenne
et de l'ecart-type d'une distribution de probabilite.

#### Normalisation de la PSD Welch
La PSD Welch est normalisee de sorte que le theoreme de Parseval soit verifie :
l'integrale de la PSD egale la variance du signal.
""",
        "es": """
#### Ventaneo de la senal (ventana de Tukey)
Antes de la FFT, la senal se suaviza en ambos extremos (5% de coseno por lado) para llegar
suavemente a cero. Esto evita la **fuga espectral** causada por el truncamiento abrupto de la senal.
Donde alpha = 0.10 (10% de ventaneo total).

#### Centroide espectral y ancho de banda
El centroide espectral (primer momento) y el ancho de banda (segundo momento central) se calculan
sobre el espectro de amplitud unilateral A[k]. Son los analogos frecuenciales de la media
y la desviacion estandar de una distribucion de probabilidad.

#### Normalizacion de la PSD Welch
La PSD Welch esta normalizada de modo que el teorema de Parseval se cumple:
la integral de la PSD es igual a la varianza de la senal.
""",
        "de": """
#### Signal-Tapering (Tukey-Fenster)
Vor der FFT wird das Signal an beiden Enden sanft auf null gebracht (5% Kosinus-Taper pro Seite).
Dies verhindert **Spektralausblutung** durch den abrupten Abbruch des Signals.
Wobei alpha = 0.10 (10% Gesamt-Taper — je 5% pro Ende).

#### Spektralzentroid und Bandbreite
Der Spektralzentroid (erstes Moment) und die Bandbreite (zweites zentrales Moment) werden aus
dem einseitigen Amplitudenspektrum A[k] berechnet. Sie sind die frequenzbereichs-Analoga von
Mittelwert und Standardabweichung einer Wahrscheinlichkeitsverteilung.

#### Welch-PSD-Normierung
Die Welch-PSD ist normiert, sodass das Parseval-Theorem gilt:
Das Integral der PSD entspricht der Signalvarianz — eine nutzliche Plausibilitatsprufung.
""",
    }
    with st.expander(_exp7_title[lang]):
        st.latex(r"w_{Tukey}[n] = \begin{cases} \frac{1}{2}\left[1 - \cos\!\left(\frac{2\pi n}{\alpha N}\right)\right] & 0 \le n < \frac{\alpha N}{2} \\ 1 & \frac{\alpha N}{2} \le n < N\left(1-\frac{\alpha}{2}\right) \\ \frac{1}{2}\left[1 - \cos\!\left(\frac{2\pi(N-n)}{\alpha N}\right)\right] & N\left(1-\frac{\alpha}{2}\right) \le n \le N \end{cases}")
        st.latex(r"f_c = \frac{\sum_k f_k \cdot A[k]}{\sum_k A[k]}")
        st.latex(r"\sigma_f = \sqrt{\frac{\sum_k (f_k - f_c)^2 \cdot A[k]}{\sum_k A[k]}}")
        st.latex(r"\int_{-\infty}^{\infty} S(f)\, df = \text{var}(x) = \frac{1}{N}\sum_{n} x^2[n]")
        st.markdown(_exp7_body[lang])

    # ── References ────────────────────────────────────────────────────────────
    _ref_title = {
        "en": "Key References",
        "it": "Riferimenti principali",
        "fr": "Références principales",
        "es": "Referencias principales",
        "de": "Wichtige Referenzen",
    }
    st.markdown(f"""
<div class='sl-card' style='margin-top:20px;'>
<div class='theory-title'>{_ref_title[lang]}</div>
<div class='theory-body' style='font-size:12px; line-height:1.8;'>
Allen, R.V. (1978). Automatic earthquake recognition and timing from single traces. <em>BSSA</em>, 68(5).<br>
Butterworth, S. (1930). On the theory of filter amplifiers. <em>Wireless Engineer</em>, 7, 536-541.<br>
Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine calculation of complex Fourier series. <em>Math. Computation</em>, 19, 297-301.<br>
Hanks, T.C. & Kanamori, H. (1979). A moment magnitude scale. <em>JGR</em>, 84(B5), 2348-2350.<br>
Kennett, B.L.N. & Engdahl, E.R. (1991). Traveltimes for global earthquake location (IASP91). <em>GJI</em>, 105, 429-465.<br>
Richter, C.F. (1935). An instrumental earthquake magnitude scale. <em>BSSA</em>, 25(1), 1-32.<br>
Wadati, K. (1933). On the travel time of earthquake waves. <em>Geophys. Mag.</em>, 7, 101-111.<br>
Welch, P.D. (1967). The use of Fast Fourier Transform for the estimation of power spectra. <em>IEEE TAES</em>, 15, 70-73.
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
        "en": "What to do with the exported data",
        "it": "Cosa fare con i dati esportati",
        "fr": "Que faire avec les donnees exportees",
        "es": "Que hacer con los datos exportados",
        "de": "Was mit den exportierten Daten tun",
    }
    st.markdown(f"<div class='section-title'>{_next_steps[lang]}</div>", unsafe_allow_html=True)

    _py_snippets_title = {
        "en": "Python code examples",
        "it": "Esempi di codice Python",
        "fr": "Exemples de code Python",
        "es": "Ejemplos de codigo Python",
        "de": "Python-Codebeispiele",
    }
    with st.expander(_py_snippets_title[lang]):
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

