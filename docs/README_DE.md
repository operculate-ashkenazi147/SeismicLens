<div align="center">

# 🌍 SeismicLens

**Interaktiver Seismischer Wellenform-Analysator**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![ObsPy](https://img.shields.io/badge/ObsPy-1.4%2B-green)](https://docs.obspy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.13%2B-8CAAE6?logo=scipy)](https://scipy.org/)

*Seismische Wellenformen direkt im Browser laden, filtern und analysieren — keine Installation außer Python erforderlich.*

[🇬🇧 English](../README.md) · [🇮🇹 Italiano](README_IT.md) · [🇫🇷 Français](README_FR.md) · [🇪🇸 Español](README_ES.md) · **🇩🇪 Deutsch**

</div>

---

## 📋 Inhaltsverzeichnis

- [Überblick](#-überblick)
- [Funktionen](#-funktionen)
- [Schnellstart](#-schnellstart)
- [Bedienungsanleitung](#-bedienungsanleitung)
- [Echte Daten beziehen](#️-echte-daten-beziehen)
- [Seismologisches Modell](#-seismologisches-modell)
- [Mathematische Grundlagen](#-mathematische-grundlagen)
- [Technologie-Stack](#️-technologie-stack)
- [Projektstruktur](#-projektstruktur)
- [Roadmap](#️-roadmap)
- [Mitwirken](#-mitwirken)
- [Lizenz](#-lizenz)

---

## 🔭 Überblick

**SeismicLens** ist ein quelloffenes, browserbasiertes Geophysik-Werkzeug, entwickelt mit [Streamlit](https://streamlit.io/). Es ermöglicht Studierenden, Forschenden und Erdbeben-Enthusiasten, seismische Signale zu erkunden — ohne eine einzige Zeile Code schreiben zu müssen.

**Was du tun kannst:**

- **Physikalisch realistische synthetische Seismogramme** mit einem IASP91-inspirierten geschichteten Krustengeschwindigkeitsmodell generieren
- **Echte MiniSEED-Wellenformen** von globalen Netzwerken (IRIS, INGV, GEOFON, …) hochladen und analysieren
- Einen **Nullphasen-Butterworth-Bandpassfilter** anwenden, um interessante Frequenzbänder zu isolieren
- P-Wellen-Einsetzen automatisch mit dem klassischen **STA/LTA-Algorithmus** (Allen, 1978) detektieren
- Das Signal per **FFT** zerlegen und Amplitudenspektrum, Phasenspektrum und Welch-PSD visualisieren
- Die Zeit-Frequenz-Entwicklung mit einem **interaktiven Spektrogramm (STFT)** untersuchen
- Alle verarbeiteten Daten als **CSV** für Weiteranalysen in Python, MATLAB oder Excel **exportieren**
- Zwischen **5 Sprachen** (EN / IT / FR / ES / DE) und **Hell- / Dunkelthema** wechseln

---

## ✨ Funktionen

| Funktion | Details |
|---|---|
|  Synthetischer Wellenformgenerator | P-, S- und Oberflächenwellen · IASP91-ähnliches Schichtmodell · konfigurierbares M, Tiefe, Distanz, Rauschen, Dauer |
|  MiniSEED-Upload | Reale Daten von IRIS FDSN oder INGV-Webservices |
|  CSV-Upload | Automatische Erkennung von 1-Spalten- (Amplitude) oder 2-Spalten-Format (Zeit, Amplitude) |
| ️ Nullphasen-Butterworth-Filter | Konfigurierbarer Bandpass · Ordnung 2–8 · keine Phasenverzerrung (`sosfiltfilt`) |
|  FFT-Spektralanalyse | Amplitudenspektrum · Phasenspektrum · Dominanzfrequenz · Spektralzentroid · RMS-Bandbreite |
|  Leistungsspektraldichte | Welch-PSD-Schätzung in counts²/Hz und dB re 1 count²/Hz |
| ️ Spektrogramm (STFT) | Interaktive Zeit-Frequenz-Heatmap · Inferno/Viridis-Farbskala |
|  STA/LTA P-Wellen-Picker | Klassischer Allen-Detektor (1978) · O(N) mit Präfixsummen · konfigurierbare Fenster & Schwelle |
|  Krustengeschwindigkeitsmodell | Interaktive Tabelle (Vp, Vs, Vp/Vs, Poisson-Zahl, Dichte) + horizontales Balkendiagramm |
|  Theorie & Mathematik | DFT/FFT mit komplexen Zahlen · Butterworth-Design · Richter- & Mw-Magnitude · Wadati-Methode |
|  CSV-Export | Gefiltertes Signal · FFT-Spektrum · Welch-PSD · STA/LTA-Verhältnis |
|  Mehrsprachige Oberfläche | English · Italiano · Français · Español · Deutsch |
|  Hell- / Dunkelthema | Umschalten in der Seitenleiste · themensensitive Plotly-Diagramme |

---

## 🚀 Schnellstart

### Voraussetzungen

- Python **3.10 oder höher**
- `pip` (wird mit Python mitgeliefert)

### Installation

```bash
# 1. Repository klonen
git clone https://github.com/your-username/seismiclens.git
cd seismiclens

# 2. (Empfohlen) Virtuelle Umgebung erstellen und aktivieren
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. App starten
streamlit run app.py
```

Die App öffnet sich automatisch unter **http://localhost:8501**.

> **Tipp:** Beim ersten Start benötigt ObsPy einige Sekunden zur Initialisierung. Nachfolgende Starts sind durch Streamlits Modul-Caching schneller.

---

## 📖 Bedienungsanleitung

### 1 — Datenquelle auswählen

Die **Seitenleiste** öffnen (☰ wenn eingeklappt) und einen der drei Modi auswählen:

| Modus | Wann benutzen |
|---|---|
| **Synthetisches Erdbeben** | Sofortige Erkundung — keine Datei erforderlich. Magnitude (M 2–8), Herdtiefe (5–200 km), epizentrale Distanz (10–500 km), Rauschpegel und Dauer einstellen. |
| **MiniSEED hochladen** | Echte Breitband-Seismogramme von globalen Netzwerken. `.mseed`-Dateien von IRIS oder INGV herunterladen (siehe [Echte Daten beziehen](#️-echte-daten-beziehen)). |
| **CSV hochladen** | Eigene Zeitreihendaten. Eine Spalte = Amplitudenproben; zwei Spalten = Zeit (s), Amplitude. |

### 2 — Butterworth-Filter konfigurieren

| Parameter | Typische Werte | Wirkung |
|---|---|---|
| Untere Grenzfrequenz (Hz) | 0,01–2 Hz (teleseismisch) · 0,5–5 Hz (regional) · 1–15 Hz (lokal) | Entfernt niederfrequente Drift und mikroseismisches Rauschen |
| Obere Grenzfrequenz (Hz) | Muss < Nyquist (fs/2) sein | Entfernt hochfrequentes Kulturlärm |
| Ordnung | 2–8 (Standard 4) | Höhere Ordnung → steilerer Abfall, mehr Klingeln |

**Nullphasen-Bandpass** ein-/ausschalten, um das gefilterte Signal mit der Rohwellenform zu vergleichen.

### 3 — STA/LTA-Detektor einstellen

```
Faustformel:  LTA >= 10 x STA
Typische Schwelle:  3 - 5
```

| Parameter | Beschreibung |
|---|---|
| STA-Fenster (s) | Kurzzeit-Mittelwert: erfasst die impulsive Einsatzenergie (0,2–2 s) |
| LTA-Fenster (s) | Langzeit-Mittelwert: verfolgt den Hintergrundrauschpegel (5–60 s) |
| Auslöseschwelle | Verhältnis R, oberhalb dessen eine seismische Phase erklärt wird |

Schwelle senken, um schwache Ereignisse zu erfassen; erhöhen, um Falschmeldungen bei verrauschten Daten zu unterdrücken.

### 4 — Analysetabs erkunden

| Tab | Inhalt |
|---|---|
| **Wellenform** | Zeitbereichsplot · P- und S-Ankunftsmarker · Roh-Signal-Overlay umschaltbar |
| **Spektralanalyse** | FFT-Amplitudenspektrum · optionales Phasenspektrum · optionale Welch-PSD |
| **Spektrogramm** | STFT Zeit-Frequenz-Heatmap |
| **STA/LTA** | STA/LTA-Kennfunktion · hervorgehobene Auslösefenster |
| **Geschwindigkeitsmodell** | IASP91-inspirierte Tabelle + Vp/Vs-Balkendiagramm |
| **Theorie & Mathematik** | DFT, Butterworth, PSD, STFT, Wellenphysik, Magnitudenskalen — mit Gleichungen |
| **Export** | CSV-Dateien für jede berechnete Größe herunterladen |

### 5 — Ergebnisse exportieren

Alle verarbeiteten Daten können als CSV vom **Export**-Tab heruntergeladen werden:

| Datei | Inhalt |
|---|---|
| `signal.csv` | Zeit (s), gefilterte Amplitude (counts) |
| `fft.csv` | Frequenz (Hz), Amplitude, Phase (Grad) |
| `psd.csv` | Frequenz (Hz), PSD (counts²/Hz) |
| `stalta.csv` | Zeit (s), STA/LTA-Verhältnis |

---

## 🛰️ Echte Daten beziehen

### IRIS (Incorporated Research Institutions for Seismology)

1. Zu **[IRIS Wilber 3](https://ds.iris.edu/wilber3/find_event)** gehen
2. Nach einem Erdbeben nach Datum, Region oder Magnitude suchen
3. Eine seismische Station in der Nähe des Ereignisses auswählen
4. **MiniSEED** als Exportformat wählen und herunterladen

### INGV (Istituto Nazionale di Geofisica e Vulcanologia — Italien)

1. Den **[INGV FDSN-Webservice](https://webservices.ingv.it/swagger-ui/index.html)** öffnen
2. Den `/dataselect`-Endpunkt mit Netz-, Stations- und Zeitparametern nutzen
3. Format auf `miniseed` setzen und herunterladen

### Weitere Netzwerke

| Netzwerk | URL |
|---|---|
| GEOFON (GFZ Potsdam) | https://geofon.gfz-potsdam.de/waveform/ |
| EMSC | https://www.seismicportal.eu/ |
| ORFEUS | https://www.orfeus-eu.org/data/eida/ |

---

## 🌐 Seismologisches Modell

### Geschichtetes Krustengeschwindigkeitsmodell (IASP91-inspiriert)

| Schicht | Tiefe (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Obere Kruste | 0–15 | 5,80 | 3,36 | 1,726 | 0,249 |
| Mittlere Kruste | 15–25 | 6,50 | 3,75 | 1,733 | 0,252 |
| Untere Kruste | 25–35 | 7,00 | 4,00 | 1,750 | 0,257 |
| Oberer Mantel | 35–200 | 8,04 | 4,47 | 1,798 | 0,272 |

### Physik der synthetischen Wellenform

```
P-Welle        :  f ~ 6-12 Hz,   Gaußscher Einhüllender,  Amplitude ~ 10^(0.8M-2.5) x 0.15
S-Welle        :  f ~ 2-6 Hz,    Gaußscher Einhüllender,  Amplitude ~ 10^(0.8M-2.5) x 0.65
Oberflächenwellen :  f ~ 0.3-1 Hz,  exponentieller Coda,  Amplitude ~ 10^(0.8M-2.5) x 0.40
Rauschen       :  Butterworth-Bandpass 0.5-30 Hz, sigma-normiert
```

Laufzeiten aus der hypozentralen Distanz `R = sqrt(d^2 + h^2)`.

---

## 📐 Mathematische Grundlagen

### Diskrete Fouriertransformation (DFT)

```
X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2pi * k * n / N)

|X[k]|  ->  Amplitude bei  f_k = k * fs / N  Hz
winkel(X[k])  ->  Phase = atan2(Im(X[k]), Re(X[k]))

Einseitiges normiertes Spektrum:  A[k] = (2/N) * |X[k]|
```

Die FFT (Cooley–Tukey, 1965) reduziert die Komplexität von O(N²) auf **O(N log N)**.  
Für reelle Signale ist das Spektrum Hermitesch: `X[N-k] = konj(X[k])`, es gibt also nur N/2+1 eindeutige Bins (ausgenutzt durch `scipy.fft.rfft`).

### Nullphasen-Butterworth-Filter

```
|H(jw)|^2 = 1 / [1 + (w/wc)^(2n)]

Abfall Einzeldurchlauf:              20*n  dB/Dekade
sosfiltfilt (effektive Ordnung 2n):  40*n  dB/Dekade, keine Phasenverzerrung
```

In SOS-Form (Zweite-Ordnung-Abschnitte) für numerische Stabilität implementiert.

### STA/LTA (Allen, 1978)

```
STA(t) = (1/Nsta) * sum( x^2[t-Nsta : t] )
LTA(t) = (1/Nlta) * sum( x^2[t-Nlta : t] )
R(t)   = STA(t) / LTA(t)   ->  Auslösung wenn R > Schwelle
```

O(N)-Implementierung mit Präfixsummen: `cs = cumsum(x^2)`.

### Wadati-Methode

```
R = sqrt(d^2 + h^2)              (hypozentrale Distanz)
t_P = R / Vp  ,  t_S = R / Vs
d ~ (t_S - t_P) * Vp * Vs / (Vp - Vs)
```

---

## 🛠️ Technologie-Stack

| Bibliothek | Version | Rolle |
|---|---|---|
| [Streamlit](https://streamlit.io/) | >= 1.35 | Web-Oberfläche, reaktives Zustandsmanagement |
| [ObsPy](https://docs.obspy.org/) | >= 1.4 | MiniSEED-I/O, Trace-Detrendierung |
| [SciPy](https://scipy.org/) | >= 1.13 | Butterworth-Filter (SOS), STFT, Welch-PSD |
| [NumPy](https://numpy.org/) | >= 1.26 | Signalarrays, FFT (rfft) |
| [Plotly](https://plotly.com/python/) | >= 5.22 | Interaktive Diagramme |
| [Pandas](https://pandas.pydata.org/) | >= 2.2 | CSV-I/O, Datenvorschau |

---

## 📁 Projektstruktur

```
seismiclens/
├── app.py                     # Haupt-Streamlit-App (UI + Orchestrierung)
├── requirements.txt           # Python-Abhängigkeiten
├── LICENSE                    # MIT-Lizenz
├── README.md                  # Englische Hauptdokumentation
├── docs/
│   ├── README_IT.md           # Documentazione italiana
│   ├── README_FR.md           # Documentation française
│   ├── README_ES.md           # Documentación española
│   └── README_DE.md           # Diese Datei
└── utils/
    ├── __init__.py
    ├── data_loader.py         # MiniSEED, CSV, synthetischer Wellenformgenerator
    └── signal_processing.py  # FFT, Butterworth-Filter, STA/LTA, PSD, Spektrogramm
```

---

## 🗺️ Roadmap

Die folgenden Funktionen sind für zukünftige Versionen geplant. Beiträge sind willkommen!

### v2.1 — Daten & I/O
- [ ] **FDSN-Client-Integration** — IRIS / INGV direkt aus der App abfragen ohne manuellen Download
- [ ] **SAC-Dateiunterstützung** — SAC-Format-Seismogramme laden, weit verbreitet in der akademischen Seismologie
- [ ] **Multi-Spur-Anzeige** — Dreikomponenten-Aufzeichnungen (Z, N, E) gleichzeitig plotten und vergleichen

### v2.2 — Signalverarbeitung
- [ ] **Instrumentenantwort-Entfernung** — RESP / StationXML-Dateien dekonvolvieren für Bodenverschiebung/-geschwindigkeit/-beschleunigung
- [ ] **Adaptives STA/LTA** — rekursives STA/LTA mit automatischer Schwellenoptimierung
- [ ] **Partikelbewegungsanalyse** — Hodogramm-Plots für Z-N-E-Komponentendaten
- [ ] **Coda-Q-Schätzung** — Streuungsabschwächung aus dem Coda-Einhüllenden-Abfall

### v2.3 — Erweiterte Analyse
- [ ] **Momenttensor-Visualisierung** — Strandball-Diagramme aus Herdmechanismus-Parametern
- [ ] **Array-Verarbeitung** — Beamforming und Langsamkeitsanalyse für seismische Arrays
- [ ] **Machine-Learning-Phasenpicker** — PhaseNet oder EQTransformer für automatisierte Phasendetektion integrieren
- [ ] **Magnitudenabschätzung** — lokale Magnitude (Ml) aus Spitzenamplitude und Stationskorrekturen

### v3.0 — Plattform
- [ ] **Echtzeit-Streaming** — Verbindung zu SeedLink-Servern für Live-Wellenformüberwachung
- [ ] **Benutzer-Session-Persistenz** — Analysekonfigurationen speichern und neu laden
- [ ] **REST-API** — Verarbeitungsfunktionen als Endpunkte für programmatischen Zugriff bereitstellen
- [ ] **Docker-Image** — Ein-Befehl-Bereitstellung mit `docker run`

---

## 🤝 Mitwirken

Beiträge, Fehlerberichte und Feature-Anfragen sind herzlich willkommen!

1. Repository **forken**
2. Feature-Branch erstellen: `git checkout -b feature/dein-feature-name`
3. Änderungen nach [Conventional Commits](https://www.conventionalcommits.org/) committen: `git commit -m "feat: feature hinzufügen"`
4. Branch pushen: `git push origin feature/dein-feature-name`
5. **Pull Request** gegen `main` öffnen

### Code-Stil

- **PEP 8** für Python-Code befolgen
- Funktionen fokussiert und mit Docstrings dokumentiert halten
- Tests hinzufügen oder aktualisieren bei neuer Signalverarbeitungslogik
- `python -m py_compile app.py utils/*.py` vor dem PR ausführen

### Fehler melden

Bitte ein GitHub-Issue öffnen und folgendes angeben:
- Python-Version und Betriebssystem
- Schritte zur Reproduktion
- Erwartetes vs. tatsächliches Verhalten
- Nach Möglichkeit eine minimale Beispieldatei (MiniSEED oder CSV), die den Fehler auslöst

---

## 📄 Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert — siehe die Datei [LICENSE](../LICENSE) für Details.

```
MIT License  —  Copyright (c) 2026 Dania Ciampalini & Dario Ciampalini

Hiermit wird unentgeltlich jeder Person, die eine Kopie der Software und der
zugehörigen Dokumentationsdateien (die "Software") erhält, die Erlaubnis erteilt,
die Software ohne Einschränkung zu benutzen, einschließlich und ohne Einschränkung
der Rechte, die Software zu verwenden, zu kopieren, zu ändern, zusammenzuführen,
zu veröffentlichen, zu vertreiben, zu unterlizenzieren und/oder zu verkaufen,
und Personen, denen die Software überlassen wird, dies ebenfalls zu erlauben,
unter den folgenden Bedingungen:

Der obige Urheberrechtshinweis und dieser Genehmigungshinweis müssen in allen
Kopien oder wesentlichen Teilen der Software enthalten sein.

DIE SOFTWARE WIRD OHNE MÄNGELGEWÄHR UND OHNE JEGLICHE AUSDRÜCKLICHE ODER
STILLSCHWEIGENDE GARANTIE, EINSCHLIESSLICH, ABER NICHT BEGRENZT AUF DIE
GARANTIEN DER MARKTGÄNGIGKEIT, DER EIGNUNG FÜR EINEN BESTIMMTEN ZWECK UND DER
NICHTVERLETZUNG VON RECHTEN DRITTER, ZUR VERFÜGUNG GESTELLT.
```
