# SeismicLens — Deutsche Dokumentation

[Italiano](README_IT.md) · [English](README_EN.md) · [Français](README_FR.md) · [Español](README_ES.md) · **Deutsch** · [Index](../README.md)

---

## Überblick

SeismicLens ist ein interaktives Geophysik-Werkzeug zum Laden, Filtern und Analysieren seismischer Wellenformen direkt im Browser. Es unterstützt physikalisch realistische synthetische Signalgenerierung, echte MiniSEED-Daten von IRIS/INGV, Spektralanalyse per FFT, automatisches P-Wellen-Picking mit dem STA/LTA-Algorithmus und CSV-Export aller Ergebnisse.

---

## Funktionen

| Funktion | Details |
|---|---|
| Synthetischer Generator | Physikalisch realistische P-, S- und Oberflächenwellen mit IASP91-artigem Schichtmodell |
| MiniSEED-Upload | Daten direkt von IRIS FDSN oder INGV-Webservices laden |
| CSV-Upload | Automatische Erkennung von 1-Spalten- (Amplitude) oder 2-Spalten-Format (Zeit, Amplitude) |
| Butterworth-Nullphasenfilter | Konfigurierbarer Bandpass, Ordnung 2–8, ohne Phasenverzerrung (`sosfiltfilt`) |
| FFT-Spektralanalyse | Amplitudenspektrum, Phasenspektrum, Dominanzfrequenz, Spektralzentroid, Bandbreite |
| Leistungsspektraldichte | Welch-PSD-Schätzung in counts²/Hz und dB |
| Spektrogramm (STFT) | Zeit-Frequenz-Karte mittels Kurzzeit-Fouriertransformation |
| STA/LTA P-Wellen-Picking | Klassischer Detektor nach Allen (1978), O(N) mit Präfixsummen, konfigurierbar |
| Krustales Geschwindigkeitsmodell | Interaktive Tabelle (Vp, Vs, Vp/Vs, Poisson-Zahl, Dichte) |
| Theorie und Mathematik | DFT/FFT mit komplexen Zahlen, Butterworth, Richter-Magnitude, Wadati-Methode |
| CSV-Export | Gefiltertes Signal, FFT-Spektrum (Amplitude + Phase), Welch-PSD, STA/LTA-Verhältnis |
| Mehrsprachige Oberfläche | Italiano, English, Français, Español, Deutsch |

---

## Seismologisches Modell

### Krustales Schichtgeschwindigkeitsmodell (IASP91-orientiert)

| Schicht | Tiefe (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Obere Kruste | 0–15 | 5.80 | 3.36 | 1.726 | 0.249 |
| Mittlere Kruste | 15–25 | 6.50 | 3.75 | 1.733 | 0.252 |
| Untere Kruste | 25–35 | 7.00 | 4.00 | 1.750 | 0.257 |
| Oberer Mantel | 35–200 | 8.04 | 4.47 | 1.798 | 0.272 |

### Physik der synthetischen Wellenform

```
P-Welle        :  f ~ 6–12 Hz,   Gaußscher Einhüllender,  Amplitude ∝ 10^(0.8M−2.5) × 0.15
S-Welle        :  f ~ 2–6 Hz,    Gaußscher Einhüllender,  Amplitude ∝ 10^(0.8M−2.5) × 0.65
Oberfl.-Wellen :  f ~ 0.3–1 Hz,  exponentieller Coda,     Amplitude ∝ 10^(0.8M−2.5) × 0.40
Rauschen       :  Butterworth-Bandpass 0.5–30 Hz, σ-normiert
```

Laufzeiten aus der hypozentralen Distanz `R = sqrt(d² + h²)`.

---

## Mathematische Grundlagen

### Diskrete Fouriertransformation (DFT)

```
X[k] = Summe_{n=0}^{N-1}  x[n] · exp(−j · 2π · k · n / N)

|X[k]|  →  Amplitude bei Frequenz  f_k = k · fs / N  (Hz)
∠X[k]   →  Phase = Argument der komplexen Zahl X[k]
           = atan2(Im(X[k]), Re(X[k]))

Einseitiges normiertes Spektrum:  A[k] = (2/N) · |X[k]|
```

Die FFT (Cooley–Tukey, 1965) reduziert O(N²) auf O(N log N).
Für reelle Signale ist das Spektrum hermitesch — `X[N−k] = konj(X[k])` — es gibt nur N/2+1 unabhängige Bins (`scipy.fft.rfft`).

### Komplexe Zahlen im Frequenzbereich

Jeder Koeffizient `X[k]` ist eine komplexe Zahl:

```
X[k] = Re(X[k]) + j · Im(X[k])
     = |X[k]| · exp(j · φ[k])          (Polarform)

|X[k]| = sqrt(Re² + Im²)               (Amplitude)
φ[k]   = atan2(Im(X[k]), Re(X[k]))     (Phase)
```

Eulers Formel `exp(jθ) = cos(θ) + j·sin(θ)` ist das mathematische Fundament:
Die DFT projiziert das Signal auf Kosinus- (Realteil) und Sinusbasis-funktionen (Imaginärteil).

### Butterworth-Nullphasenfilter

```
|H(jω)|² = 1 / [1 + (ω/ωc)^(2n)]

Abfall jenseits ωc:   20·n dB/Dekade (Einzeldurchlauf)
Mit sosfiltfilt (effektive Ordnung 2n):  40·n dB/Dekade
```

In SOS-Form (Zweite-Ordnung-Abschnitte) für numerische Stabilität implementiert.
`sosfiltfilt` führt den Filter vorwärts dann rückwärts aus: die Phasenantwort hebt sich exakt auf und bewahrt die Wellenankunftszeiten.

### STA/LTA

```
STA(t) = Mittelwert( x²[t−Nsta : t] )
LTA(t) = Mittelwert( x²[t−Nlta : t] )
R(t)   = STA(t) / LTA(t)   →  Auslösung wenn R > Schwelle
```

O(N)-Implementierung mit Präfixsummen: `cs = cumsum(x²)`.

### Wadati-Methode

```
Hypozentrale Distanz :  R = sqrt(d² + h²)
Laufzeiten :            t_P = R / Vp
                        t_S = R / Vs
Epizentrale Distanz :   d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)
```

---

## Schnellstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

http://localhost:8501 im Browser öffnen.

### Bedienung der Anwendung

1. Datenquelle in der Seitenleiste wählen: synthetisches Erdbeben, MiniSEED-Upload oder CSV.
2. Butterworth-Bandpassfilter konfigurieren (untere/obere Grenzfrequenz, Ordnung).
3. STA/LTA-Detektor einstellen (STA-Fenster, LTA-Fenster, Schwelle).
4. Tabs erkunden: Wellenform, Spektralanalyse, Spektrogramm, STA/LTA, Geschwindigkeitsmodell, Theorie & Mathematik.
5. Ergebnisse als CSV-Dateien exportieren.

### Echte MiniSEED-Daten beziehen

- IRIS FDSN: https://ds.iris.edu/wilber3/find_event
- INGV: https://webservices.ingv.it/swagger-ui/index.html
  Exportformat: MiniSEED

---

## Technologie-Stack

| Bibliothek | Rolle |
|---|---|
| ObsPy | MiniSEED-I/O, Detrendierung |
| SciPy | Butterworth-Filter (SOS), STFT, Welch-PSD |
| NumPy | Signalarrays, FFT |
| Plotly | Interaktive Diagramme |
| Streamlit | Web-Oberfläche |
| Pandas | CSV-I/O, Datenvorschau |

---

## Projektstruktur

```
seismiclens/
├── app.py                     # Haupt-Streamlit-App (UI + Orchestrierung)
├── requirements.txt
├── docs/
│   ├── README_EN.md
│   ├── README_IT.md
│   ├── README_FR.md
│   ├── README_ES.md
│   └── README_DE.md           # diese Datei
└── utils/
    ├── data_loader.py         # MiniSEED, CSV, synthetischer Generator
    └── signal_processing.py  # FFT, Filter, STA/LTA, PSD, Spektrogramm
```

