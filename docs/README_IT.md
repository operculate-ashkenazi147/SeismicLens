# SeismicLens — Documentazione italiana

[English](/Users/daniaciampalini/IdeaProjects/seismiclens/README.md) · **Italiano** · [Français](README_FR.md) · [Español](README_ES.md) · [Deutsch](README_DE.md) 

---

## Panoramica

SeismicLens è uno strumento interattivo di geofisica per caricare, filtrare e analizzare forme d'onda sismiche direttamente nel browser. Supporta la generazione di segnali sintetici fisicamente realistici, dati reali MiniSEED da IRIS/INGV, analisi spettrale tramite FFT, picking automatico dell'onda P con l'algoritmo STA/LTA ed esportazione CSV di tutti i risultati.

---

## Funzionalità

| Funzione | Dettagli |
|---|---|
| Generatore sintetico | Onde P, S e superficiali con modello crostale a strati IASP91 |
| Upload MiniSEED | Dati reali da IRIS FDSN o INGV webservices |
| Upload CSV | Formato 1 colonna (ampiezza) o 2 colonne (tempo, ampiezza), rilevamento automatico |
| Filtro Butterworth zero-phase | Bandpass configurabile, ordine 2–8, nessuna distorsione di fase (`sosfiltfilt`) |
| Analisi spettrale FFT | Spettro di ampiezza, spettro di fase, frequenza dominante, centroide, bandwidth |
| Densità spettrale di potenza | Stima Welch in counts²/Hz e dB |
| Spettrogramma (STFT) | Mappa tempo-frequenza tramite Short-Time Fourier Transform |
| Picking P-wave STA/LTA | Detector classico Allen (1978), O(N) con prefix sums, finestre e soglia configurabili |
| Modello di velocità crostale | Tabella interattiva (Vp, Vs, Vp/Vs, coefficiente di Poisson, densità) |
| Pannello teoria e matematica | DFT/FFT con numeri complessi, Butterworth, scala Richter, metodo di Wadati |
| Esportazione CSV | Segnale filtrato, spettro FFT (ampiezza + fase), PSD Welch, rapporto STA/LTA |
| Interfaccia multilingua | Italiano, English, Français, Español, Deutsch |

---

## Modello sismologico

### Modello di velocità crostale a strati (ispirato a IASP91)

| Strato | Profondità (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Crosta superiore | 0–15 | 5.80 | 3.36 | 1.726 | 0.249 |
| Crosta media | 15–25 | 6.50 | 3.75 | 1.733 | 0.252 |
| Crosta inferiore | 25–35 | 7.00 | 4.00 | 1.750 | 0.257 |
| Mantello superiore | 35–200 | 8.04 | 4.47 | 1.798 | 0.272 |

### Fisica della forma d'onda sintetica

```
Onda P        :  f ~ 6–12 Hz,   inviluppo gaussiano,  ampiezza ∝ 10^(0.8M−2.5) × 0.15
Onda S        :  f ~ 2–6 Hz,    inviluppo gaussiano,  ampiezza ∝ 10^(0.8M−2.5) × 0.65
Onde superf.  :  f ~ 0.3–1 Hz,  coda esponenziale,    ampiezza ∝ 10^(0.8M−2.5) × 0.40
Rumore        :  Butterworth bandpass 0.5–30 Hz, normalizzato a σ
```

Tempi di percorrenza dalla distanza ipocentrale `R = sqrt(d² + h²)`.

---

## Basi matematiche

### Trasformata di Fourier Discreta (DFT)

```
X[k] = somma_{n=0}^{N-1}  x[n] · exp(−j · 2π · k · n / N)

|X[k]|  →  ampiezza alla frequenza  f_k = k · fs / N  (Hz)
∠X[k]   →  fase = argomento del numero complesso X[k]
           = atan2(Im(X[k]), Re(X[k]))

Spettro monolaterale normalizzato:  A[k] = (2/N) · |X[k]|
```

La FFT (Cooley–Tukey, 1965) riduce la complessità da O(N²) a O(N log N).
Per segnali reali lo spettro è hermitiano — `X[N−k] = conj(X[k])` — quindi esistono solo N/2+1 bin indipendenti (sfruttato da `scipy.fft.rfft`).

### Numeri complessi nel dominio della frequenza

Ogni coefficiente `X[k]` è un numero complesso:

```
X[k] = Re(X[k]) + j · Im(X[k])
     = |X[k]| · exp(j · φ[k])          (forma polare)

|X[k]| = sqrt(Re² + Im²)               (ampiezza)
φ[k]   = atan2(Im(X[k]), Re(X[k]))     (fase)
```

La formula di Eulero `exp(jθ) = cos(θ) + j·sin(θ)` è il fondamento matematico:
la DFT proietta il segnale sulle basi coseno (parte reale) e seno (parte immaginaria).

### Filtro Butterworth zero-phase

```
|H(jω)|² = 1 / [1 + (ω/ωc)^(2n)]

Roll-off oltre ωc:   20·n dB/decade (passata singola)
Con sosfiltfilt (ordine effettivo 2n):  40·n dB/decade
```

Il filtro è implementato in forma SOS (Second-Order Sections) per stabilità numerica.
`sosfiltfilt` esegue la passata in avanti poi all'indietro: la risposta di fase si annulla esattamente, preservando i tempi di arrivo delle onde.

### STA/LTA

```
STA(t) = media( x²[t−Nsta : t] )
LTA(t) = media( x²[t−Nlta : t] )
R(t)   = STA(t) / LTA(t)   →  trigger quando R > soglia
```

Implementazione O(N) tramite prefix sums: `cs = cumsum(x²)`.

### Metodo di Wadati

```
Distanza ipocentrale:  R = sqrt(d² + h²)
Tempi di arrivo:       t_P = R / Vp
                       t_S = R / Vs
Distanza epicentrale:  d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)
```

---

## Avvio rapido

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aprire http://localhost:8501 nel browser.

### Come usare l'app

1. Scegli la sorgente dati nella barra laterale: terremoto sintetico, MiniSEED o CSV.
2. Configura il filtro Butterworth (taglio basso, taglio alto, ordine).
3. Regola il detector STA/LTA (finestra STA, finestra LTA, soglia).
4. Esplora le schede: Forma d'onda, Analisi spettrale, Spettrogramma, STA/LTA, Modello di velocità, Teoria.
5. Esporta i risultati come CSV.

### Ottenere dati MiniSEED reali

- IRIS FDSN: https://ds.iris.edu/wilber3/find_event
- INGV: https://webservices.ingv.it/swagger-ui/index.html
  Formato di export: MiniSEED

---

## Stack tecnologico

| Libreria | Ruolo |
|---|---|
| ObsPy | Lettura MiniSEED, detrendizzazione |
| SciPy | Filtro Butterworth (SOS), STFT, Welch PSD |
| NumPy | Array di segnale, FFT |
| Plotly | Grafici interattivi |
| Streamlit | Interfaccia web |
| Pandas | CSV I/O, anteprima dati |

---

## Struttura del progetto

```
seismiclens/
├── app.py                     # App principale Streamlit (UI + orchestrazione)
├── requirements.txt
├── docs/
│   ├── README_EN.md
│   ├── README_IT.md           # questo file
│   ├── README_FR.md
│   ├── README_ES.md
│   └── README_DE.md
└── utils/
    ├── data_loader.py         # MiniSEED, CSV, generatore sintetico
    └── signal_processing.py  # FFT, filtro, STA/LTA, PSD, spettrogramma
```

