# SeismicLens — English Documentation

**[Italiano](README_IT.md) · English · [Français](README_FR.md) · [Español](README_ES.md) · [Deutsch](README_DE.md) · [Index](../README.md)**

---

## Overview

SeismicLens is an interactive geophysics web application for loading, filtering, and analysing seismic waveforms in the browser. It supports physically realistic synthetic signal generation, real MiniSEED data from IRIS/INGV, spectral analysis via FFT, automatic P-wave picking with the STA/LTA algorithm, and CSV export of all results.

---

## Features

| Feature | Details |
|---|---|
| Synthetic waveform generator | Physically realistic P, S and surface waves using an IASP91-like layered crustal velocity model |
| Real MiniSEED upload | Load waveforms directly from IRIS FDSN or INGV webservices |
| CSV upload | Auto-detects single-column (amplitude) or two-column (time, amplitude) formats |
| Zero-phase Butterworth filter | Configurable bandpass, order 2–8, no phase distortion (`sosfiltfilt`) |
| FFT spectral analysis | Amplitude spectrum, phase spectrum, dominant frequency, spectral centroid, bandwidth |
| Power Spectral Density | Welch PSD estimate in counts²/Hz and dB |
| Spectrogram (STFT) | Time–frequency heatmap using Short-Time Fourier Transform |
| STA/LTA P-wave picker | Classic Allen (1978) detector, O(N) via prefix sums, configurable windows and threshold |
| Crustal velocity model | Interactive table (Vp, Vs, Vp/Vs, Poisson ratio, density) with bar chart |
| Theory and math panel | DFT/FFT with complex numbers, Butterworth design, Richter magnitude, Wadati method |
| CSV export | Filtered signal, FFT spectrum (amplitude + phase), Welch PSD, STA/LTA ratio |
| Multilingual interface | English, Italiano, Français, Español, Deutsch |

---

## Seismological Model

### Layered Crustal Velocity (IASP91-inspired)

| Layer | Depth (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Upper crust | 0–15 | 5.80 | 3.36 | 1.726 | 0.249 |
| Middle crust | 15–25 | 6.50 | 3.75 | 1.733 | 0.252 |
| Lower crust | 25–35 | 7.00 | 4.00 | 1.750 | 0.257 |
| Upper mantle | 35–200 | 8.04 | 4.47 | 1.798 | 0.272 |

### Synthetic Waveform Physics

```
P-wave   :  f ~ 6–12 Hz,   Gaussian envelope,    amplitude ∝ 10^(0.8M−2.5) × 0.15
S-wave   :  f ~ 2–6 Hz,    Gaussian envelope,    amplitude ∝ 10^(0.8M−2.5) × 0.65
Surface  :  f ~ 0.3–1 Hz,  exponential coda,     amplitude ∝ 10^(0.8M−2.5) × 0.40
Noise    :  Butterworth bandpass 0.5–30 Hz, σ-normalised
```

Travel times from hypocentral distance `R = sqrt(d² + h²)`.

---

## Mathematical Background

### Discrete Fourier Transform (DFT)

```
X[k] = sum_{n=0}^{N-1}  x[n] · exp(−j · 2π · k · n / N)

|X[k]|  →  amplitude at frequency  f_k = k · fs / N  (Hz)
∠X[k]   →  phase = argument of the complex number X[k]
           = atan2(Im(X[k]), Re(X[k]))

One-sided normalised spectrum:  A[k] = (2/N) · |X[k]|
```

The FFT (Cooley–Tukey, 1965) reduces O(N²) to O(N log N).
For real signals the spectrum is Hermitian — `X[N−k] = conj(X[k])` — so only N/2+1 unique bins exist (`scipy.fft.rfft`).

### Complex Numbers in the Frequency Domain

Each coefficient `X[k]` is a complex number:

```
X[k] = Re(X[k]) + j · Im(X[k])
     = |X[k]| · exp(j · φ[k])          (polar form)

|X[k]| = sqrt(Re² + Im²)               (amplitude)
φ[k]   = atan2(Im(X[k]), Re(X[k]))     (phase)
```

Euler's formula `exp(jθ) = cos(θ) + j·sin(θ)` is the mathematical foundation:
the DFT projects the signal onto cosine (real part) and sine (imaginary part) basis functions.

### Zero-phase Butterworth Filter

```
|H(jω)|² = 1 / [1 + (ω/ωc)^(2n)]

Roll-off beyond ωc:   20·n dB/decade (single pass)
With sosfiltfilt (effective order 2n):  40·n dB/decade
```

Implemented in SOS (Second-Order Sections) form for numerical stability.
`sosfiltfilt` runs forward then backward: the phase response cancels exactly, preserving wave arrival times.

### STA/LTA

```
STA(t) = mean( x²[t−Nsta : t] )
LTA(t) = mean( x²[t−Nlta : t] )
R(t)   = STA(t) / LTA(t)   →  trigger when R > threshold
```

O(N) implementation via prefix sums: `cs = cumsum(x²)`.

### Wadati Method

```
Hypocentral distance:  R = sqrt(d² + h²)
Travel times:          t_P = R / Vp
                       t_S = R / Vs
Epicentral distance:   d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)
```

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### How to use the app

1. Choose a data source in the sidebar: synthetic earthquake, MiniSEED upload, or CSV upload.
2. Configure the Butterworth bandpass filter (low cut, high cut, order).
3. Adjust the STA/LTA detector (STA window, LTA window, threshold).
4. Explore the tabs: Waveform, Spectral Analysis, Spectrogram, STA/LTA, Velocity Model, Theory & Math.
5. Export results as CSV files.

### Getting real MiniSEED data

- IRIS FDSN: https://ds.iris.edu/wilber3/find_event
- INGV: https://webservices.ingv.it/swagger-ui/index.html
  Export format: MiniSEED

---

## Stack

| Library | Role |
|---|---|
| ObsPy | MiniSEED I/O, detrending |
| SciPy | Butterworth filter (SOS), STFT, Welch PSD |
| NumPy | Signal arrays, FFT |
| Plotly | Interactive charts |
| Streamlit | Web UI |
| Pandas | CSV I/O, data preview |

---

## Project Structure

```
seismiclens/
├── app.py                     # Main Streamlit app (UI + orchestration)
├── requirements.txt
├── docs/
│   ├── README_EN.md           # this file
│   ├── README_IT.md
│   ├── README_FR.md
│   ├── README_ES.md
│   └── README_DE.md
└── utils/
    ├── data_loader.py         # MiniSEED, CSV, synthetic generator
    └── signal_processing.py  # FFT, filter, STA/LTA, PSD, spectrogram
```

