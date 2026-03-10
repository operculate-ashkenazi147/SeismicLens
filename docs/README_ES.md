# SeismicLens — Documentación en español

[Italiano](README_IT.md) · [English](README_EN.md) · [Français](README_FR.md) · **Español** · [Deutsch](README_DE.md) · [Índice](../README.md)

---

## Descripción general

SeismicLens es una herramienta interactiva de geofísica para cargar, filtrar y analizar formas de onda sísmicas directamente en el navegador. Soporta la generación de señales sintéticas físicamente realistas, datos MiniSEED reales de IRIS/INGV, análisis espectral mediante FFT, detección automática de la onda P con el algoritmo STA/LTA y exportación CSV de todos los resultados.

---

## Funcionalidades

| Función | Detalles |
|---|---|
| Generador sintético | Ondas P, S y superficiales físicamente realistas con modelo cortical en capas IASP91 |
| Carga MiniSEED | Datos reales desde IRIS FDSN o INGV webservices |
| Carga CSV | Detección automática de formato 1 columna (amplitud) o 2 columnas (tiempo, amplitud) |
| Filtro Butterworth fase cero | Pasa-banda configurable, orden 2–8, sin distorsión de fase (`sosfiltfilt`) |
| Análisis espectral FFT | Espectro de amplitud, espectro de fase, frecuencia dominante, centroide espectral, ancho de banda |
| Densidad espectral de potencia | Estimación Welch en counts²/Hz y dB |
| Espectrograma (STFT) | Mapa tiempo-frecuencia mediante transformada de Fourier de tiempo corto |
| Detección STA/LTA de onda P | Detector clásico Allen (1978), O(N) con sumas prefijas, ventanas y umbral configurables |
| Modelo de velocidad cortical | Tabla interactiva (Vp, Vs, Vp/Vs, coeficiente de Poisson, densidad) |
| Teoría y matemáticas | DFT/FFT con números complejos, Butterworth, escala de Richter, método de Wadati |
| Exportación CSV | Señal filtrada, espectro FFT (amplitud + fase), PSD Welch, razón STA/LTA |
| Interfaz multilingüe | Italiano, English, Français, Español, Deutsch |

---

## Modelo sismológico

### Modelo de velocidad cortical en capas (inspirado en IASP91)

| Capa | Profundidad (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Corteza superior | 0–15 | 5.80 | 3.36 | 1.726 | 0.249 |
| Corteza media | 15–25 | 6.50 | 3.75 | 1.733 | 0.252 |
| Corteza inferior | 25–35 | 7.00 | 4.00 | 1.750 | 0.257 |
| Manto superior | 35–200 | 8.04 | 4.47 | 1.798 | 0.272 |

### Física de la forma de onda sintética

```
Onda P        :  f ~ 6–12 Hz,   envolvente gaussiana,  amplitud ∝ 10^(0.8M−2.5) × 0.15
Onda S        :  f ~ 2–6 Hz,    envolvente gaussiana,  amplitud ∝ 10^(0.8M−2.5) × 0.65
Ondas superf. :  f ~ 0.3–1 Hz,  coda exponencial,      amplitud ∝ 10^(0.8M−2.5) × 0.40
Ruido         :  Butterworth pasa-banda 0.5–30 Hz, normalizado a σ
```

Tiempos de recorrido desde la distancia hipocentral `R = sqrt(d² + h²)`.

---

## Fundamentos matemáticos

### Transformada de Fourier Discreta (DFT)

```
X[k] = suma_{n=0}^{N-1}  x[n] · exp(−j · 2π · k · n / N)

|X[k]|  →  amplitud a la frecuencia  f_k = k · fs / N  (Hz)
∠X[k]   →  fase = argumento del número complejo X[k]
           = atan2(Im(X[k]), Re(X[k]))

Espectro unilateral normalizado:  A[k] = (2/N) · |X[k]|
```

La FFT (Cooley–Tukey, 1965) reduce la complejidad de O(N²) a O(N log N).
Para señales reales el espectro es hermítico — `X[N−k] = conj(X[k])` — por lo que solo existen N/2+1 bins independientes (`scipy.fft.rfft`).

### Números complejos en el dominio de la frecuencia

Cada coeficiente `X[k]` es un número complejo:

```
X[k] = Re(X[k]) + j · Im(X[k])
     = |X[k]| · exp(j · φ[k])          (forma polar)

|X[k]| = sqrt(Re² + Im²)               (amplitud)
φ[k]   = atan2(Im(X[k]), Re(X[k]))     (fase)
```

La fórmula de Euler `exp(jθ) = cos(θ) + j·sin(θ)` es el fundamento matemático:
la DFT proyecta la señal sobre las bases coseno (parte real) y seno (parte imaginaria).

### Filtro Butterworth de fase cero

```
|H(jω)|² = 1 / [1 + (ω/ωc)^(2n)]

Atenuación más allá de ωc:   20·n dB/década (pasada simple)
Con sosfiltfilt (orden efectivo 2n):  40·n dB/década
```

Implementado en forma SOS (Secciones de Segundo Orden) para estabilidad numérica.
`sosfiltfilt` aplica el filtro en sentido directo y luego inverso: la respuesta de fase se cancela exactamente, preservando los tiempos de llegada de las ondas.

### STA/LTA

```
STA(t) = media( x²[t−Nsta : t] )
LTA(t) = media( x²[t−Nlta : t] )
R(t)   = STA(t) / LTA(t)   →  disparo cuando R > umbral
```

Implementación O(N) mediante sumas prefijas: `cs = cumsum(x²)`.

### Método de Wadati

```
Distancia hipocentral :  R = sqrt(d² + h²)
Tiempos de llegada :     t_P = R / Vp
                         t_S = R / Vs
Distancia epicentral :   d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)
```

---

## Inicio rápido

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abrir http://localhost:8501 en el navegador.

### Cómo usar la aplicación

1. Elegir una fuente de datos en la barra lateral: sismo sintético, archivo MiniSEED o CSV.
2. Configurar el filtro Butterworth pasa-banda (corte bajo, corte alto, orden).
3. Ajustar el detector STA/LTA (ventana STA, ventana LTA, umbral).
4. Explorar las pestañas: Forma de onda, Análisis espectral, Espectrograma, STA/LTA, Modelo de velocidad, Teoría y Matemáticas.
5. Exportar los resultados como archivos CSV.

### Obtener datos MiniSEED reales

- IRIS FDSN: https://ds.iris.edu/wilber3/find_event
- INGV: https://webservices.ingv.it/swagger-ui/index.html
  Formato de exportación: MiniSEED

---

## Stack tecnológico

| Biblioteca | Rol |
|---|---|
| ObsPy | Lectura MiniSEED, detendencia |
| SciPy | Filtro Butterworth (SOS), STFT, Welch PSD |
| NumPy | Arrays de señal, FFT |
| Plotly | Gráficos interactivos |
| Streamlit | Interfaz web |
| Pandas | CSV I/O, vista previa de datos |

---

## Estructura del proyecto

```
seismiclens/
├── app.py                     # App principal Streamlit (UI + orquestación)
├── requirements.txt
├── docs/
│   ├── README_EN.md
│   ├── README_IT.md
│   ├── README_FR.md
│   ├── README_ES.md           # este archivo
│   └── README_DE.md
└── utils/
    ├── data_loader.py         # MiniSEED, CSV, generador sintético
    └── signal_processing.py  # FFT, filtro, STA/LTA, PSD, espectrograma
```

