# SeismicLens — Documentation française

[Italiano](README_IT.md) · [English](README_EN.md) · **Français** · [Español](README_ES.md) · [Deutsch](README_DE.md) · [Index](../README.md)

---

## Vue d'ensemble

SeismicLens est un outil interactif de géophysique permettant de charger, filtrer et analyser des formes d'onde sismiques directement dans le navigateur. Il prend en charge la génération de signaux synthétiques physiquement réalistes, les données MiniSEED réelles d'IRIS/INGV, l'analyse spectrale par FFT, le pointé automatique de l'onde P avec l'algorithme STA/LTA et l'export CSV de tous les résultats.

---

## Fonctionnalités

| Fonction | Détails |
|---|---|
| Générateur synthétique | Ondes P, S et de surface physiquement réalistes avec modèle crustal en couches IASP91 |
| Import MiniSEED | Chargement depuis IRIS FDSN ou INGV webservices |
| Import CSV | Détection automatique du format 1 colonne (amplitude) ou 2 colonnes (temps, amplitude) |
| Filtre Butterworth zéro-phase | Passe-bande configurable, ordre 2–8, sans distorsion de phase (`sosfiltfilt`) |
| Analyse spectrale FFT | Spectre d'amplitude, spectre de phase, fréquence dominante, centroïde spectral, largeur de bande |
| Densité spectrale de puissance | Estimation Welch en counts²/Hz et dB |
| Spectrogramme (STFT) | Carte temps-fréquence par transformée de Fourier à court terme |
| Pointé STA/LTA de l'onde P | Détecteur classique Allen (1978), O(N) par sommes préfixes, fenêtres et seuil configurables |
| Modèle de vitesse crustal | Tableau interactif (Vp, Vs, Vp/Vs, coeff. de Poisson, densité) |
| Théorie et mathématiques | DFT/FFT avec nombres complexes, Butterworth, échelle de Richter, méthode de Wadati |
| Export CSV | Signal filtré, spectre FFT (amplitude + phase), DSP Welch, rapport STA/LTA |
| Interface multilingue | Italiano, English, Français, Español, Deutsch |

---

## Modèle sismologique

### Modèle de vitesse crustal en couches (inspiré d'IASP91)

| Couche | Profondeur (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Croûte supérieure | 0–15 | 5.80 | 3.36 | 1.726 | 0.249 |
| Croûte moyenne | 15–25 | 6.50 | 3.75 | 1.733 | 0.252 |
| Croûte inférieure | 25–35 | 7.00 | 4.00 | 1.750 | 0.257 |
| Manteau supérieur | 35–200 | 8.04 | 4.47 | 1.798 | 0.272 |

### Physique de la forme d'onde synthétique

```
Onde P        :  f ~ 6–12 Hz,   enveloppe gaussienne,  amplitude ∝ 10^(0.8M−2.5) × 0.15
Onde S        :  f ~ 2–6 Hz,    enveloppe gaussienne,  amplitude ∝ 10^(0.8M−2.5) × 0.65
Ondes superf. :  f ~ 0.3–1 Hz,  coda exponentielle,    amplitude ∝ 10^(0.8M−2.5) × 0.40
Bruit         :  Butterworth passe-bande 0.5–30 Hz, normalisé à σ
```

Temps de parcours depuis la distance hypocentrale `R = sqrt(d² + h²)`.

---

## Bases mathématiques

### Transformée de Fourier Discrète (TFD)

```
X[k] = somme_{n=0}^{N-1}  x[n] · exp(−j · 2π · k · n / N)

|X[k]|  →  amplitude à la fréquence  f_k = k · fs / N  (Hz)
∠X[k]   →  phase = argument du nombre complexe X[k]
           = atan2(Im(X[k]), Re(X[k]))

Spectre monolateral normalisé :  A[k] = (2/N) · |X[k]|
```

La FFT (Cooley–Tukey, 1965) réduit la complexité de O(N²) à O(N log N).
Pour les signaux réels le spectre est hermitien — `X[N−k] = conj(X[k])` — il n'existe que N/2+1 bins indépendants (`scipy.fft.rfft`).

### Nombres complexes dans le domaine fréquentiel

Chaque coefficient `X[k]` est un nombre complexe :

```
X[k] = Re(X[k]) + j · Im(X[k])
     = |X[k]| · exp(j · φ[k])          (forme polaire)

|X[k]| = sqrt(Re² + Im²)               (amplitude)
φ[k]   = atan2(Im(X[k]), Re(X[k]))     (phase)
```

La formule d'Euler `exp(jθ) = cos(θ) + j·sin(θ)` est le fondement mathématique :
la TFD projette le signal sur les bases cosinus (partie réelle) et sinus (partie imaginaire).

### Filtre Butterworth zéro-phase

```
|H(jω)|² = 1 / [1 + (ω/ωc)^(2n)]

Atténuation au-delà de ωc :   20·n dB/décade (passe unique)
Avec sosfiltfilt (ordre effectif 2n) :  40·n dB/décade
```

Implémenté en forme SOS (Sections du Second Ordre) pour la stabilité numérique.
`sosfiltfilt` effectue la passe avant puis arrière : la réponse en phase s'annule exactement, préservant les temps d'arrivée des ondes.

### STA/LTA

```
STA(t) = moyenne( x²[t−Nsta : t] )
LTA(t) = moyenne( x²[t−Nlta : t] )
R(t)   = STA(t) / LTA(t)   →  déclenchement quand R > seuil
```

Implémentation O(N) par sommes préfixes : `cs = cumsum(x²)`.

### Méthode de Wadati

```
Distance hypocentrale :  R = sqrt(d² + h²)
Temps de parcours :      t_P = R / Vp
                         t_S = R / Vs
Distance épicentrale :   d ≈ (t_S − t_P) · Vp · Vs / (Vp − Vs)
```

---

## Démarrage rapide

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ouvrir http://localhost:8501 dans le navigateur.

### Comment utiliser l'application

1. Choisir une source de données dans la barre latérale : séisme synthétique, fichier MiniSEED ou CSV.
2. Configurer le filtre Butterworth passe-bande (coupure basse, coupure haute, ordre).
3. Ajuster le détecteur STA/LTA (fenêtre STA, fenêtre LTA, seuil).
4. Explorer les onglets : Forme d'onde, Analyse spectrale, Spectrogramme, STA/LTA, Modèle de vitesse, Théorie.
5. Exporter les résultats en CSV.

### Obtenir des données MiniSEED réelles

- IRIS FDSN : https://ds.iris.edu/wilber3/find_event
- INGV : https://webservices.ingv.it/swagger-ui/index.html
  Format d'export : MiniSEED

---

## Technologies utilisées

| Bibliothèque | Rôle |
|---|---|
| ObsPy | Lecture MiniSEED, détendance |
| SciPy | Filtre Butterworth (SOS), STFT, Welch PSD |
| NumPy | Tableaux de signaux, FFT |
| Plotly | Graphiques interactifs |
| Streamlit | Interface web |
| Pandas | CSV I/O, aperçu des données |

---

## Structure du projet

```
seismiclens/
├── app.py                     # Application Streamlit principale (UI + orchestration)
├── requirements.txt
├── docs/
│   ├── README_EN.md
│   ├── README_IT.md
│   ├── README_FR.md           # ce fichier
│   ├── README_ES.md
│   └── README_DE.md
└── utils/
    ├── data_loader.py         # MiniSEED, CSV, générateur synthétique
    └── signal_processing.py  # FFT, filtre, STA/LTA, PSD, spectrogramme
```

