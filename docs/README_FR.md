<div align="center">

# 🌍 SeismicLens

**Analyseur Interactif de Formes d'Ondes Sismiques**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![ObsPy](https://img.shields.io/badge/ObsPy-1.4%2B-green)](https://docs.obspy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.13%2B-8CAAE6?logo=scipy)](https://scipy.org/)

*Chargez, filtrez et analysez des formes d'ondes sismiques directement dans votre navigateur — aucune installation au-delà de Python n'est requise.*

[🇬🇧 English](../README.md) · [🇮🇹 Italiano](README_IT.md) · **🇫🇷 Français** · [🇪🇸 Español](README_ES.md) · [🇩🇪 Deutsch](README_DE.md)

</div>

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Démarrage rapide](#-démarrage-rapide)
- [Guide d'utilisation](#-guide-dutilisation)
- [Obtenir des données réelles](#️-obtenir-des-données-réelles)
- [Modèle sismologique](#-modèle-sismologique)
- [Bases mathématiques](#-bases-mathématiques)
- [Stack technologique](#️-stack-technologique)
- [Structure du projet](#-structure-du-projet)
- [Feuille de route](#️-feuille-de-route)
- [Contribuer](#-contribuer)
- [Licence](#-licence)

---

## 🔭 Vue d'ensemble

**SeismicLens** est un établi de géophysique open-source, basé sur le navigateur et développé avec [Streamlit](https://streamlit.io/). Il permet aux étudiants, chercheurs et passionnés de séismes d'explorer les signaux sismiques sans écrire une seule ligne de code.

**Ce que vous pouvez faire :**

- Générer des **sismogrammes synthétiques physiquement réalistes** à l'aide d'un modèle de vitesse crustal en couches inspiré d'IASP91
- Charger et analyser de **vraies formes d'ondes MiniSEED** provenant de réseaux mondiaux (IRIS, INGV, GEOFON, …)
- Appliquer un **filtre passe-bande Butterworth à phase nulle** pour isoler les bandes de fréquence d'intérêt
- Détecter automatiquement les arrivées d'ondes P avec le classique **algorithme STA/LTA** (Allen, 1978)
- Décomposer le signal via **FFT** et visualiser le spectre d'amplitude, le spectre de phase et la PSD de Welch
- Inspecter l'évolution temps-fréquence avec un **spectrogramme interactif (STFT)**
- **Exporter** toutes les données traitées en CSV pour une analyse ultérieure dans Python, MATLAB ou Excel
- Basculer entre **5 langues** (EN / IT / FR / ES / DE) et les **thèmes sombre / clair**

---

## ✨ Fonctionnalités

| Fonctionnalité | Détails |
|---|---|
|  Générateur de formes d'ondes synthétiques | Ondes P, S et de surface · modèle crustal en couches IASP91 · M, profondeur, distance, bruit et durée configurables |
|  Chargement MiniSEED | Données réelles depuis IRIS FDSN ou les webservices INGV |
|  Chargement CSV | Détection automatique du format à 1 colonne (amplitude) ou 2 colonnes (temps, amplitude) |
| ️ Filtre Butterworth à phase nulle | Passe-bande configurable · ordre 2–8 · aucune distorsion de phase (`sosfiltfilt`) |
|  Analyse spectrale FFT | Spectre d'amplitude · spectre de phase · fréquence dominante · centroïde spectral · largeur de bande RMS |
|  Densité spectrale de puissance | Estimation PSD de Welch en counts²/Hz et dB re 1 count²/Hz |
| ️ Spectrogramme (STFT) | Carte de chaleur temps-fréquence interactive · palette Inferno/Viridis |
|  Détecteur STA/LTA d'onde P | Détecteur classique Allen (1978) · O(N) via sommes préfixes · fenêtres et seuil configurables |
|  Modèle de vitesse crustal | Tableau interactif (Vp, Vs, Vp/Vs, coefficient de Poisson, densité) + diagramme à barres horizontal |
|  Panneau Théorie & Maths | DFT/FFT avec nombres complexes · conception Butterworth · magnitude Richter & Mw · méthode de Wadati |
|  Export CSV | Signal filtré · spectre FFT · PSD de Welch · ratio STA/LTA |
|  Interface multilingue | English · Italiano · Français · Español · Deutsch |
|  Thème sombre / clair | Basculer dans la barre latérale · graphiques Plotly adaptatifs au thème |

---

## 🚀 Démarrage rapide

### Prérequis

- Python **3.10 ou supérieur**
- `pip` (fourni avec Python)

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/your-username/seismiclens.git
cd seismiclens

# 2. (Recommandé) Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvrira automatiquement à l'adresse **http://localhost:8501**.

> **Astuce :** Au premier lancement, ObsPy peut prendre quelques secondes pour s'initialiser. Les lancements suivants sont plus rapides grâce au cache de modules de Streamlit.

---

## 📖 Guide d'utilisation

### 1 — Choisir une source de données

Ouvrir la **barre latérale** (☰ si réduite) et sélectionner l'un des trois modes :

| Mode | Quand l'utiliser |
|---|---|
| **Séisme synthétique** | Exploration immédiate — aucun fichier requis. Ajuster la magnitude (M 2–8), la profondeur focale (5–200 km), la distance épicentrale (10–500 km), le niveau de bruit et la durée. |
| **Charger MiniSEED** | Sismogrammes large-bande réels de réseaux mondiaux. Télécharger des fichiers `.mseed` depuis IRIS ou INGV (voir [Obtenir des données réelles](#️-obtenir-des-données-réelles)). |
| **Charger CSV** | Vos propres données de séries temporelles. Une colonne = échantillons d'amplitude ; deux colonnes = temps (s), amplitude. |

### 2 — Configurer le filtre Butterworth

| Paramètre | Valeurs typiques | Effet |
|---|---|---|
| Coupure basse (Hz) | 0,01–2 Hz (télesismique) · 0,5–5 Hz (régional) · 1–15 Hz (local) | Supprime la dérive basse fréquence et le bruit microsismique |
| Coupure haute (Hz) | Doit être < Nyquist (fs/2) | Supprime le bruit culturel haute fréquence |
| Ordre | 2–8 (défaut 4) | Ordre plus élevé → décroissance plus raide, plus de résonances |

Activer/désactiver le **passe-bande à phase nulle** pour comparer le signal filtré avec la forme d'onde brute.

### 3 — Régler le détecteur STA/LTA

```
Règle empirique :  LTA >= 10 x STA
Seuil typique :    3 - 5
```

| Paramètre | Description |
|---|---|
| Fenêtre STA (s) | Moyenne à court terme : capture l'énergie impulsive de l'arrivée (0,2–2 s) |
| Fenêtre LTA (s) | Moyenne à long terme : suit le niveau de bruit de fond (5–60 s) |
| Seuil de déclenchement | Ratio R au-delà duquel une phase sismique est déclarée |

Abaisser le seuil pour capturer les événements faibles ; l'augmenter pour supprimer les faux déclenchements sur des données bruitées.

### 4 — Explorer les onglets d'analyse

| Onglet | Contenu |
|---|---|
| **Forme d'onde** | Graphique temporel · marqueurs d'arrivée P et S · superposition du signal brut activable |
| **Analyse spectrale** | Spectre d'amplitude FFT · spectre de phase optionnel · PSD de Welch optionnelle |
| **Spectrogramme** | Carte de chaleur temps-fréquence STFT |
| **STA/LTA** | Fonction caractéristique STA/LTA · fenêtres de déclenchement mises en évidence |
| **Modèle de vitesse** | Tableau inspiré IASP91 + diagramme à barres Vp/Vs |
| **Théorie & Maths** | DFT, Butterworth, PSD, STFT, physique des ondes, échelles de magnitude — avec équations |
| **Exporter** | Télécharger les fichiers CSV pour chaque grandeur calculée |

### 5 — Exporter les résultats

Toutes les données traitées peuvent être téléchargées en CSV depuis l'onglet **Exporter** :

| Fichier | Contenu |
|---|---|
| `signal.csv` | Temps (s), amplitude filtrée (counts) |
| `fft.csv` | Fréquence (Hz), amplitude, phase (degrés) |
| `psd.csv` | Fréquence (Hz), PSD (counts²/Hz) |
| `stalta.csv` | Temps (s), ratio STA/LTA |

---

## 🛰️ Obtenir des données réelles

### IRIS (Incorporated Research Institutions for Seismology)

1. Aller sur **[IRIS Wilber 3](https://ds.iris.edu/wilber3/find_event)**
2. Rechercher un séisme par date, région ou magnitude
3. Sélectionner une station sismique proche de l'événement
4. Choisir **MiniSEED** comme format d'export et télécharger

### INGV (Istituto Nazionale di Geofisica e Vulcanologia — Italie)

1. Ouvrir le **[webservice INGV FDSN](https://webservices.ingv.it/swagger-ui/index.html)**
2. Utiliser l'endpoint `/dataselect` avec les paramètres de réseau, station et temps
3. Définir le format sur `miniseed` et télécharger

### Autres réseaux

| Réseau | URL |
|---|---|
| GEOFON (GFZ Potsdam) | https://geofon.gfz-potsdam.de/waveform/ |
| EMSC | https://www.seismicportal.eu/ |
| ORFEUS | https://www.orfeus-eu.org/data/eida/ |

---

## 🌐 Modèle sismologique

### Modèle de vitesse crustal en couches (inspiré d'IASP91)

| Couche | Profondeur (km) | Vp (km/s) | Vs (km/s) | Vp/Vs | Poisson ν |
|---|---|---|---|---|---|
| Croûte supérieure | 0–15 | 5,80 | 3,36 | 1,726 | 0,249 |
| Croûte médiane | 15–25 | 6,50 | 3,75 | 1,733 | 0,252 |
| Croûte inférieure | 25–35 | 7,00 | 4,00 | 1,750 | 0,257 |
| Manteau supérieur | 35–200 | 8,04 | 4,47 | 1,798 | 0,272 |

### Physique de la forme d'onde synthétique

```
Onde P        :  f ~ 6-12 Hz,   enveloppe gaussienne,  amplitude ~ 10^(0.8M-2.5) x 0.15
Onde S        :  f ~ 2-6 Hz,    enveloppe gaussienne,  amplitude ~ 10^(0.8M-2.5) x 0.65
Ondes superf. :  f ~ 0.3-1 Hz,  coda exponentielle,    amplitude ~ 10^(0.8M-2.5) x 0.40
Bruit         :  Butterworth passe-bande 0.5-30 Hz, normalisé à sigma
```

Temps de trajet calculés à partir de la distance hypocentrale `R = sqrt(d^2 + h^2)`.

---

## 📐 Bases mathématiques

### Transformée de Fourier Discrète (DFT)

```
X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2pi * k * n / N)

|X[k]|  ->  amplitude à  f_k = k * fs / N  Hz
angle(X[k])  ->  phase = atan2(Im(X[k]), Re(X[k]))

Spectre unilatéral normalisé :  A[k] = (2/N) * |X[k]|
```

La FFT (Cooley–Tukey, 1965) réduit la complexité de O(N²) à **O(N log N)**.  
Pour les signaux réels, le spectre est hermitien : `X[N-k] = conj(X[k])`, donc seuls N/2+1 bins uniques existent (exploité par `scipy.fft.rfft`).

### Filtre Butterworth à phase nulle

```
|H(jw)|^2 = 1 / [1 + (w/wc)^(2n)]

Décroissance en un seul passage :         20*n  dB/décade
sosfiltfilt (ordre effectif 2n) :         40*n  dB/décade, aucune distorsion de phase
```

Implémenté en forme SOS (Sections du Second Ordre) pour la stabilité numérique.

### STA/LTA (Allen, 1978)

```
STA(t) = (1/Nsta) * sum( x^2[t-Nsta : t] )
LTA(t) = (1/Nlta) * sum( x^2[t-Nlta : t] )
R(t)   = STA(t) / LTA(t)   ->  déclenchement quand R > seuil
```

Implémentation O(N) utilisant des sommes préfixes : `cs = cumsum(x^2)`.

### Méthode de Wadati

```
R = sqrt(d^2 + h^2)              (distance hypocentrale)
t_P = R / Vp  ,  t_S = R / Vs
d ~ (t_S - t_P) * Vp * Vs / (Vp - Vs)
```

---

## 🛠️ Stack technologique

| Bibliothèque | Version | Rôle |
|---|---|---|
| [Streamlit](https://streamlit.io/) | >= 1.35 | Interface web, gestion d'état réactive |
| [ObsPy](https://docs.obspy.org/) | >= 1.4 | I/O MiniSEED, détendance des traces |
| [SciPy](https://scipy.org/) | >= 1.13 | Filtre Butterworth (SOS), STFT, PSD de Welch |
| [NumPy](https://numpy.org/) | >= 1.26 | Tableaux de signal, FFT (rfft) |
| [Plotly](https://plotly.com/python/) | >= 5.22 | Graphiques interactifs |
| [Pandas](https://pandas.pydata.org/) | >= 2.2 | I/O CSV, aperçu des données |

---

## 📁 Structure du projet

```
seismiclens/
├── app.py                     # Application Streamlit principale (UI + orchestration)
├── requirements.txt           # Dépendances Python
├── LICENSE                    # Licence MIT
├── README.md                  # Documentation principale (anglais)
├── docs/
│   ├── README_IT.md           # Documentazione italiana
│   ├── README_FR.md           # Ce fichier
│   ├── README_ES.md           # Documentación española
│   └── README_DE.md           # Deutsche Dokumentation
└── utils/
    ├── __init__.py
    ├── data_loader.py         # MiniSEED, CSV, générateur de formes d'ondes synthétiques
    └── signal_processing.py  # FFT, filtre Butterworth, STA/LTA, PSD, spectrogramme
```

---

## 🗺️ Feuille de route

Les fonctionnalités suivantes sont prévues pour les versions futures. Les contributions sont les bienvenues !

### v2.1 — Données & E/S
- [ ] **Intégration du client FDSN** — interroger IRIS / INGV directement depuis l'app sans téléchargement manuel
- [ ] **Support des fichiers SAC** — charger des sismogrammes au format SAC, largement utilisé en sismologie académique
- [ ] **Affichage multi-traces** — tracer et comparer des enregistrements à trois composantes (Z, N, E) simultanément

### v2.2 — Traitement du signal
- [ ] **Suppression de la réponse instrumentale** — déconvoluer des fichiers RESP / StationXML pour obtenir déplacement/vitesse/accélération du sol
- [ ] **STA/LTA adaptatif** — STA/LTA récursif avec optimisation automatique du seuil
- [ ] **Analyse du mouvement de particule** — diagrammes hodogramme pour les données de composantes Z-N-E
- [ ] **Estimation Coda-Q** — atténuation par diffusion à partir de la décroissance de l'enveloppe de la coda

### v2.3 — Analyse avancée
- [ ] **Visualisation du tenseur de moment** — diagrammes de beach ball à partir des paramètres du mécanisme focal
- [ ] **Traitement en réseau** — beamforming et analyse de lenteur pour les réseaux sismiques
- [ ] **Détecteur de phases par apprentissage automatique** — intégrer PhaseNet ou EQTransformer pour la détection automatisée de phases
- [ ] **Estimation de la magnitude** — calcul de la magnitude locale (Ml) à partir de l'amplitude de crête et des corrections de station

### v3.0 — Plateforme
- [ ] **Streaming en temps réel** — se connecter aux serveurs SeedLink pour la surveillance en direct des formes d'ondes
- [ ] **Persistance de session utilisateur** — sauvegarder et recharger les configurations d'analyse
- [ ] **API REST** — exposer les fonctions de traitement comme endpoints pour un accès programmatique
- [ ] **Image Docker** — déploiement en une seule commande avec `docker run`

---

## 🤝 Contribuer

Les contributions, rapports de bugs et demandes de fonctionnalités sont chaleureusement bienvenus !

1. **Forker** le dépôt
2. Créer une branche de fonctionnalité : `git checkout -b feature/nom-de-votre-fonctionnalite`
3. Commiter vos changements en suivant [Conventional Commits](https://www.conventionalcommits.org/) : `git commit -m "feat: ajouter votre fonctionnalité"`
4. Pousser vers la branche : `git push origin feature/nom-de-votre-fonctionnalite`
5. Ouvrir une **Pull Request** contre `main`

### Style de code

- Suivre **PEP 8** pour le code Python
- Garder les fonctions ciblées et documentées avec des docstrings
- Ajouter ou mettre à jour les tests lors de l'introduction d'une nouvelle logique de traitement du signal
- Exécuter `python -m py_compile app.py utils/*.py` avant d'ouvrir une PR

### Signaler des bugs

Veuillez ouvrir une GitHub Issue en incluant :
- La version de Python et le système d'exploitation
- Les étapes pour reproduire
- Le comportement attendu vs. réel
- Si possible, un fichier d'exemple minimal (MiniSEED ou CSV) qui déclenche le bug

---

## 📄 Licence

Ce projet est sous licence **MIT** — voir le fichier [LICENSE](../LICENSE) pour les détails.

```
MIT License  —  Copyright (c) 2026 Dania Ciampalini & Dario Ciampalini

Il est accordé par la présente, gratuitement, à toute personne obtenant une copie
de ce logiciel et des fichiers de documentation associés (le « Logiciel »), la permission
de traiter le Logiciel sans restriction, notamment les droits d'utiliser, copier, modifier,
fusionner, publier, distribuer, sous-licencier et/ou vendre des copies du Logiciel,
et d'autoriser les personnes à qui le Logiciel est fourni à le faire, sous réserve
des conditions suivantes :

L'avis de droit d'auteur ci-dessus et cet avis de permission doivent être inclus dans
toutes les copies ou parties substantielles du Logiciel.

LE LOGICIEL EST FOURNI « EN L'ÉTAT », SANS GARANTIE D'AUCUNE SORTE, EXPRESSE OU IMPLICITE,
Y COMPRIS, MAIS SANS S'Y LIMITER, LES GARANTIES DE QUALITÉ MARCHANDE, D'ADÉQUATION À UN
USAGE PARTICULIER ET D'ABSENCE DE CONTREFAÇON.
```

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

