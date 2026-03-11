"""
Data loading utilities for SeismicLens.
Supports MiniSEED (via ObsPy), CSV, FDSN live fetch (INGV / IRIS),
and synthetic earthquake generation with a physically realistic
layered-crust velocity model.
"""

import numpy as np
import pandas as pd
import io
import urllib.request
import urllib.parse
import urllib.error
from typing import Tuple


# ── MiniSEED ─────────────────────────────────────────────────────────────────

def load_mseed(file_obj) -> Tuple[np.ndarray, float, dict]:
    """
    Load a MiniSEED file using ObsPy.

    Returns
    -------
    signal   : numpy array of samples (first trace, detrended)
    fs       : sampling rate (Hz)
    metadata : dict with network/station/channel/starttime info
    """
    try:
        from obspy import read
    except ImportError:
        raise ImportError(
            "ObsPy is required for MiniSEED support. "
            "Install it with: pip install obspy"
        )

    buf = io.BytesIO(file_obj.read())
    st = read(buf)
    st.detrend("demean")

    tr = st[0]
    signal = tr.data.astype(np.float64)
    fs = float(tr.stats.sampling_rate)

    metadata = {
        "network":          tr.stats.network,
        "station":          tr.stats.station,
        "location":         tr.stats.location,
        "channel":          tr.stats.channel,
        "starttime":        str(tr.stats.starttime),
        "endtime":          str(tr.stats.endtime),
        "npts":             int(tr.stats.npts),
        "sampling_rate_hz": fs,
    }

    return signal, fs, metadata


# ── FDSN Web Services — live fetch (INGV / IRIS / others) ────────────────────

# Known FDSN dataselect endpoints
FDSN_PROVIDERS = {
    "INGV  (Italy)":     "https://webservices.ingv.it/fdsnws/dataselect/1/query",
    "IRIS  (Global)":    "https://service.iris.edu/fdsnws/dataselect/1/query",
    "ORFEUS (Europe)":   "https://www.orfeus-eu.org/fdsnws/dataselect/1/query",
    "GFZ   (Germany)":   "https://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query",
    "RESIF (France)":    "https://ws.resif.fr/fdsnws/dataselect/1/query",
}

# Station search (FDSN station service) — same provider base URL substitution
_STATION_PATH = {
    "INGV  (Italy)":     "https://webservices.ingv.it/fdsnws/station/1/query",
    "IRIS  (Global)":    "https://service.iris.edu/fdsnws/station/1/query",
    "ORFEUS (Europe)":   "https://www.orfeus-eu.org/fdsnws/station/1/query",
    "GFZ   (Germany)":   "https://geofon.gfz-potsdam.de/fdsnws/station/1/query",
    "RESIF (France)":    "https://ws.resif.fr/fdsnws/station/1/query",
}


def fetch_fdsn_waveform(
        provider_name: str,
        network:       str,
        station:       str,
        location:      str,
        channel:       str,
        starttime:     str,
        endtime:       str,
        timeout:       int = 60,
) -> Tuple[np.ndarray, float, dict]:
    """
    Fetch a MiniSEED waveform directly from an FDSN web-service endpoint
    (INGV, IRIS, ORFEUS, GFZ, RESIF …) and return the same
    (signal, fs, metadata) tuple as load_mseed().

    Parameters
    ----------
    provider_name : key from FDSN_PROVIDERS dict
    network       : SEED network code  (e.g. "IV", "IU")
    station       : SEED station code  (e.g. "ACER", "ANMO")
    location      : location code      (e.g. "00", "--", "")
    channel       : channel code       (e.g. "HHZ", "BHZ")
    starttime     : ISO-8601 string    (e.g. "2016-08-24T01:36:00")
    endtime       : ISO-8601 string    (e.g. "2016-08-24T01:40:00")
    timeout       : HTTP timeout in seconds

    Returns
    -------
    signal   : numpy float64 array (first trace, demeaned)
    fs       : sampling rate (Hz)
    metadata : dict with SEED identifiers + timing info
    """
    try:
        from obspy import read as obspy_read
    except ImportError:
        raise ImportError(
            "ObsPy is required for FDSN fetch. "
            "Install it with: pip install obspy"
        )

    base_url = FDSN_PROVIDERS.get(provider_name)
    if base_url is None:
        raise ValueError(f"Unknown FDSN provider: {provider_name!r}. "
                         f"Choose from: {list(FDSN_PROVIDERS)}")

    # location '--' is used by some services to mean empty location code
    loc = "" if location.strip() in ("--", "-") else location.strip()

    params = {
        "network":   network.strip().upper(),
        "station":   station.strip().upper(),
        "location":  loc,
        "channel":   channel.strip().upper(),
        "starttime": starttime.strip(),
        "endtime":   endtime.strip(),
        "format":    "miniseed",
        "nodata":    "404",
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SeismicLens/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"No data found for {network}.{station}.{loc}.{channel} "
                f"between {starttime} and {endtime} at {provider_name}."
            )
        raise ConnectionError(
            f"HTTP {exc.code} from {provider_name}: {exc.reason}\nURL: {url}"
        )
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach {provider_name}: {exc.reason}\nURL: {url}"
        )

    buf = io.BytesIO(raw_bytes)
    st = obspy_read(buf)
    st.detrend("demean")

    tr = st[0]
    signal = tr.data.astype(np.float64)
    fs     = float(tr.stats.sampling_rate)

    metadata = {
        "source":           provider_name,
        "network":          tr.stats.network,
        "station":          tr.stats.station,
        "location":         tr.stats.location,
        "channel":          tr.stats.channel,
        "starttime":        str(tr.stats.starttime),
        "endtime":          str(tr.stats.endtime),
        "npts":             int(tr.stats.npts),
        "sampling_rate_hz": fs,
        "fdsn_url":         url,
    }
    return signal, fs, metadata


# ── CSV ──────────────────────────────────────────────────────────────────────

def load_csv_signal(file_obj) -> Tuple[np.ndarray, float, dict]:
    """
    Load a seismic signal from a CSV file.

    Expected formats (auto-detected):
      - Single column : amplitude values only (fs assumed 100 Hz)
      - Two columns   : time(s), amplitude  → fs inferred from time column
      - Column names are flexible (time/t/seconds + amplitude/amp/counts/data)

    Returns
    -------
    signal   : numpy array of amplitude values
    fs       : sampling rate in Hz
    metadata : basic info dict
    """
    df = pd.read_csv(file_obj)
    df.columns = [c.strip().lower() for c in df.columns]

    time_cols = [c for c in df.columns if any(k in c for k in ["time", "t", "sec", "s"])]
    amp_cols  = [c for c in df.columns if any(k in c for k in ["amp", "count", "data", "val", "y"])]

    if len(df.columns) == 1:
        signal = df.iloc[:, 0].to_numpy(dtype=np.float64)
        fs = 100.0
    elif time_cols and amp_cols:
        t_arr  = df[time_cols[0]].to_numpy(dtype=np.float64)
        signal = df[amp_cols[0]].to_numpy(dtype=np.float64)
        dt_median = np.median(np.diff(t_arr))
        fs = 1.0 / dt_median if dt_median > 0 else 100.0
    else:
        t_arr  = df.iloc[:, 0].to_numpy(dtype=np.float64)
        signal = df.iloc[:, 1].to_numpy(dtype=np.float64)
        dt_median = np.median(np.diff(t_arr))
        fs = 1.0 / dt_median if dt_median > 0 else 100.0

    signal = np.nan_to_num(signal)

    metadata = {
        "source":           "CSV upload",
        "n_samples":        len(signal),
        "sampling_rate_hz": round(fs, 4),
        "duration_s":       round(len(signal) / fs, 2),
    }
    return signal, fs, metadata


# ── Crustal velocity model ────────────────────────────────────────────────────

# IASP91-inspired layered crustal model
# Each layer: (depth_km_top, depth_km_bottom, Vp_km_s, Vs_km_s, rho_g_cm3)
CRUSTAL_LAYERS = [
    (0,   15,  5.80, 3.36, 2.72),   # Upper crust
    (15,  25,  6.50, 3.75, 2.92),   # Middle crust
    (25,  35,  7.00, 4.00, 3.05),   # Lower crust
    (35, 200,  8.04, 4.47, 3.32),   # Upper mantle
]


def crustal_velocity_at_depth(depth_km: float) -> Tuple[float, float]:
    """Return (Vp, Vs) in km/s for the given focal depth using the layered model."""
    for (z0, z1, vp, vs, _) in CRUSTAL_LAYERS:
        if z0 <= depth_km < z1:
            return vp, vs
    # Default: upper-mantle values
    return 8.04, 4.47


def travel_time_layered(dist_km: float, depth_km: float) -> Tuple[float, float]:
    """
    Approximate P and S travel times using a straight-ray through the
    layered crust (good approximation for local earthquakes, dist < 200 km).
    Uses hypocentral distance R = sqrt(dist² + depth²).

    Returns
    -------
    t_p, t_s : travel times in seconds
    """
    R = np.sqrt(dist_km ** 2 + depth_km ** 2)  # hypocentral distance
    vp, vs = crustal_velocity_at_depth(depth_km)
    t_p = R / vp
    t_s = R / vs
    return t_p, t_s


# ── Synthetic earthquake ─────────────────────────────────────────────────────

def generate_synthetic_quake(
        magnitude:   float = 5.5,
        depth_km:    float = 30.0,
        dist_km:     float = None,
        noise_level: float = 0.15,
        duration_s:  float = 120.0,
        fs:          float = 100.0,
) -> Tuple[np.ndarray, float, dict]:
    """
    Generate a physically-inspired synthetic earthquake waveform.

    Seismological model
    -------------------
    Velocity model : layered IASP91-like crust
        Layer           Vp (km/s)   Vs (km/s)
        Upper crust     5.80        3.36
        Middle crust    6.50        3.75
        Lower crust     7.00        4.00
        Upper mantle    8.04        4.47

    Wave phases
    -----------
    - P-wave  : primary compressional wave (highest freq, arrives first)
    - S-wave  : secondary shear wave (lower freq, ~√3 × larger amplitude)
    - Surface : Rayleigh / Love wave coda (very low freq, long duration)

    Amplitude scaling follows the Richter magnitude relation:
        A ∝ 10^(0.8 M - 2.5)

    Noise model
    -----------
    Broadband bandpass noise (0.5–30 Hz) simulates microseismic background.
    """
    rng = np.random.default_rng(seed=42)
    n = int(duration_s * fs)
    t = np.linspace(0, duration_s, n)

    # Epicentral distance
    if dist_km is None:
        dist_km = depth_km * 2.0 + magnitude * 15.0

    t_p, t_s = travel_time_layered(dist_km, depth_km)
    t_surface = t_s + dist_km * 0.025   # approximate Love/Rayleigh delay

    vp, vs = crustal_velocity_at_depth(depth_km)

    # Clamp arrivals within duration
    t_p       = min(t_p,       duration_s * 0.25)
    t_s       = min(t_s,       duration_s * 0.50)
    t_surface = min(t_surface, duration_s * 0.70)

    # Amplitude scaling (Richter-style)
    amp_scale = 10 ** (0.8 * magnitude - 2.5)
    amp_scale = np.clip(amp_scale, 50, 5e5)

    signal = np.zeros(n)

    # ── P-wave ────────────────────────────────────────────────────────────────
    # High-frequency (6–12 Hz), short duration, ~15% of total amplitude
    p_freq  = 6.0 + magnitude * 0.7
    p_width = max(0.4, 2.5 - magnitude * 0.15)
    p_env   = np.exp(-((t - t_p) ** 2) / (2 * p_width ** 2))
    signal += amp_scale * 0.15 * p_env * np.sin(2 * np.pi * p_freq * (t - t_p))

    # ── S-wave ────────────────────────────────────────────────────────────────
    # Lower frequency (2–6 Hz), wider envelope, ~65% of total amplitude
    # Vs ≈ Vp/√3 for Poisson solid → Vs ≈ 0.577 Vp
    s_freq  = 2.5 + magnitude * 0.25
    s_width = max(0.8, 4.5 - magnitude * 0.1)
    s_env   = np.exp(-((t - t_s) ** 2) / (2 * s_width ** 2))
    signal += amp_scale * 0.65 * s_env * np.sin(2 * np.pi * s_freq * (t - t_s))

    # ── Surface waves ─────────────────────────────────────────────────────────
    # Very low frequency (0.3–1 Hz), slowly decaying coda, ~40% amplitude
    if t_surface < duration_s * 0.90:
        surf_freq  = 0.3 + magnitude * 0.06
        surf_decay = 0.04 + 0.008 * magnitude
        surf_env   = np.where(t >= t_surface,
                              np.exp(-surf_decay * (t - t_surface)), 0.0)
        signal += amp_scale * 0.40 * surf_env * np.sin(
            2 * np.pi * surf_freq * (t - t_surface)
        )

    # ── Broadband noise (0.5–30 Hz Butterworth) ───────────────────────────────
    from scipy.signal import butter, sosfiltfilt
    noise = rng.normal(0, 1, n)
    sos   = butter(2, [0.5 / (fs / 2), 30.0 / (fs / 2)], btype="band", output="sos")
    noise = sosfiltfilt(sos, noise)
    noise /= np.std(noise) + 1e-12
    signal += noise_level * amp_scale * 0.05 * noise

    metadata = {
        "type":            "synthetic",
        "magnitude":       magnitude,
        "depth_km":        depth_km,
        "dist_km":         round(dist_km, 1),
        "velocity_model":  f"Vp={vp} km/s  Vs={vs} km/s  (layer at depth)",
        "vp_vs_ratio":     round(vp / vs, 3),
        "p_arrival_s":     round(t_p, 2),
        "s_arrival_s":     round(t_s, 2),
        "sp_delay_s":      round(t_s - t_p, 2),
        "surface_wave_s":  round(t_surface, 2),
        "fs_hz":           fs,
        "duration_s":      duration_s,
        "n_samples":       n,
    }

    return signal.astype(np.float64), fs, metadata
