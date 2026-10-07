# 🔬 DeepSpectra-Forensics (v2.0 Robust Release)
> Resilient Frequency-Domain Deepfake & Synthetic Artifact Detection Framework Resistant to Social Media Resampling & Compression.

[![CI Pipeline](https://github.com/usufalbaz/DeepSpectra-Forensics/actions/workflows/ci.yml/badge.svg)](https://github.com/usufalbaz/DeepSpectra-Forensics/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain: AI Safety](https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20AI%20Safety-red.svg)]()

---

## 📌 Theoretical Framework & The Anti-Compression Paradigm

Standard spatial-domain deepfake detectors frequently fail when images undergo lossy JPEG compression, downsampling, or screen capture (social media resampling), as high-frequency generative grid artifacts are aggressively smoothed out.

DeepSpectra v2.0 circumvents this vulnerability via dual frequency-domain mechanisms:
1. Concentric Spectral Variance Tracking: Beyond calculating peak resonance magnitude, the extractor isolates intra-band frequency dispersion across concentric radial Euclidean rings, isolating synthetic lattice perturbations that survive heavy quantization.
2. Adversarial Resampling Calibration: Decision boundaries are calibrated against simulated screenshots, downsampling, and lossy compression artifacts.

---

## 🧠 System Architecture & Mathematical Foundations

    +------------------+      +-------------------------+      +-----------------------------+      +--------------------+
    |   Input Image    | ---> |  2D-FFT Decomposition   | ---> |  Concentric Ring Extractor  | ---> |   Calibrated SVC   |
    | (RGB/Screenshot) |      |   & Log Magnitude Shift |      |  (Peak + Variance = 240-D)  |      |  (Real vs Fake)    |
    +------------------+      +-------------------------+      +-----------------------------+      +--------------------+

### 1. 2D Fast Fourier Transform (FFT)
Given an input grayscale image f(x, y) of dimensions M x N, the centered log magnitude spectrum S(u, v) is derived via:

    S(u, v) = 20 * log( |F_shift(u, v)| + 1 )

### 2. Dual-Metric Concentric Ring Extraction
For each radial Euclidean distance r from the matrix center (u0, v0):
* Peak Energy: V_peak(r) = max S(u, v)
* Spectral Dispersion: V_var(r) = std(S(u, v))

The resulting concatenated feature vector x in R^240 provides a compression-resilient forensic signature evaluated by a calibrated Support Vector Classifier (SVC).

---

## 🛠️ Tech Stack
* Signal Processing & CV: OpenCV, NumPy, Scikit-Image, SciPy
* Classification: Scikit-Learn (Support Vector Machines)
* Testing & Automation: Pytest, GitHub Actions CI
* User Interface: Streamlit, Gradio

---

## 🚀 Quickstart & Usage

1. Clone the Repository:
    git clone https://github.com/usufalbaz/DeepSpectra-Forensics.git
    cd DeepSpectra-Forensics

2. Install Dependencies:
    pip install -r requirements.txt

3. Run Automated Test Suite:
    pytest tests/

4. Launch Interactive Forensic Scanner:
    streamlit run streamlit_app.py

---

## 📄 License
Distributed under the MIT License. See LICENSE for details.
