# DeepSpectra-Forensics 🔬 (v2.0 Robust Release)
> Resilient Frequency-Domain Deepfake Detection Framework Resistant to Social Media Resampling & Screen Capture Artifacts.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Release](https://img.shields.io/badge/Release-v2.0--Robust-orange.svg)](https://github.com/usufalbaz/DeepSpectra-Forensics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20AI%20Safety-red.svg)]()

---

## 📌 What's New in v2.0: The Anti-Compression Upgrade
Standard frequency-based forensic methods often collapse when an image undergoes lossy JPEG compression or screen capture (Screenshots), as high-frequency components are aggressively smoothed out.

DeepSpectra v2.0 solves this vulnerability via two mechanisms:
1. **Concentric Spectral Variance Tracking:** Beyond maximum magnitude, the extractor computes intra-band dispersion, capturing synthetic asymmetry that survives compression.
2. **Adversarial Resampling Augmentation:** The classification boundary is calibrated directly against simulated screen captures, downsampling, and aggressive quantization.

---

## 🧠 Core Methodology & Architecture

The pipeline processes input images through the following deterministic stages:

1. **Spatial to Frequency Decomposition:** Computes 2D Fast Fourier Transform (FFT) on grayscale imagery.
2. **Zero-Frequency Centering:** Shifts DC components to the matrix origin.
3. **Dual Metric Radial Extraction:** Simultaneously captures Peak Energy and Spectral Variance across concentric radial distances.
4. **Discriminative Classification:** Generates a 240-D robust feature vector evaluated by a calibrated Support Vector Classifier (SVC).

---

## 🔬 Mathematical Formulation

### 1. 2D Discrete Fourier Transform
Given an input grayscale image f(x, y) of dimensions M x N, its discrete frequency representation F(u, v) is defined, and the centered magnitude spectrum is:

S(u, v) = 20 * log( |F_shift(u, v)| + 1 )

### 2. Dual-Metric Concentric Ring Extraction
For each frequency band at radial Euclidean distance r, we extract both peak resonance and intra-band variance:

- V_peak(r) = max S(u, v)
- V_variance(r) = standard deviation of S(u, v) within ring r

Even if JPEG compression attenuates the absolute magnitude V_peak, structural generative artifacts remain exposed through the localized perturbation captured in V_variance.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
git clone https://github.com/usufalbaz/DeepSpectra-Forensics.git
cd DeepSpectra-Forensics

### 2. Install Dependencies
pip install -r requirements.txt
pip install gradio

### 3. Run Pipeline Demo
python demo.py

---

## 📂 Repository Structure
- extractor.py : 2D-FFT & Radial Max/Variance Feature Extractor
- classifier.py : Serialized Model Engine with Probability Calibration
- demo.py : CLI Pipeline & Adversarial Calibration Script
- requirements.txt : Production Dependencies
- README.md : Technical & Theoretical Documentation

---

## 📜 License
Distributed under the MIT License. Open for forensic benchmarking and adversarial AI safety research.
