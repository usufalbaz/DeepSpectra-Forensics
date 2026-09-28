# DeepSpectra-Forensics:

# DeepSpectra-Forensics 🔬
> **Frequency-Domain Deepfake & Synthetic Artifact Forensics Framework using 2D Fast Fourier Transform (FFT) and Radial Max-Pooling.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20AI%20Safety-red.svg)]()

---

## 📌 Executive Summary & Problem Formulation
Modern generative architectures (such as GANs, Diffusion Models, and VAEs) produce photorealistic facial imagery that reliably fools human observers and spatial-domain convolutional detectors. 

However, generative upsampling operations (such as `Conv2DTranspose` and sub-pixel convolutions) inherently induce periodic, high-frequency structural anomalies—commonly referred to as **Checkerboard Artifacts**. While these traces are virtually invisible in the spatial pixel domain ($X, Y$), they manifest as discrete spectral spikes (Dirac-like delta peaks) in the 2D Frequency Domain.

**DeepSpectra** is an open-source forensic pipeline designed to isolate and classify synthetic generative signatures using frequency decomposition and radial energy distribution analysis.

---

## 🧠 Core Methodology & Architecture

```text
Input Image (Spatial Domain)
       │
       ▼
[ 2D Fast Fourier Transform (FFT) ]
       │
       ▼
[ Spectral Centering (Zero-Frequency Shift) ]
       │
       ▼
[ Concentric Radial Max-Pooling ] ──> Extracts High-Energy Resonant Spikes
       │
       ▼
[ 1D Discriminative Feature Vector (120-D) ]
       │
       ▼
[ Calibrated Linear Support Vector Machine (SVC) ] ──> Output: REAL vs. DEEPFAKE (%)

1. Mathematical Formulation

Given an input grayscale image f(x, y) of dimensions M \times N, its discrete
frequency representation F(u, v) is computed via the 2D Discrete Fourier
Transform:

F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}

The centered magnitude spectrum is then computed:
S(u, v) = 20 \log \left( |F_{shift}(u, v)| + 1 \right)

2. Radial Feature Extraction

Rather than standard azimuthal mean-pooling—which dilutes sparse high-frequency
anomalies—DeepSpectra implements Concentric Radial Max-Pooling across radial
distances r:

V(r) = \max_{(u, v) \in \Omega_r} S(u, v) \quad \text{where} \quad \Omega_r = \left\{ (u, v) \mid \lfloor \sqrt{(u - u_0)^2 + (v - v_0)^2} \rfloor = r \right\}

🚀 Quickstart & Installation

1. Clone the Repository

git clone https://github.com/<YOUR-USERNAME>/DeepSpectra-Forensics.git
cd DeepSpectra-Forensics

2. Install Dependencies

pip install -r requirements.txt

3. Run Pipeline Demo

python demo.py

📊 Verification & Baseline Output

Running the end-to-end forensic calibration pipeline yields calibrated
probabilistic outputs:

=======================================================
      DEEPSPECTRA: FREQUENCY FORENSICS ENGINE       
=======================================================
[1/4] Synthesizing Frequency-Domain Artifact Calibration Dataset...
[2/4] Training the Decision Boundary...
[3/4] Running Inference on Unseen Validation Samples...
[4/4] Verification Completed.
-------------------------------------------------------
Sample 1 (Ground Truth: REAL) -> Predicted: REAL       | Confidence: ~83%
Sample 2 (Ground Truth: FAKE) -> Predicted: DEEPFAKE   | Confidence: ~88%
-------------------------------------------------------

📂 Repository Structure

DeepSpectra-Forensics/
│
├── extractor.py        # Core 2D-FFT and Radial Max-Pooling feature extractor
├── classifier.py       # Scikit-learn SVM classifier with model serialization
├── demo.py             # End-to-end execution and validation script
├── requirements.txt    # Production dependencies
└── README.md           # Engineering documentation

📜 License

Distributed under the MIT License. Open for academic research and forensic
benchmarking.
