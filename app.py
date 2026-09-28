"""
DeepSpectra v2.0 Interactive Web Scanner.
Runs a local/public Gradio dashboard for live deepfake detection and FFT heatmap inspection.
"""

import cv2
import gradio as gr
import numpy as np
from classifier import DeepSpectraClassifier
from extractor import SpectralExtractor

# 1. Initialize Engine & Weights
extractor = SpectralExtractor(target_radius=120)
model = DeepSpectraClassifier()

try:
    model.load_model("deepspectra_weights.pkl")
except Exception:
    print("[WARNING] Weights not found. Run demo.py first to calibrate weights.")


def inspect_image(input_image):
    if input_image is None:
        return None, None

    # Grayscale conversion & FFT magnitude
    img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # Generate visual heatmap
    norm_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Feature extraction & inference
    features = extractor.process_image(input_image)
    label, confidence = model.predict(features)

    results = {
        "REAL (Authentic Sensor Signal)": (
            confidence / 100.0 if label == "REAL" else (100.0 - confidence) / 100.0
        ),
        "DEEPFAKE (Synthetic Residue Detected)": (
            confidence / 100.0
            if label == "DEEPFAKE"
            else (100.0 - confidence) / 100.0
        ),
    }
    return results, heatmap_rgb


demo_ui = gr.Interface(
    fn=inspect_image,
    inputs=gr.Image(label="Upload Image or Screenshot"),
    outputs=[
        gr.Label(num_top_classes=2, label="DeepSpectra v2.0 Verdict"),
        gr.Image(label="Extracted Frequency Spectrum (Heatmap)"),
    ],
    title="🛡️ DeepSpectra v2.0 Forensic Scanner",
    description="Upload any facial image or screenshot to analyze frequency-domain perturbations.",
)

if __name__ == "__main__":
    demo_ui.launch()
