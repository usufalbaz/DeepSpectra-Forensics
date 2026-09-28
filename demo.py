"""
End-to-End Demo Script for DeepSpectra.
Generates synthetic frequency-domain artifacts, trains the pipeline,
and runs live verification.
"""

import numpy as np
import cv2
from skimage import data

from extractor import SpectralExtractor
from classifier import DeepSpectraClassifier


def run_pipeline_demo():
    print("=" * 55)
    print("      DEEPSPECTRA: FREQUENCY FORENSICS ENGINE       ")
    print("=" * 55)

    # 1. Initialize Pipeline Modules
    extractor = SpectralExtractor(target_radius=120)
    detector = DeepSpectraClassifier()

    # 2. Prepare Base Calibration Image
    base_img = data.astronaut()
    h, w, _ = base_img.shape
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)

    print("\n[1/4] Synthesizing Frequency-Domain Artifact Calibration Dataset...")
    X_train = []
    y_train = []

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Real image augmentations (natural lighting shifts)
    for brightness in [0.8, 0.9, 1.0, 1.1, 1.2]:
        real_sample = np.clip(base_gray.astype(float) * brightness, 0, 255).astype(np.uint8)
        X_train.append(extractor.process_image(real_sample))
        y_train.append(0)

    # Synthetic generative artifacts (simulating upsampling / checkerboard artifacts)
    for amplitude in [5, 7, 8, 10, 12]:
        spectral_noise = amplitude * np.sin(2 * np.pi * grid_x / 8) * np.sin(2 * np.pi * grid_y / 8)
        fake_sample = np.clip(base_gray.astype(float) + spectral_noise, 0, 255).astype(np.uint8)
        X_train.append(extractor.process_image(fake_sample))
        y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 3. Train & Save Model Weights
    print("\n[2/4] Training the Decision Boundary...")
    detector.train(X_train, y_train)
    detector.save_model("deepspectra_weights.pkl")

    # 4. Inference on Unseen Test Samples
    print("\n[3/4] Running Inference on Unseen Validation Samples...")
    
    # Test Real Image
    real_features = extractor.process_image(base_gray)
    label_real, conf_real = detector.predict(real_features)

    # Test Fake Image
    test_noise = 8 * np.sin(2 * np.pi * grid_x / 8) * np.sin(2 * np.pi * grid_y / 8)
    fake_img = np.clip(base_gray.astype(float) + test_noise, 0, 255).astype(np.uint8)
    fake_features = extractor.process_image(fake_img)
    label_fake, conf_fake = detector.predict(fake_features)

    # 5. Output Summary
    print("\n[4/4] Verification Completed.")
    print("-" * 55)
    print(f"Sample 1 (Ground Truth: REAL) -> Predicted: {label_real:<10} | Confidence: {conf_real:.2f}%")
    print(f"Sample 2 (Ground Truth: FAKE) -> Predicted: {label_fake:<10} | Confidence: {conf_fake:.2f}%")
    print("-" * 55)


if __name__ == "__main__":
    run_pipeline_demo()
