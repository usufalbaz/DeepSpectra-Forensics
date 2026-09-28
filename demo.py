"""
End-to-End Demo & Calibration Pipeline for DeepSpectra v2.0.
Implements Adversarial Resampling & JPEG Compression Simulation,
trains a robust decision boundary, and validates on degraded samples.
"""

import cv2
import numpy as np
from classifier import DeepSpectraClassifier
from extractor import SpectralExtractor
from skimage import data


def simulate_compression_and_screenshot(
    image: np.ndarray, quality: int = 40, scale_down: float = 0.7
) -> np.ndarray:
    """Simulates social media recompression, resolution downscaling,

    and reconstruction artifacts (mimicking a smartphone screenshot).
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_down), int(w * scale_down)

    # Resampling degradation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    # Lossy JPEG quantization
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", restored, encode_param)

    is_gray = len(image.shape) == 2
    flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
    compressed = cv2.imdecode(encimg, flag)
    return compressed


def run_pipeline_demo():
    print("=" * 65)
    print("   DEEPSPECTRA v2.0: ROBUST FREQUENCY FORENSICS PIPELINE   ")
    print("=" * 65)

    # 1. Initialize Pipeline Modules
    extractor = SpectralExtractor(target_radius=120)
    detector = DeepSpectraClassifier()

    # 2. Base Calibration Sample
    base_img = data.astronaut()
    h, w, _ = base_img.shape
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)

    print("\n[1/4] Generating Adversarial Training Dataset (Raw + Degraded)...")
    X_train = []
    y_train = []
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Real samples: raw lighting shifts + compressed variants
    for brightness in [0.8, 1.0, 1.2]:
        real_sample = np.clip(
            base_gray.astype(float) * brightness, 0, 255
        ).astype(np.uint8)
        X_train.append(extractor.process_image(real_sample))
        y_train.append(0)

        # Injected degraded real sample
        degraded_real = simulate_compression_and_screenshot(
            real_sample, quality=50
        )
        X_train.append(extractor.process_image(degraded_real))
        y_train.append(0)

    # Fake samples: synthetic upsampling grid + compressed variants
    for amplitude in [5, 8, 12]:
        noise = (
            amplitude
            * np.sin(2 * np.pi * grid_x / 8)
            * np.sin(2 * np.pi * grid_y / 8)
        )
        fake_sample = np.clip(
            base_gray.astype(float) + noise, 0, 255
        ).astype(np.uint8)
        X_train.append(extractor.process_image(fake_sample))
        y_train.append(1)

        # Injected degraded fake sample (simulating screenshotted deepfake)
        degraded_fake = simulate_compression_and_screenshot(
            fake_sample, quality=50
        )
        X_train.append(extractor.process_image(degraded_fake))
        y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 3. Train Robust Decision Boundary
    print(
        f"\n[2/4] Calibrating Classifier on {len(X_train)} Robust Feature Vectors (240-D)..."
    )
    detector.train(X_train, y_train)
    detector.save_model("deepspectra_weights.pkl")

    # 4. Evaluation against Unseen Compressed Test Samples
    print(
        "\n[3/4] Stress-Testing Inference on Screenshotted & Compressed Validation Samples..."
    )

    # Test 1: Real image heavily screenshotted
    test_real_screenshotted = simulate_compression_and_screenshot(
        base_gray, quality=35
    )
    feat_real = extractor.process_image(test_real_screenshotted)
    label_real, conf_real = detector.predict(feat_real)

    # Test 2: Fake image heavily screenshotted
    test_fake_noise = (
        8 * np.sin(2 * np.pi * grid_x / 8) * np.sin(2 * np.pi * grid_y / 8)
    )
    test_fake_raw = np.clip(
        base_gray.astype(float) + test_fake_noise, 0, 255
    ).astype(np.uint8)
    test_fake_screenshotted = simulate_compression_and_screenshot(
        test_fake_raw, quality=35
    )
    feat_fake = extractor.process_image(test_fake_screenshotted)
    label_fake, conf_fake = detector.predict(feat_fake)

    # 5. Output Summary
    print("\n[4/4] Verification Summary Completed.")
    print("-" * 65)
    print(
        f"Screenshotted REAL Image -> Predicted: {label_real:<10} | Confidence: {conf_real:.2f}%"
    )
    print(
        f"Screenshotted FAKE Image -> Predicted: {label_fake:<10} | Confidence: {conf_fake:.2f}%"
    )
    print("-" * 65)


if __name__ == "__main__":
    run_pipeline_demo()
