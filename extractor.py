"""
Core Feature Extraction Module for DeepSpectra v2.0.
Extracts dual-metric frequency representations using 2D Fast Fourier Transform (FFT),
capturing both Spectral Peak Energy and Intra-band Variance (Std) to resist lossy compression.
"""

import cv2
import numpy as np


class SpectralExtractor:
    def __init__(self, target_radius: int = 120):
        """
        Initialize the extractor with a target frequency radius.
        :param target_radius: Number of concentric radial frequency bands to inspect.
        """
        self.target_radius = target_radius

    def process_image(self, image_input) -> np.ndarray:
        """
        Converts an image into a 240-D robust spectral feature vector.
        Extracts both max peak magnitude and standard deviation per radial ring.
        :param image_input: Image filepath (str) or loaded numpy image array.
        :return: 1D numpy array of size (target_radius * 2,)
        """
        # 1. Load or convert to grayscale
        if isinstance(image_input, str):
            img_gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        else:
            if len(image_input.shape) == 3:
                img_gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = image_input

        # 2. Compute 2D Fast Fourier Transform (FFT)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        # 3. Compute radial Euclidean distance map from matrix center
        rows, cols = magnitude_spectrum.shape
        y_idx, x_idx = np.indices((rows, cols))
        center_x, center_y = cols // 2, rows // 2
        radial_map = np.hypot(x_idx - center_x, y_idx - center_y).astype(int)

        # 4. Extract Dual Features (Peak + Variance) per concentric ring
        features = []
        for rad in range(self.target_radius):
            mask = (radial_map == rad)
            if np.any(mask):
                ring_values = magnitude_spectrum[mask]
                features.append(np.max(ring_values))
                features.append(np.std(ring_values))
            else:
                features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)
