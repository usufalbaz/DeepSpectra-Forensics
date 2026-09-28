"""
Core Feature Extraction Module for DeepSpectra.
Extracts frequency-domain spectral representations using 2D Fast Fourier Transform (FFT)
and Radial Max-Pooling.
"""

import numpy as np
import cv2


class SpectralExtractor:
    def __init__(self, target_radius: int = 120):
        """
        Initialize the extractor with a target frequency radius.
        :param target_radius: Number of radial frequency bands to inspect.
        """
        self.target_radius = target_radius

    def process_image(self, image_input) -> np.ndarray:
        """
        Converts an image into a 1D spectral feature vector.
        :param image_input: Image filepath (str) or loaded numpy image array.
        :return: 1D numpy array of size (target_radius,)
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

        # 3. Compute radial distance map from center
        rows, cols = magnitude_spectrum.shape
        y_idx, x_idx = np.indices((rows, cols))
        center_x, center_y = cols // 2, rows // 2
        radial_map = np.hypot(x_idx - center_x, y_idx - center_y).astype(int)

        # 4. Extract maximum spectral energy at each radial distance
        features = []
        for rad in range(self.target_radius):
            mask = (radial_map == rad)
            if np.any(mask):
                features.append(np.max(magnitude_spectrum[mask]))
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)
