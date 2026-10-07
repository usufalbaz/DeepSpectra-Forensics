import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure root directory is accessible to imports
sys.path.append(str(Path(__file__).parent.parent))

from extractor import SpectralExtractor
from classifier import DeepSpectraClassifier

def test_spectral_extractor_output_dimension():
    extractor = SpectralExtractor(target_radius=120)
    # Generate dummy 100x100 grayscale image
    dummy_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    features = extractor.process_image(dummy_img)
    
    # 120 radial rings * 2 (max + std) = 240 dimensions
    assert isinstance(features, np.ndarray)
    assert features.shape == (240,)
    assert features.dtype == np.float32

def test_classifier_untrained_error_handling():
    detector = DeepSpectraClassifier()
    dummy_features = np.zeros(240, dtype=np.float32)
    
    with pytest.raises(ValueError):
        detector.predict(dummy_features)
