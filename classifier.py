"""
Classifier Module for DeepSpectra.
Handles model training, serialization, deserialization, and probabilistic inference.
"""

import numpy as np
import joblib
from sklearn.svm import SVC


class DeepSpectraClassifier:
    def __init__(self, kernel: str = 'linear'):
        """
        Initialize the SVM classifier with probability calibration.
        """
        self.model = SVC(kernel=kernel, probability=True)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the Support Vector Classifier on spectral feature vectors.
        :param X_train: Matrix of feature vectors of shape (N_samples, N_features).
        :param y_train: Labels (0 = Real, 1 = Fake).
        """
        print("[INFO] Training DeepSpectra Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("[INFO] Training completed successfully.")

    def save_model(self, file_path: str = "deepspectra_model.pkl"):
        """
        Persist the trained model weights to disk.
        """
        if not self.is_trained:
            raise ValueError("Cannot serialize an untrained model!")
        joblib.dump(self.model, file_path)
        print(f"[INFO] Model weights persisted to: {file_path}")

    def load_model(self, file_path: str = "deepspectra_model.pkl"):
        """
        Load pretrained model weights from disk.
        """
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"[INFO] Model weights loaded from: {file_path}")

    def predict(self, feature_vector: np.ndarray):
        """
        Run inference on a single feature vector.
        :param feature_vector: 1D array of frequency features.
        :return: Tuple of (Status string, confidence percentage float).
        """
        if not self.is_trained:
            raise ValueError("Model is not initialized with weights. Train or load weights first.")

        # Ensure correct 2D matrix shape for scikit-learn
        features_2d = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        
        prediction = self.model.predict(features_2d)[0]
        probabilities = self.model.predict_proba(features_2d)[0]
        confidence = float(probabilities[prediction] * 100.0)
        
        label = "DEEPFAKE" if prediction == 1 else "REAL"
        return label, confidence 
