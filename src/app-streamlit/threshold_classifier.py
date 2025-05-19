# src/app-streamlit/threshold_classifier.py

from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        proba = self.model.predict(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            pos = proba[:, 1]
        else:
            pos = proba.ravel()
        return (pos >= self.threshold).astype(int)
