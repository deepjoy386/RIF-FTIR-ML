from sklearn.base import BaseEstimator, TransformerMixin
from preprocess import smooth_spectra, vector_normalize


class SpectralPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = smooth_spectra(X)
        X = vector_normalize(X)
        return X