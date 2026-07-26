from sklearn.pipeline import Pipeline

from transformers import SpectralPreprocessor
from models import get_random_forest


def get_rf_pipeline():

    return Pipeline([
        ("preprocess", SpectralPreprocessor()),
        ("classifier", get_random_forest())
    ])