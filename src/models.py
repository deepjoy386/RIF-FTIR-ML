import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from transformers import SpectralPreprocessor


def get_logistic_regression():
    return LogisticRegression(
        random_state=42,
        max_iter=1000
    )


def get_svm():
    return SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )


def get_xgboost():
    return XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )


def get_lda():
    """Shrinkage LDA is stable when spectra have more variables than samples."""

    return LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Binary PLS-DA classifier with a 0.5 decision threshold.

    PLSRegression produces continuous class scores; this wrapper makes it
    compatible with scikit-learn classification evaluation and ROC-AUC.
    """

    def __init__(self, n_components=2, scale=False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PLSDAClassifier supports binary classification only.")

        y_binary = (np.asarray(y) == self.classes_[1]).astype(float)
        max_components = min(X.shape[0] - 1, X.shape[1], self.n_components)
        self.model_ = PLSRegression(
            n_components=max_components,
            scale=self.scale
        )
        self.model_.fit(X, y_binary)
        return self

    def predict_proba(self, X):
        positive_probability = np.clip(
            self.model_.predict(X).ravel(), 0, 1
        )
        return np.column_stack((1 - positive_probability, positive_probability))

    def predict(self, X):
        positive_probability = self.predict_proba(X)[:, 1]
        return np.where(
            positive_probability >= 0.5,
            self.classes_[1],
            self.classes_[0]
        )


def get_pls_da():
    return PLSDAClassifier(n_components=2, scale=False)


def get_all_models():

    return {

        "LDA": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_lda())
        ]),

        "PLS-DA": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_pls_da())
        ]),

        "Logistic Regression": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_logistic_regression())
        ]),

        "SVM (RBF)": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_svm())
        ]),

        "Random Forest": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_random_forest())
        ]),

        "XGBoost": Pipeline([
            ("preprocess", SpectralPreprocessor()),
            ("classifier", get_xgboost())
        ])

    }


def get_model_param_grids():
    """Return tuning grids aligned with the pipelines from ``get_all_models``.

    All parameter names are prefixed with ``classifier__`` because model
    selection happens inside a scikit-learn Pipeline.
    """

    return {
        "LDA": {
            "classifier__shrinkage": ["auto", 0.1, 0.5, 0.9]
        },
        "PLS-DA": {
            "classifier__n_components": [1, 2, 3, 4, 5],
            "classifier__scale": [False, True]
        },
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        },
        "SVM (RBF)": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", 0.001, 0.01, 0.1],
            "classifier__class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "classifier__n_estimators": [200, 500],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"]
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.5, 1.0]
        }
    }
