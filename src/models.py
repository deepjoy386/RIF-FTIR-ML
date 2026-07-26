from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


def get_all_models():

    return {

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