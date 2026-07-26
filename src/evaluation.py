import numpy as np

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    make_scorer,
    confusion_matrix
)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp)


def evaluate_model_cv(
    model,
    X,
    y,
    folds=5,
    repeats=20
):

    scoring = {

        "accuracy": "accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

        "roc_auc": "roc_auc",

        "specificity":
            make_scorer(
                specificity_score
            )

    }

    cv = RepeatedStratifiedKFold(

        n_splits=folds,

        n_repeats=repeats,

        random_state=42

    )

    scores = cross_validate(

        model,

        X,

        y,

        scoring=scoring,

        cv=cv,

        n_jobs=-1

    )

    summary = {}

    for metric in scoring:

        summary[metric] = (

            np.mean(
                scores[f"test_{metric}"]
            ),

            np.std(
                scores[f"test_{metric}"]
            )

        )

    return summary