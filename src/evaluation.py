import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
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

        "precision": make_scorer(precision_score, zero_division=0),

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


def nested_cross_validate_models(
    models,
    param_grids,
    X,
    y,
    outer_splits=5,
    inner_splits=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
):
    """Tune and evaluate each model with an unbiased nested-CV design.

    The inner folds select hyperparameters. The held-out outer fold is used
    only once for evaluation, so reported performance does not benefit from
    seeing the test samples during hyperparameter selection.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )
    summary_rows, fold_rows = [], []

    for model_name, model in models.items():
        if model_name not in param_grids:
            raise KeyError(f"No parameter grid was supplied for {model_name!r}.")

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                scoring=scoring,
                cv=inner_cv,
                refit=True,
                n_jobs=n_jobs
            )
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_prob = search.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            fold_rows.append({
                "Model": model_name,
                "Fold": fold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "best_inner_roc_auc": search.best_score_,
                "best_params": str(search.best_params_)
            })

        model_folds = pd.DataFrame(
            [row for row in fold_rows if row["Model"] == model_name]
        )
        summary = {"Model": model_name}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity"):
            summary[f"{metric}_mean"] = model_folds[metric].mean()
            summary[f"{metric}_std"] = model_folds[metric].std(ddof=0)
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)
