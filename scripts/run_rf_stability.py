"""Check random-forest nested-CV stability and export an OOF confusion matrix."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import get_all_models, get_model_param_grids


def main():
    spectra = pd.read_csv(ROOT / "data" / "processed_spectra.csv", index_col=0)
    X, y = spectra.drop(columns="Label"), spectra["Label"]
    model = get_all_models()["Random Forest"]
    grid = get_model_param_grids()["Random Forest"]
    rows = []
    seed42_truth, seed42_pred = None, None

    for seed in range(42, 47):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        predictions = np.empty(len(y), dtype=int)
        fold_accuracies = []
        for train_idx, test_idx in outer.split(X, y):
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            search = GridSearchCV(
                model, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True
            )
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = search.predict(X.iloc[test_idx])
            predictions[test_idx] = pred
            fold_accuracies.append(accuracy_score(y.iloc[test_idx], pred))

        rows.append({
            "outer_seed": seed,
            "oof_accuracy": accuracy_score(y, predictions),
            "outer_fold_accuracy_mean": float(np.mean(fold_accuracies)),
            "outer_fold_accuracy_std": float(np.std(fold_accuracies)),
        })
        if seed == 42:
            seed42_truth, seed42_pred = y.to_numpy(), predictions.copy()

    stability = pd.DataFrame(rows)
    stability.to_csv(ROOT / "results" / "rf_seed_stability.csv", index=False)

    cm = confusion_matrix(seed42_truth, seed42_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=["Control", "RIF"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Random Forest: 5-fold nested-CV out-of-fold predictions")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "Figure6_ConfusionMatrix.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print(stability.to_string(index=False))
    print("Seed 42 confusion matrix (rows=true Control/RIF; columns=predicted Control/RIF):")
    print(cm)


if __name__ == "__main__":
    main()
