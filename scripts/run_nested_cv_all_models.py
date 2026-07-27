"""Nested cross-validation with model-specific hyperparameter grids.

Run from the repository root:
    python scripts/run_nested_cv_all_models.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import nested_cross_validate_models
from models import get_all_models, get_model_param_grids


def plot_nested_results(results):
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    x = np.arange(len(results))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for index, (metric, label) in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            results[f"{metric}_mean"],
            width,
            yerr=results[f"{metric}_std"],
            capsize=3,
            label=label
        )

    ax.set(ylim=(0, 1.08), ylabel="Nested-CV outer-fold score",
           title="Nested Cross-Validation: Tuned Model Comparison")
    ax.set_xticks(x, results["Model"], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "Figure4_NestedCVModelComparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    spectra = pd.read_csv(ROOT / "data" / "processed_spectra.csv", index_col=0)
    X, y = spectra.drop(columns="Label"), spectra["Label"]
    print("Running 5x5 nested CV for all models; inner searches optimize ROC-AUC.")
    summary, folds = nested_cross_validate_models(
        get_all_models(), get_model_param_grids(), X, y,
        outer_splits=5, inner_splits=5, scoring="roc_auc", random_state=42,
        n_jobs=1
    )
    summary.to_csv(ROOT / "results" / "nested_cv_model_comparison.csv", index=False)
    folds.to_csv(ROOT / "results" / "nested_cv_fold_results.csv", index=False)
    plot_nested_results(summary)
    print(summary.round(3).to_string(index=False))
    print("Nested-CV results and figure generated successfully.")


if __name__ == "__main__":
    main()
