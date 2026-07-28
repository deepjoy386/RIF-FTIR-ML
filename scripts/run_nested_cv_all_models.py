"""Nested cross-validation with model-specific hyperparameter grids.

Run from the repository root:
    python scripts/run_nested_cv_all_models.py
"""

from pathlib import Path
import sys
import ast

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
    """Plot nested-CV metrics using a color-blind-safe, publication palette."""
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    x = np.arange(len(results))
    width = 0.18
    # Okabe-Ito palette: distinct in print and for common color-vision deficiencies.
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(12.5, 6.4))

    for index, ((metric, label), color) in enumerate(zip(metrics, colors)):
        ax.bar(
            x + (index - 1.5) * width,
            results[f"{metric}_mean"],
            width,
            yerr=results[f"{metric}_std"],
            capsize=3,
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            label=label,
        )

    # Reserve a small in-panel band above the tallest error bar for the legend.
    ax.set(ylim=(0, 1.14), ylabel="Nested-CV outer-fold score",
           title="Nested Cross-Validation: Tuned Model Comparison")
    ax.set_xticks(x, results["Model"], rotation=18, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Compact two-column key in the clear upper-right corner.
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDBDBD", ncols=2, loc="upper right",
              bbox_to_anchor=(0.985, 0.985), borderaxespad=0.0,
              handlelength=1.3, handletextpad=0.35, columnspacing=0.9,
              fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "Figure4_NestedCVModelComparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_optimized_hyperparameters(folds):
    """Export every fold's inner-CV-selected settings in analysis-ready columns."""
    rows = []
    for _, fold in folds.iterrows():
        params = ast.literal_eval(fold["best_params"])
        row = {
            "Model": fold["Model"],
            "Outer_fold": fold["Fold"],
            "Inner_CV_ROC_AUC": fold["best_inner_roc_auc"],
        }
        row.update({key.removeprefix("classifier__"): value for key, value in params.items()})
        rows.append(row)

    parameter_columns = sorted({key for row in rows for key in row} -
                               {"Model", "Outer_fold", "Inner_CV_ROC_AUC"})
    columns = ["Model", "Outer_fold", "Inner_CV_ROC_AUC", *parameter_columns]
    pd.DataFrame(rows).reindex(columns=columns).to_csv(
        ROOT / "results" / "optimized_hyperparameters.csv", index=False
    )


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
    export_optimized_hyperparameters(folds)
    plot_nested_results(summary)
    print(summary.round(3).to_string(index=False))
    print("Nested-CV results and figure generated successfully.")


if __name__ == "__main__":
    main()
