"""Regenerate the analyses requested in project feedback.

Run from the repository root:
    python scripts/generate_feedback_outputs.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # Script is intended to run without a GUI backend.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import evaluate_model_cv
from models import get_all_models
from preprocess import smooth_spectra, vector_normalize


FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
COLORS = {"Control": "#0072B2", "RIF": "#D55E00"}


def load_and_process_spectra():
    """Load the raw export and apply the project preprocessing once."""
    raw = pd.read_csv(DATA / "RIF_FTIR.csv")
    labels = raw.iloc[0, 1:]
    wavenumbers = raw.iloc[1:, 0].astype(float)
    spectra = raw.iloc[1:, 1:].T.astype(float)
    spectra.columns = wavenumbers
    y = labels.map({"control": 0, "Control": 0, "RIF": 1}).astype(int)

    X_processed = vector_normalize(smooth_spectra(spectra))
    processed = X_processed.copy()
    processed["Label"] = y.to_numpy()
    processed.to_csv(DATA / "processed_spectra.csv")
    return X_processed, y


def plot_processed_mean_spectra(X, y):
    """Plot group means and standard deviations from processed, not raw, data."""
    wn = X.columns.astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group_name in ((0, "Control"), (1, "RIF")):
        group = X.loc[y.to_numpy() == label]
        mean, std = group.mean(axis=0), group.std(axis=0)
        ax.plot(wn, mean, label=group_name, linewidth=2.2, color=COLORS[group_name])
        ax.fill_between(wn, mean - std, mean + std, alpha=0.15, color=COLORS[group_name])

    ax.invert_xaxis()
    ax.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="Vector-normalized absorbance (a.u.)",
           title="Mean Processed FTIR Spectra of Control and RIF Samples")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure2_MeanSpectra.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ordinations(X, y):
    """Show one unsupervised and two supervised spectral projections."""
    y_array = y.to_numpy()
    projections = []

    pca = PCA(n_components=2, random_state=42)
    pca_scores = pca.fit_transform(X)
    projections.append(("PCA (unsupervised)", pca_scores,
                        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"))

    lda = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    lda_scores = lda.fit_transform(X, y_array)
    projections.append(("LDA (supervised)", lda_scores,
                        "LD1", "Jitter (display only)"))

    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X, y_array)
    pls_scores = pls.transform(X)
    projections.append(("PLS-DA (supervised)", pls_scores, "PLS component 1", "PLS component 2"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rng = np.random.default_rng(42)
    for ax, (title, scores, x_label, y_label) in zip(axes, projections):
        for label, group_name, marker in ((0, "Control", "o"), (1, "RIF", "^")):
            mask = y_array == label
            y_values = (rng.normal(0, 0.035, mask.sum()) if scores.shape[1] == 1
                        else scores[mask, 1])
            ax.scatter(scores[mask, 0], y_values, label=group_name, marker=marker,
                       c=COLORS[group_name], edgecolor="black", linewidth=0.4, alpha=0.82)
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure3_Ordinations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_lda(X, y):
    """Project spectra with PCA, then separate classes with LDA on PCA scores."""
    y_array = y.to_numpy()
    # PCA avoids fitting LDA directly on the high-dimensional spectra.  Retain
    # enough unsupervised components to represent 95% of spectral variance,
    # then learn the discriminant axis using the class labels.
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    pca_scores = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    lda_scores = lda.fit_transform(pca_scores, y_array).ravel()

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.035, len(lda_scores))
    for label, group_name, marker in ((0, "Control", "o"), (1, "RIF", "^")):
        mask = y_array == label
        ax.scatter(lda_scores[mask], jitter[mask], label=group_name, marker=marker,
                   s=58, c=COLORS[group_name], edgecolor="black", linewidth=0.45, alpha=0.88)
    ax.axvline(0, color="#8A8A8A", linewidth=0.8, linestyle="--", zorder=1)
    ax.set(
        title=f"PCA-LDA Separation of Control and RIF Samples ({pca.n_components_} PCs; 95% variance)",
        xlabel="Linear discriminant 1 (LD1)",
        ylabel="Jittered sample position (display only)",
        yticks=[],
    )
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(title="Sample group", frameon=True, facecolor="white", edgecolor="#BDBDBD")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure3_PCA_LDA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lda(X, y):
    """Create a stand-alone supervised LDA score plot with a clear legend."""
    y_array = y.to_numpy()
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    scores = lda.fit_transform(X, y_array).ravel()
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, group_name, marker in ((0, "Control", "o"), (1, "RIF", "^")):
        mask = y_array == label
        ax.scatter(
            scores[mask], rng.normal(0, 0.035, mask.sum()),
            label=group_name, marker=marker, s=58, color=COLORS[group_name],
            edgecolor="#222222", linewidth=0.45, alpha=0.88, zorder=3,
        )

    ax.axvline(0, color="#8A8A8A", linewidth=0.8, linestyle="--", zorder=1)
    ax.set(
        title="Supervised Separation of Control and RIF Samples by LDA",
        xlabel="Linear discriminant 1 (LD1)",
        ylabel="Jittered sample position (display only)",
        yticks=[],
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        title="Sample group", frameon=True, facecolor="white", edgecolor="#BDBDBD",
        ncols=1, loc="upper right", bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0, handletextpad=0.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure3_LDA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(X, y):
    rows = []
    for name, model in get_all_models().items():
        print(f"Evaluating {name}...")
        scores = evaluate_model_cv(model, X, y)
        row = {"Model": name}
        for metric, (mean, std) in scores.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    results = pd.DataFrame(rows)
    results[["Model", "accuracy", "precision", "recall", "f1", "roc_auc", "specificity"]].to_csv(
        RESULTS / "model_comparison.csv", index=False
    )
    results.to_csv(RESULTS / "model_comparison_numeric.csv", index=False)
    return results


def plot_model_comparisons(results):
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    model_names = results["Model"].to_numpy()
    x = np.arange(len(model_names))
    width = 0.18
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, ((metric, label), color) in enumerate(zip(metrics, palette)):
        ax.bar(x + (index - 1.5) * width, results[f"{metric}_mean"], width,
               yerr=results[f"{metric}_std"], capsize=3, color=color, label=label)
    ax.set(ylim=(0, 1.08), ylabel="Cross-validation score",
           title="Model Comparison: Metrics Grouped by Classifier")
    ax.set_xticks(x, model_names, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure4_ModelComparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Complementary view requested in the feedback: each metric is a group,
    # so differences between classifiers can be compared within a metric.
    x = np.arange(len(metrics))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (_, model) in enumerate(results.iterrows()):
        ax.bar(x + (index - (len(results) - 1) / 2) * width,
               [model[f"{metric}_mean"] for metric, _ in metrics], width,
               yerr=[model[f"{metric}_std"] for metric, _ in metrics], capsize=2,
               label=model["Model"])
    ax.set(ylim=(0, 1.08), ylabel="Cross-validation score",
           title="Model Comparison: Classifiers Grouped by Metric")
    ax.set_xticks(x, [label for _, label in metrics])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure4_ModelComparison_ByMetric.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    X, y = load_and_process_spectra()
    plot_processed_mean_spectra(X, y)
    plot_ordinations(X, y)
    plot_pca_lda(X, y)
    plot_lda(X, y)
    results = evaluate_models(X, y)
    plot_model_comparisons(results)
    print("Feedback outputs generated successfully.")


if __name__ == "__main__":
    main()
