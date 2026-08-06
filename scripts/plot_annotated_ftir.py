"""Create an annotated FTIR mean-spectrum figure for the RIF dataset.

Run from the repository root:
    python scripts/plot_annotated_ftir.py
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

from preprocess import smooth_spectra, vector_normalize


PEAK_ASSIGNMENTS = {
    1065: "Carbohydrates and phosphates",
    1160: "DNA, phospholipids",
    1225: "Amide III",
    1390: "CH3 bending of lipids and proteins",
    1463: "CH2 bending of lipids",
    1535: "Amide II",
    1628: "C=O bending of Amide I",
    1742: "C=O bending of lipids",
    2092: "Fatty acyl groups",
    2851: "Asymmetric stretch of CH2 of acyl chains (lipids)",
    2912: "CH2 and CH3 stretch of cholesterol, phospholipids",
    3274: "Amide A",
}
COLORS = {"Control": "#0072B2", "RIF": "#D55E00"}


def load_processed_spectra():
    raw = pd.read_csv(ROOT / "data" / "RIF_FTIR.csv")
    labels = raw.iloc[0, 1:].astype(str)
    wavenumbers = raw.iloc[1:, 0].astype(float).to_numpy()
    spectra = raw.iloc[1:, 1:].T.astype(float)
    processed = vector_normalize(smooth_spectra(spectra))
    groups = labels.str.lower().map({"control": "Control", "rif": "RIF"})
    return wavenumbers, processed.to_numpy(), groups.to_numpy()


def main():
    wavenumbers, spectra, groups = load_processed_spectra()
    order = np.argsort(wavenumbers)
    wn, spectra = wavenumbers[order], spectra[:, order]

    fig, ax = plt.subplots(figsize=(15, 8.2))
    for group in ("Control", "RIF"):
        values = spectra[groups == group]
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.plot(wn, mean, color=COLORS[group], linewidth=2.35, label=group, zorder=3)
        ax.fill_between(wn, mean - std, mean + std, color=COLORS[group], alpha=0.14, zorder=2)

    y_min, y_max = ax.get_ylim()
    measured_min, measured_max = wn.min(), wn.max()
    plotted_peaks = [peak for peak in PEAK_ASSIGNMENTS if measured_min <= peak <= measured_max]
    for peak in plotted_peaks:
        ax.axvline(peak, color="#777777", linestyle=":", linewidth=1.0, alpha=0.72, zorder=1)
        ax.text(peak, y_max * 0.992, f"{peak}", rotation=90, ha="center", va="top",
                fontsize=12, color="#252525", clip_on=False)

    ax.set_xlim(measured_max, measured_min)
    ax.set_ylim(y_min, y_max * 1.10)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=13)
    ax.set_ylabel("Vector-normalized absorbance (a.u.)", fontsize=13)
    ax.set_title("Annotated Mean FTIR Spectra of RIF and Control Samples", fontsize=15, pad=16)
    ax.tick_params(labelsize=11)
    ax.legend(title="Group", loc="upper right", frameon=True, edgecolor="#BDBDBD")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    fig.tight_layout()
    output = ROOT / "figures" / "Figure8_AnnotatedFTIRSpectra.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
