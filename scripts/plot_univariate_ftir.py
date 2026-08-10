"""Create a publication-ready univariate FTIR band-area comparison figure.

Bars show mean integrated vector-normalized absorbance across +/- 10 cm^-1;
error bars are SD. Group comparisons use two-sided Mann-Whitney U tests.
Run: python scripts/plot_univariate_ftir.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((1065, "Carbohydrates and phosphates"), (1160, "DNA and phospholipids"),
         (1535, "Amide II"), (1742, "Lipid ester C=O"))
COLORS = ("#F2B879", "#169BB5")


def stars(p):
    return "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    data = pd.read_csv(ROOT / "data" / "processed_spectra.csv")
    wn = data.columns[1:-1].astype(float).to_numpy()
    spectra = data.iloc[:, 1:-1].to_numpy(float)
    labels = data["Label"].to_numpy()
    records = []
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 9.5))

    for ax, panel, (centre, assignment) in zip(axes.flat, "ABCD", BANDS):
        mask = (wn >= centre - 10) & (wn <= centre + 10)
        area = np.trapezoid(spectra[:, mask], wn[mask], axis=1)
        control, rif = area[labels == 0], area[labels == 1]
        u_statistic, p_value = mannwhitneyu(control, rif, alternative="two-sided")
        mean, sd = np.array((control.mean(), rif.mean())), np.array((control.std(ddof=1), rif.std(ddof=1)))
        ax.bar((0, 1), mean, yerr=sd, width=.64, capsize=6, color=COLORS, edgecolor="black", linewidth=1.8,
               error_kw={"elinewidth": 1.4, "capthick": 1.4})
        ax.set_xticks((0, 1), ("Control", "RIF"), rotation=45, ha="right", rotation_mode="anchor", fontweight="bold")
        ax.set_ylabel("Spectral band area (a.u. cm$^{-1}$)", fontweight="bold")
        ax.set_title(f"{centre} cm$^{{-1}}$", fontsize=15, fontweight="bold", pad=13)
        ax.text(-.27, 1.19, f"({panel})", transform=ax.transAxes, fontsize=15, fontweight="bold")
        top = max(mean + sd); bracket, arm = top * 1.17, top * .055
        ax.plot((0, 0, 1, 1), (bracket-arm, bracket, bracket, bracket-arm), color="black", linewidth=1.8, clip_on=False)
        ax.text(.5, bracket + top*.035, stars(p_value), ha="center", va="bottom", fontsize=16, fontweight="bold")
        ax.set_ylim(0, bracket + top*.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.5)
        ax.tick_params(width=1.5, length=5)
        records.append({"Band centre (cm^-1)": centre, "Integration range (cm^-1)": f"{wn[mask][0]:.1f}-{wn[mask][-1]:.1f}",
                        "Assignment": assignment, "Control mean area": mean[0], "Control SD": sd[0], "RIF mean area": mean[1],
                        "RIF SD": sd[1], "Mann-Whitney U": u_statistic, "Two-sided p value": p_value, "Significance": stars(p_value)})
    fig.suptitle("Univariate FTIR Spectral Band-Area Comparisons", fontsize=17, fontweight="bold", y=.99)
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=3.2, w_pad=2.6)
    output = ROOT / "figures" / "Figure3_UnivariateBandAreas.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    pd.DataFrame(records).to_csv(ROOT / "results" / "univariate_band_area_results.csv", index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
