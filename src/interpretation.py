import pandas as pd


def group_adjacent_peaks(
    importance_df,
    threshold=0.008,
    max_gap=8
):
    """
    Groups neighbouring important FTIR peaks into spectral regions.

    Parameters
    ----------
    importance_df : DataFrame
        Must contain:
            Wavenumber
            Importance

    threshold : float
        Ignore features below this importance.

    max_gap : float
        Maximum spacing (cm⁻¹) between adjacent peaks
        to belong to the same region.
    """

    peaks = importance_df[
        importance_df["Importance"] >= threshold
    ].copy()

    peaks = peaks.sort_values("Wavenumber")

    regions = []

    current = [peaks.iloc[0]]

    for _, row in peaks.iloc[1:].iterrows():

        previous = current[-1]

        if abs(
            row["Wavenumber"] -
            previous["Wavenumber"]
        ) <= max_gap:

            current.append(row)

        else:

            regions.append(current)

            current = [row]

    regions.append(current)

    output = []

    for region in regions:

        values = [x["Wavenumber"] for x in region]

        importances = [
            x["Importance"] for x in region
        ]

        output.append({

            "Start": min(values),

            "End": max(values),

            "Mean Importance":
                sum(importances) /
                len(importances),

            "Max Importance":
                max(importances),

            "Peaks":
                len(values)

        })

    return pd.DataFrame(output)