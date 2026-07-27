import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize


def smooth_spectra(
    X,
    window_length=11,
    polyorder=2
):
    """
    Apply Savitzky-Golay smoothing.
    """

    X_smooth = savgol_filter(
        X.values,
        window_length=window_length,
        polyorder=polyorder,
        axis=1
    )

    return pd.DataFrame(
        X_smooth,
        index=X.index,
        columns=X.columns
    )


def vector_normalize(X):
    """
    L2 (vector) normalization.
    """

    X_norm = normalize(
        X.values,
        norm="l2",
        axis=1
    )

    return pd.DataFrame(
        X_norm,
        index=X.index,
        columns=X.columns
    )
