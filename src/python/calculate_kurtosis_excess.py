"""calculate_kurtosis_excess.py -- excess kurtosis of a weighted 1D profile.
Thanks to Julia Im for writing the original method."""
import numpy as np


def calculate_kurtosis_excess(x, y):
    """Excess kurtosis (kurtosis - 3) of the distribution defined by the
    (unnormalized) weighted profile y(x). Gaussian -> 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    total_y = np.trapz(y, x)                    # normalization constant
    y_norm = y / total_y                        # normalized density

    mu = np.trapz(x * y_norm, x)                # mean position (1st moment)
    m2 = np.trapz((x - mu) ** 2 * y_norm, x)    # 2nd central moment (variance)
    m4 = np.trapz((x - mu) ** 4 * y_norm, x)    # 4th central moment

    return (m4 / (m2 ** 2)) - 3.0               # excess kurtosis