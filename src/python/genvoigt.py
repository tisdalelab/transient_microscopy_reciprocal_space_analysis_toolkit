"""genvoigt.py -- peak-normalized Voigt profile via the Faddeeva function.
"""
import numpy as np
from scipy.special import wofz


def genvoigt(gamma, sigma, center, xvec):
    """Generate a (peak-normalized) Voigt profile.

    A Voigt profile is the convolution of a Gaussian (std-dev `sigma`) and a
    Lorentzian (half-width `gamma`), computed via the Faddeeva/complex error
    function (scipy.special.wofz).

    Parameters
    ----------
    gamma : float   Lorentzian half-width.
    sigma : float   Gaussian standard deviation.
    center : float  peak center position.
    xvec : array    positions at which to evaluate the profile.

    Returns
    -------
    voigt : ndarray  profile values, normalized so the peak = 1.
    """
    xvec = np.asarray(xvec, dtype=float)
    zvector = (xvec - center + 1j * gamma) / (sigma * np.sqrt(2.0))
    voigt = np.real(wofz(zvector)) / (sigma * np.sqrt(2.0 * np.pi))
    voigt = voigt / np.max(voigt)   # peak-normalize
    return voigt