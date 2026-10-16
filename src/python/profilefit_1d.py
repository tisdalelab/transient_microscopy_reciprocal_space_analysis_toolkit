"""profilefit_1d.py -- model a later profile as the initial profile spread
by Gaussian diffusion of variance `var` (supports var<0 sharpening trick)."""
import numpy as np
from gaussblur import gaussblur


def profilefit_1d(x, init_profile, later_profile, var,
                  conv_method="zero_pad", trim=0):
    """Model a later profile as the initial profile spread by Gaussian
    diffusion of variance `var`. Supports spreading (var>=0) and, via a
    first-order deconvolution trick, apparent sharpening (var<0)."""
    x = np.asarray(x, dtype=float)
    init_profile = np.asarray(init_profile, dtype=float)
    later_profile = np.asarray(later_profile, dtype=float)

    if var >= 0:
        # Normal case: blur the initial profile by the diffusive variance.
        return gaussblur(x, init_profile, var, conv_method, trim)
    else:
        # Negative variance: approximate "un-blurring".
        later_rescaled = later_profile / np.sum(later_profile) * np.sum(init_profile)
        return (later_rescaled + init_profile
                - gaussblur(x, later_rescaled, -var, conv_method, trim))