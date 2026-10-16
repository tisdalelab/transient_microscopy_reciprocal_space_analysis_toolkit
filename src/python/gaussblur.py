"""gaussblur.py -- convolve a 1D profile with a Gaussian kernel."""
import numpy as np


def gaussblur(x, profile, var, conv_method="zero_pad", trim=0):
    """Convolve a 1D profile with a Gaussian kernel of variance `var`.

    Parameters
    ----------
    x : array           position vector (uniform spacing assumed).
    profile : array     profile values sampled at x.
    var : float         Gaussian variance (nm^2). var == 0 returns unblurred.
    conv_method : str   "zero_pad" | "valid_only" | "same_pad".
    trim : int | 'adaptive'  edge samples to trim then zero-pad back.

    Returns
    -------
    blurred : ndarray   blurred profile, resized to len(x).
    """
    x = np.asarray(x, dtype=float)
    profile = np.asarray(profile, dtype=float)

    # Zero-variance shortcut: no blurring, just trim/repad bookkeeping.
    if var == 0:
        blurred = profile.copy()
        if isinstance(trim, str):
            trim = 0
        if trim > 0:
            blurred = blurred[trim:len(blurred) - trim]
        blurred = np.pad(blurred, trim)
        return blurred[:len(x)]

    dx = x[1] - x[0]

    if isinstance(trim, str) and trim == 'adaptive':
        trim = int(round(5 * np.sqrt(var) / dx))

    # Build the kernel spanning +/- 5 sigma, on the same grid spacing as x.
    gauss_start = dx * round(5 * np.sqrt(var) / dx)
    gauss_xvec = np.arange(-gauss_start, gauss_start + dx / 2.0, dx)

    # Continuous Gaussian kernel, area-normalized to conserve total signal.
    gauss_profile = np.exp(-gauss_xvec ** 2 / (2.0 * var))
    gauss_profile = gauss_profile / np.sum(gauss_profile)
    gauss_len = len(gauss_profile)

    if conv_method == "zero_pad":
        blurred = np.convolve(profile, gauss_profile, 'same')
    elif conv_method == "valid_only":
        if len(gauss_xvec) > 0.2 * len(profile):
            print(f"Warning: cutting off {2*len(gauss_xvec)/len(profile)} of profile")
        blurred = np.convolve(profile, gauss_profile, 'valid')
        pad = int(0.5 * (len(profile) - len(blurred)))
        blurred = np.pad(blurred, pad)
    elif conv_method == "same_pad":
        # Replicate edges so the convolution doesn't pull the profile to zero.
        expanded = np.convolve(np.pad(profile, gauss_len, mode='edge'),
                               gauss_profile, 'same')
        blurred = expanded[gauss_len:len(expanded) - gauss_len]
    else:
        print("Invalid method, defaulting to same_pad")
        expanded = np.convolve(np.pad(profile, gauss_len, mode='edge'),
                               gauss_profile, 'same')
        blurred = expanded[gauss_len:len(expanded) - gauss_len]

    # Optional edge trim + re-pad, then force output length to match x.
    if trim > 0:
        blurred = blurred[trim:len(blurred) - trim]
    blurred = np.pad(blurred, trim)
    return blurred[:len(x)]