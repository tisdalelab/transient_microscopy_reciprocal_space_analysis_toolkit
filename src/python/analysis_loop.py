"""analysis_loop.py

Extract carrier-transport parameters from transient (time- and space-resolved)
microscopy data.

This is the main analysis driver. Given a 2D dataset of intensity as a function
of position and time, it:
    1.  Produces diagnostic plots (spatially-resolved PL, time slices, 2D maps).
    2.  Fits the spatially-integrated decay (TRPL) to a biexponential.
    3.  Performs a reciprocal-space (Fourier / Hankel) spatial-mode-decay
        analysis and fits decay-rate vs. k to a power law (diffusive -> slope 2).
    4.  Estimates the mean-squared displacement (MSD) of the profile at every
        time point using SEVEN independent methods (see below).
    5.  Converts MSD-vs-time slopes into diffusivities and diffusion lengths.
    6.  Computes the excess kurtosis of the profile vs. time.
    7.  Saves all intermediate/derived quantities to <outfilepath>/keyValues.json.

Curve fitting is done with scipy.optimize.curve_fit. For each fit we report a
95% confidence-interval half-width (derived from the covariance matrix via the
Student-t distribution) as the parameter "uncertainty", and an ordinary
coefficient of determination R^2 as the goodness-of-fit metric.

All figures are written as PNGs to the <outfilepath>/figs subfolder.

The seven MSD methods (rows of `msds` / `rsq`) are:
    1 Gaussian fit          (variance c)
    2 Voigt fit             (Gaussian part of a Voigt; Lorentzian gamma fixed)
    3 Numerical convolution (blur the initial profile to match later ones)
    4 Numerical integration, 1D (2nd central moment)
    5 Numerical integration, 2D/radial (moment estimate for cylindrical case)
    6 Fourier fit           (Gaussian falloff of the FFT)
    7 Hankel fit            (Gaussian falloff of the Hankel transform)

DEPENDENCIES: gaussblur, genvoigt, profilefit_1d, calculate_kurtosis_excess,
              hankel_transformer, and optionally C.mat (custom colormap).
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline
from scipy.stats import t as tdist

from gaussblur import gaussblur
from genvoigt import genvoigt
from profilefit_1d import profilefit_1d
from calculate_kurtosis_excess import calculate_kurtosis_excess
from hankel_transformer import HankelTransformer

try:
    from tqdm import tqdm
except ImportError:                      # graceful fallback if tqdm is absent
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------------------- #
#  small fitting / indexing helpers
# --------------------------------------------------------------------------- #
def prepare_curve_data(x, y):
    """Flatten x and y to 1D and drop any pairs containing non-finite values."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _fit(func, x, y, p0, lower=-np.inf, upper=np.inf):
    """Nonlinear least-squares fit of `func` to (x, y).

    Returns
    -------
    coeffs : ndarray   best-fit parameters.
    uncert : ndarray   95% confidence-interval half-width for each parameter.
    r2 : float         coefficient of determination.
    """
    p0 = np.array(p0, dtype=float)
    # keep the starting guess strictly inside the bounds
    p0 = np.clip(p0, np.where(np.isfinite(lower), lower, p0),
                 np.where(np.isfinite(upper), upper, p0))
    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=(lower, upper),
                           maxfev=200000)
    perr = np.sqrt(np.abs(np.diag(pcov)))
    dof = max(1, len(y) - len(popt))
    uncert = tdist.ppf(0.975, dof) * perr
    resid = y - func(x, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return popt, uncert, r2


def _linfit(x, y):
    """Straight-line fit y = a*x + b -> (coeffs=[a, b], uncert, r2)."""
    return _fit(lambda xx, a, b: a * xx + b, x, y, [0.0, 0.0])

def _quad_weights(xr):
    """Radial quadrature weights (annular-ring areas) for one half-profile."""
    xr = np.asarray(xr, dtype=float).ravel()
    first = np.sqrt(0.5) * (xr[0] + xr[1]) ** 2
    mid = (xr[2:] - xr[:-2]) * (xr[:-2] + 2 * xr[1:-1] + xr[2:])
    last = 4 * xr[-1] ** 2 - (xr[-2] + xr[-1]) ** 2
    return np.pi / 8 * np.concatenate([[first], mid, [last]])

def _find_last(cond):
    """Number of leading elements up to and including the last True index."""
    idx = np.nonzero(cond)[0]
    return int(idx[-1]) + 1 if idx.size else 0


def _find_first(cond):
    """Index of the first True element, or None if there is none."""
    idx = np.nonzero(cond)[0]
    return int(idx[0]) if idx.size else None


def _to_jsonable(obj):
    """Recursively convert NumPy arrays/scalars (and nested dicts/lists) into
    plain Python types so the structure can be serialized with json.dump."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------- #
def analysis_loop(rawdata, xvec, tvec, analysis_end_time,
                  outfilepath=None, diff_fit_time=None, lifetime_fit_time=None,
                  freq_fit_time=None, geometry="cartesian",
                  background_subtract=False, profile_center=0,
                  max_k_plot=0.02, baseblur=0.0):
    """Run the full transport-parameter analysis on a transient microscopy dataset.

    Required parameters
    -------------------
    rawdata : ndarray [Nx x Nt]
        Intensity matrix. Rows = spatial positions, columns = time bins;
        rawdata[i, j] is the intensity at position i, time j.
    xvec : ndarray [Nx]
        Position vector (nm).
    tvec : ndarray [Nt]
        Time vector (ns).
    analysis_end_time : float
        Time (ns) beyond which data are not analyzed. Used as the default
        cutoff for every sub-fit and as the x-limit of most plots.

    Optional parameters
    -------------------
    outfilepath : str
        Output directory (figures go in its figs/ subfolder; keyValues.json is
        written here). Default: current directory.
    diff_fit_time : float
        End time (ns) for the linear MSD-vs-t diffusivity fit.
    lifetime_fit_time : float
        End time (ns) for the TRPL biexponential fit.
    freq_fit_time : float
        End time (ns) for the reciprocal-space decay fits.
    geometry : str
        "cartesian" (1D, use FFT) or "cylindrical" (2D radial, use Hankel
        transform). Any value other than "cartesian" is treated as cylindrical.
    background_subtract : {False, float, ndarray, "estimate"}
        False applies no background subtraction to the profiles. A numeric
        constant or vector is subtracted from all profiles. "estimate" derives
        a background profile from the pre-rise time window automatically.
    profile_center : {"find", float}
        "find" locates the peak automatically; a number is used directly to
        center the spatial axis (default 0).
    max_k_plot : float
        Largest k (1/nm) to include in reciprocal-space plots.
    baseblur : float
        Baseline blur std-dev (nm) removed in the numerical-convolution MSD
        method.

    Returns
    -------
    diffs : ndarray [7 x 1]
        Diffusivity from each MSD method (cm^2/s).
    diff_uncerts : ndarray [7 x 1]
        Corresponding uncertainties.
    msds : ndarray [7 x Nt_fit]
        MSD vs time for each method (nm^2), zeroed at t=0.
    """

    # ---- defaults for the optional parameters ----------------------------
    if outfilepath is None:
        outfilepath = os.getcwd()
    if diff_fit_time is None:
        diff_fit_time = analysis_end_time
    if lifetime_fit_time is None:
        lifetime_fit_time = analysis_end_time
    if freq_fit_time is None:
        freq_fit_time = analysis_end_time

    # Force 1D arrays so indexing/broadcasting is consistent.
    tvec = np.asarray(tvec, dtype=float).ravel()
    xvec = np.asarray(xvec, dtype=float).ravel()
    rawdata = np.asarray(rawdata, dtype=float)

    # Create output folder + the figs subfolder that holds every figure.
    os.makedirs(outfilepath, exist_ok=True)
    figpath = os.path.join(outfilepath, "figs")
    os.makedirs(figpath, exist_ok=True)

    # Time-integrated intensity at each spatial position (sum over time bins).
    lifetimes_sum = np.sum(rawdata, axis=1)
    pulse_time = np.max(tvec)   # length of the trace (ns) = time between pulses

    # Load the custom colormap used for the 2D maps; fall back to viridis.
    try:
        from scipy.io import loadmat
        from matplotlib.colors import ListedColormap
        Cmap = ListedColormap(loadmat('C.mat')['C'])
    except Exception:
        Cmap = plt.get_cmap('viridis')

    # keyValues accumulates every derived quantity; saved to disk at the end.
    keyValues = {}

    # ===================================================================== #
    #  Preliminary looks at dataset
    # ===================================================================== #
    # Spatially-resolved PL: counts vs position (integrated over all time).
    xData, yData = prepare_curve_data(xvec, lifetimes_sum)
    keyValues['summed_space_x_pos_nm'] = xData
    keyValues['summed_space_y_cts_arb'] = yData

    plt.figure()
    plt.plot(xData, yData)
    plt.xlabel('position (nm)'); plt.ylabel('counts')
    plt.axis([xvec.min(), xvec.max(), 0, yData.max() * 1.2])
    plt.title('spatially-resolved PL')
    plt.savefig(os.path.join(figpath, 'cts vs pos.png'))

    # ---- optionally locate the profile center automatically --------------
    # Blur the time-integrated profile, find the coarse peak, then refine the
    # peak position with a spline + bounded optimizer.
    if str(profile_center) == "find":
        blurred_prof_total = gaussblur(xvec, lifetimes_sum,
                                       10 * (xvec[1] - xvec[0]) ** 2)
        spl = CubicSpline(xvec, blurred_prof_total)
        max_ind = int(np.argmax(blurred_prof_total))
        res = minimize(lambda xp: -spl(xp[0]), x0=[xvec[max_ind]],
                       bounds=[(xvec.min(), xvec.max())])
        profile_center = float(res.x[0])
    else:
        profile_center = float(profile_center)
    # Center the spatial axis on the profile center.
    pos_cor = xvec - profile_center

    # ---- five normalized spatial profiles at evenly spaced times ---------
    end_ind_fit = _find_last(tvec < analysis_end_time)   # count of analyzed bins
    profiles_to_plot = np.round(np.linspace(0, end_ind_fit - 1, 5)).astype(int)
    colors = np.array([[0.1, 0.1, 0.8], [0.1, 0.3, 0.6], [0.1, 0.5, 0.4],
                       [0.1, 0.7, 0.2], [0.1, 1.0, 0.0]])

    plt.figure()
    for i in range(5):
        j = profiles_to_plot[i]
        plt.plot(pos_cor, rawdata[:, j] / rawdata[:, j].max(), lw=1.5,
                 color=colors[i], label=f"{tvec[j]:.2f} ns")
    plt.legend(loc='upper right')
    plt.xlabel('position (nm)'); plt.ylabel('normalized counts')
    plt.axis([pos_cor.min(), pos_cor.max(), 0, 1.2])
    plt.savefig(os.path.join(figpath, 'emission_time_slices.png'))

    # ---- spatially integrated decay (TRPL) -------------------------------
    # Sum over positions -> intensity vs time, normalized to its peak.
    TRPL = np.sum(rawdata, axis=0).astype(float)
    TRPL = TRPL / TRPL.max()
    keyValues['summed_time_norm_intensities'] = TRPL

    # Fit TRPL to a normalized monoexponential in log space:
    #   log(TRPL) = -t / tau     (single coefficient b = tau, no background term)
    maxfittindex = _find_last(tvec < lifetime_fit_time)
    xData, yData = prepare_curve_data(tvec[:maxfittindex],
                                      np.log(TRPL[:maxfittindex]))

    def monoexp_log(x, b):
        return -x / b

    coeffs, uncert, _ = _fit(monoexp_log, xData, yData, [1.0],
                             lower=[1e-6], upper=[100.0])

    keyValues['lifetime'] = coeffs[0]
    keyValues['lifetime_uncert'] = uncert[0]
    print(f"lifetime: {coeffs[0]} ns")

    # Evaluate the fitted decay on a fine grid for overlay plotting.
    x1 = np.arange(0.01, pulse_time - 2, 0.1)
    y1 = np.exp(-x1 / coeffs[0])

    plt.figure()
    plt.plot(tvec, TRPL, '.k')
    plt.plot(x1, y1, 'r-', lw=1)
    plt.yscale('log')
    plt.xlabel('time (ns)'); plt.ylabel('normalized PL')
    plt.axis([0, max(1, round(pulse_time - 2)), 1e-4, 1.2])
    plt.title('TRPL')
    plt.savefig(os.path.join(figpath, 'cts vs time.png'))

    # ===================================================================== #
    #  2D plots (intensity vs position and time)
    # ===================================================================== #
    plt.figure()
    plt.pcolormesh(pos_cor, tvec, rawdata.T, shading='auto', cmap=Cmap)
    plt.colorbar()
    plt.axis([pos_cor.min(), pos_cor.max(), 0, analysis_end_time])
    plt.xlabel('position (nm)'); plt.ylabel('time (ns)')
    plt.savefig(os.path.join(figpath, '2D_map.png'))

    # Same map but each time slice normalized to its own max (shows spreading).
    lifetimes_normtomax = rawdata / rawdata.max(axis=0, keepdims=True)
    plt.figure()
    plt.pcolormesh(pos_cor, tvec, lifetimes_normtomax.T, shading='auto',
                   cmap=Cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.axis([pos_cor.min(), pos_cor.max(), 0, analysis_end_time])
    plt.xlabel('position (nm)'); plt.ylabel('time (ns)')
    plt.savefig(os.path.join(figpath, '2D_normtomax.png'))

    # ===================================================================== #
    #  Background subtraction
    # ===================================================================== #
    back_prof = np.zeros(rawdata.shape[0])
    if str(background_subtract) == "estimate":
        # Walk backwards from the end of the trace to find where the signal
        # starts rising above the local mean (the pre-pulse rise onset), then
        # average a window before that as the background spatial profile.
        rise_begin = len(TRPL) - 1
        while TRPL[rise_begin] > np.mean(TRPL[rise_begin - 5:rise_begin]):
            rise_begin -= 1
        back_prof = np.mean(rawdata[:, rise_begin - 250:rise_begin - 50], axis=1)
    elif np.ndim(background_subtract) or (background_subtract is not False
                                          and background_subtract):
        # A supplied constant or vector background.
        back_prof = back_prof + np.asarray(background_subtract, dtype=float)

    # ===================================================================== #
    #  Reciprocal-space spatial-mode-decay analysis
    # ===================================================================== #
    # Diffusion decouples spatial Fourier/Hankel modes: each mode k should decay
    # exponentially at rate Gamma(k) = D*k^2 + 1/tau. Fitting Gamma(k) vs k
    # (log-log) gives slope 2 for pure diffusion and a prefactor ~ D.
    dx = pos_cor[1] - pos_cor[0]
    Nx = len(pos_cor)
    k_fourier = np.arange(Nx) * 2 * np.pi / (Nx * dx)
    fft_all = np.real(np.fft.fft(np.fft.ifftshift(rawdata - back_prof[:, None],
                                                  axes=0), axis=0))

    # Quasi-discrete Hankel transform (for radially symmetric / cylindrical data).
    hankel_txer = HankelTransformer(pos_cor)
    hankel_all, k_hankel = HankelTransformer.ht(rawdata - back_prof[:, None],
                                                hankel_txer)

    # ---- calculate the k = 0 value separately ----------------------------
    # The QDHT does not return k = 0, so integrate each profile over the plane
    # with radial quadrature weights. Done once here for EVERY time point.
    left_mask = pos_cor <= 0
    right_mask = pos_cor >= 0
    xData_left = pos_cor[left_mask]
    xData_right = pos_cor[right_mask]

    quad_weights_right = _quad_weights(xData_right)
    quad_weights_left = np.flip(_quad_weights(-np.flip(xData_left)))

    prof_left = rawdata[left_mask, :] - back_prof[left_mask, None]
    prof_right = rawdata[right_mask, :] - back_prof[right_mask, None]

    hankel_k0 = (np.sum(quad_weights_right[:, None] * prof_right, axis=0)
                 + np.sum(quad_weights_left[:, None] * prof_left, axis=0))

    # Double-normalize: divide out the t=0 spectrum AND the k=0 decay so every
    # curve starts at 1, isolating the k-dependent decay.
    if str(geometry) == "cartesian":
        recip = (fft_all / fft_all[0, :]) / fft_all[:, 0:1] * fft_all[0, 0]
        k_vec = k_fourier
    else:   # cylindrical (any value other than "cartesian")
        recip = (hankel_all / hankel_k0[None, :]) / hankel_all[:, 0:1] * hankel_k0[0]
        k_vec = k_hankel.ravel()

    # ---- fit an exponential decay rate at each spatial frequency k -------
    fourier_end = _find_last(tvec < freq_fit_time)
    tvec_ff = tvec[:fourier_end]
    kmax = _find_first(k_vec > max_k_plot)   # count of usable k indices
    if kmax is None:
        kmax = len(k_vec)

    spatial_decay_rate = np.zeros(kmax)
    spatial_decay_rate_uncerts = np.zeros(kmax)
    spatial_decay_fit_r2s = np.zeros(kmax)
    all_valid = np.ones(kmax, dtype=bool)   # which k had strictly positive data
    decay_fit_coeffs = np.zeros((kmax, 2))

    for i in range(1, kmax):        # skip i=0 (k=0, the DC/global-decay mode)
        y = recip[i, :fourier_end]
        # If any value is <= 0, log() is complex -> flag this k "not valid" and
        # fit in linear space instead of log space.
        valid = np.imag(np.log(y.astype(complex))) == 0
        if np.any(~valid):
            all_valid[i] = False
        try:
            if all_valid[i]:
                # Clean case: linear fit of log(intensity) vs time; slope = -Gamma.
                c, u, r2 = _linfit(tvec_ff, np.real(np.log(y)))
            else:
                # Noisy case: nonlinear exponential fit directly on the data.
                c, u, r2 = _fit(lambda xx, a, b: np.exp(xx * a) * b,
                                tvec_ff, np.real(y), [0.0, y[0] if y[0] else 1.0])
            spatial_decay_rate[i] = -c[0]
            spatial_decay_rate_uncerts[i] = u[0]
            spatial_decay_fit_r2s[i] = r2
            decay_fit_coeffs[i, :] = c
        except Exception:
            all_valid[i] = False

    # ---- choose a subset of k values to display --------------------------
    r2_thr = 0.0   # keep all fits by default; raise to filter poor fits
    good = np.zeros(kmax, dtype=bool)
    good[1:] = (spatial_decay_fit_r2s[1:] > r2_thr) & (k_vec[1:kmax] <= max_k_plot)
    good_inds = np.nonzero(good)[0]
    nlines = min(len(good_inds), 10)   # plot at most 10 curves
    if nlines > 1:
        k_inds_plot = good_inds[np.round(np.linspace(0, len(good_inds) - 1,
                                                     nlines)).astype(int)]
    else:
        k_inds_plot = good_inds

    # ---- normalized modal intensity vs time (per k) ----------------------
    plt.figure()
    for ki in k_inds_plot:
        plt.semilogy(tvec, recip[ki, :], '-',
                     label=f'k={k_vec[ki]:.4f} (R2={spatial_decay_fit_r2s[ki]:.3f})')
    plt.xlabel('Time (ns)'); plt.ylabel('Normalized Intensity')
    plt.title('Fourier Intensity vs Time (k-normalized)')
    plt.xlim([0, analysis_end_time])
    plt.ylim([0.05, 1.2])
    if len(k_inds_plot):
        plt.legend(fontsize=6)
    plt.savefig(os.path.join(figpath, 'fourier_intensity_vs_t_norm.png'))

    # ---- fit decay-rate vs k to a power law: Gamma = prefactor * k^exp ----
    fi0 = _find_first(~all_valid)
    first_invalid = fi0 if fi0 is not None else kmax
    kvec_fit_valid = k_vec[:kmax][all_valid]
    decay_rate_fit_valid = spatial_decay_rate[all_valid]

    # Only fit the low-k (long-wavelength) regime with the highest signal.
    k_max_slope_fit = 0.01001
    f0 = _find_first(k_vec > k_max_slope_fit)
    k_max_slope_fit_ind = min(kmax, f0 if f0 is not None else kmax)
    count = min(first_invalid, k_max_slope_fit_ind)

    kvec_for_fit = kvec_fit_valid[:count]
    decay_for_fit = decay_rate_fit_valid[:count]
    # Require positive, finite k and decay rate for a log-log fit.
    # Exclude lowest non-zero k value to avoid windowing.
    vf = (~np.isnan(decay_for_fit)) & (decay_for_fit > 0) & (kvec_for_fit > 0)
    kvec_for_fit = kvec_for_fit[vf]
    decay_for_fit = decay_for_fit[vf]

    if len(kvec_for_fit) >= 3:
        # Power law in log-log space: log(Gamma) = exp*log(k) + log(prefactor).
        coeffs_pl, uncert_pl, r2fit = _linfit(np.log(kvec_for_fit),
                                              np.log(decay_for_fit))
        has_power_law_fit = True
    else:
        # Not enough points: assume ideal diffusive slope of 2, flag no fit.
        coeffs_pl = np.array([2.0, 0.0])
        r2fit = 0.0
        has_power_law_fit = False

    # Reference line for the plot: the fitted power law.
    k_vec_fits = k_vec[:kmax]
    fit_line_k = np.linspace(0.8 * k_vec_fits[1],
                             np.max(k_vec_fits[all_valid]) * 1.25, 50)
    fit_line_values = np.exp(coeffs_pl[0] * np.log(fit_line_k) + coeffs_pl[1])
    # Quick diffusivity estimate from the lowest usable k: D = Gamma/k^2.
    # (1e-5 converts nm^2/ns -> cm^2/s.)
    k_D_ind = 1
    D_from_second_lowest_k = spatial_decay_rate[k_D_ind] / k_vec_fits[k_D_ind] ** 2 * 1e-5

    plt.figure()
    # Black = valid (log-space) fits, red = flagged/noisy fits.
    plt.errorbar(k_vec_fits[all_valid], spatial_decay_rate[all_valid],
                 spatial_decay_rate_uncerts[all_valid], fmt='.k')
    plt.errorbar(k_vec_fits[~all_valid], spatial_decay_rate[~all_valid],
                 spatial_decay_rate_uncerts[~all_valid], fmt='.r')
    if has_power_law_fit:
        plt.plot(fit_line_k, fit_line_values, '-b', lw=1.5)
        plt.title(f"Gamma = k^{coeffs_pl[0]:.2f} * {np.exp(coeffs_pl[1]):.3g}, "
                  f"R2={r2fit:.4f}, D @ k={k_vec_fits[k_D_ind]}: {D_from_second_lowest_k:.4f} cm^2/s")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('k (1/nm)'); plt.ylabel('Spatial decay rate (1/ns)')
    plt.savefig(os.path.join(figpath, 'hankel_spatial_decay_vs_rho.png'))
    print("Normalized Spatial Mode analysis:")
    print(f"Gamma = k^{coeffs_pl[0]:.2f} * {np.exp(coeffs_pl[1]):.3g}, "
              f"R2={r2fit:.4f}, D @ k={k_vec_fits[k_D_ind]:.5f}: {D_from_second_lowest_k:.4f} cm^2/s")

    # Store the reciprocal-space results.
    keyValues['k_vec'] = k_vec
    keyValues['freq_analysis'] = dict(
        spatial_decay_rate=spatial_decay_rate,
        spatial_decay_rate_uncerts=spatial_decay_rate_uncerts,
        fit_r2s=spatial_decay_fit_r2s, all_valid=all_valid,
        k_max_ind=kmax, r2_plot_threshold=r2_thr,
        decay_fit_coeffs=decay_fit_coeffs,
        power_law_exponent=coeffs_pl[0],
        power_law_prefactor=np.exp(coeffs_pl[1]),
        power_law_r2=r2fit, has_power_law_fit=has_power_law_fit)

    # ===================================================================== #
    #  MSD extraction (seven methods)
    # ===================================================================== #
    msds = np.zeros((7, end_ind_fit))
    gauss_coeffs = np.zeros((end_ind_fit, 3)); gauss_uncerts = np.zeros((end_ind_fit, 3))
    voigt_coeffs = np.zeros((end_ind_fit, 3)); voigt_uncerts = np.zeros((end_ind_fit, 3))
    num_conv_coeffs = np.zeros((end_ind_fit, 2)); num_conv_uncerts = np.zeros((end_ind_fit, 2))
    num_int_coeffs = np.zeros((2, end_ind_fit))
    fourier_fit_coeffs = np.zeros((end_ind_fit, 2)); fourier_fit_uncerts = np.zeros((end_ind_fit, 2))
    hankel_fit_coeffs = np.zeros((end_ind_fit, 2)); hankel_fit_uncerts = np.zeros((end_ind_fit, 2))
    rsq = np.zeros((7, end_ind_fit))
    times = np.zeros(7)   # accumulated wall-clock time per method (profiling)

    gamma = None          # fixed Lorentzian width (set from the t=0 Voigt fit)
    init_prof = None      # reference profile at t=0 (numerical convolution)
    fft_init = None; cutoffind_init = 1
    hk = {}               # persistent Hankel-fit state built at k=0

    for k in tqdm(range(end_ind_fit), desc='Fitting profiles'):
        # Background-subtracted spatial profile at this time.
        spec = rawdata[:, k] - back_prof
        xDataF, rawyData = prepare_curve_data(pos_cor, spec)

        # ----- Method 1: Gaussian fit -----
        # a*exp(-(x-b)^2 / (2c)); the MSD is the variance c.
        t0 = time.perf_counter()
        def gauss(x, a, b, c):
            return a * np.exp(-((x - b) ** 2) / (2 * c))
        if k == 0:
            p0 = [rawyData.max(), 0, 250000]
        else:
            # Warm-start from the previous time point's fit for stability/speed.
            p0 = gauss_coeffs[k - 1, :]
        c, u, r2 = _fit(gauss, xDataF, rawyData, p0,
                        lower=[0, -100, 1e-9], upper=[np.inf, 100, 40e6])
        gauss_coeffs[k, :] = c; gauss_uncerts[k, :] = u
        rsq[0, k] = r2; msds[0, k] = c[2]
        times[0] += time.perf_counter() - t0

        # ----- Method 2: Voigt fit -----
        # A Voigt is a Gaussian convolved with a Lorentzian. The Lorentzian
        # width gamma is treated as a fixed instrument/intrinsic parameter
        # estimated from the FIRST (t=0) profile, then held constant so that
        # only the Gaussian variance (the diffusive part) varies with time.
        t0 = time.perf_counter()
        yn = rawyData / rawyData.max()
        if k == 0:
            # First frame: fit gamma, sigma^2, center and constant together.
            def voigt4(x, a, b, c2, d):
                return a * genvoigt(b, np.sqrt(c2), d, x)
            c, u, r2 = _fit(voigt4, xDataF, yn, [1, 500, 250000, 0],
                            lower=[0, 1e-9, 1e-9, -100],
                            upper=[np.inf, 2000, 40e6, 100])
            gamma = c[1]                  # fixed Lorentzian width for later frames
            voigt_coeffs[k, :] = c[[0, 2, 3]]
            voigt_uncerts[k, :] = u[[0, 2, 3]]
            rsq[1, k] = r2
            msds[1, k] = voigt_coeffs[k, 1]   # Gaussian variance = MSD contribution
        else:
            # Later frames: gamma is held fixed at the t=0 value.
            def voigt3(x, a, b2, cc):
                return a * genvoigt(gamma, np.sqrt(b2), cc, x)
            c, u, r2 = _fit(voigt3, xDataF, yn, voigt_coeffs[k - 1, :],
                            lower=[0, 1e-9, -100], upper=[np.inf, 40e6, 100])
            voigt_coeffs[k, :] = c; voigt_uncerts[k, :] = u
            rsq[1, k] = r2; msds[1, k] = c[1]
        times[1] += time.perf_counter() - t0

        # ----- Method 3: Numerical convolution -----
        # Model each later profile as the INITIAL profile blurred by a Gaussian
        # of variance sigma2. That added variance IS the MSD (diffusive spread),
        # with no assumption about the profile shape. baseblur accounts for a
        # baseline instrument blur and is subtracted back out afterward.
        t0 = time.perf_counter()
        if k == 0:
            init_prof = rawyData.copy()   # reference profile at t=0
            num_conv_coeffs[k, :] = [1, 0]
            rsq[2, k] = 1; msds[2, k] = 0
        else:
            def convmodel(x, a, sigma2):
                return a * profilefit_1d(x, init_prof, rawyData, sigma2) / init_prof.max()
            lower_var = -200.0 ** 2   # allow slight negative (sharpening) variance
            start2 = baseblur ** 2 + max((xDataF[1] - xDataF[0]) ** 2,
                                         num_conv_coeffs[k - 1, 1])
            ydat = gaussblur(xDataF, rawyData, baseblur ** 2) / rawyData.max()
            c, u, r2 = _fit(convmodel, xDataF, ydat,
                            [num_conv_coeffs[k - 1, 0], start2],
                            lower=[0, lower_var], upper=[np.inf, 40e6])
            num_conv_coeffs[k, :] = c; num_conv_uncerts[k, :] = u
            num_conv_coeffs[k, 1] -= baseblur ** 2    # remove baseline blur
            rsq[2, k] = r2; msds[2, k] = num_conv_coeffs[k, 1]
        times[2] += time.perf_counter() - t0

        # ----- Methods 4 & 5: Numerical integration (moments) -----
        # Model-free MSD from the data moments directly.
        t0 = time.perf_counter()
        mean_x = np.sum(xDataF * rawyData) / np.sum(rawyData)   # centroid
        # 1D 2nd central moment.
        num_int_coeffs[0, k] = np.sum((xDataF - mean_x) ** 2 * rawyData) / np.sum(rawyData)
        # Radial (cylindrical) estimate of the 2D profile variance <r^2>,
        # divided by 2 to express as a per-dimension (1D) MSD.
        num_int_coeffs[1, k] = np.sum(rawyData * np.abs(xDataF ** 3)) \
            / (2 * np.sum(rawyData * np.abs(xDataF)))
        msds[3, k] = num_int_coeffs[0, k]
        msds[4, k] = num_int_coeffs[1, k]
        times[3] += time.perf_counter() - t0

        # ----- Method 6: Fourier fit -----
        # In k-space the profile is the initial spectrum times a Gaussian
        # falloff exp(-k^2*sigma2/2). Fit sigma2 (= MSD) over the k-range where
        # the signal is above a noise threshold. The amplitude is held fixed
        # at 1 rather than fitted.
        t0 = time.perf_counter()
        fourier_noise_threshold = 3
        lower_var = -200.0 ** 2
        if k == 0:
            fft_init = fft_all[:, 0]
            n = len(fft_init)
            # Estimate noise from the mid-band of the spectrum.
            fnoise = np.std(fft_init[n // 3: (2 * n) // 3 + 1])
            cutoffind_init = 1
            while abs(fft_init[cutoffind_init]) > abs(fnoise * fourier_noise_threshold):
                cutoffind_init += 1
            rsq[5, k] = 1; msds[5, k] = 0
        else:
            fft_data = fft_all[:, k]
            n = len(fft_data)
            fnoise = np.std(fft_data[n // 3: (2 * n) // 3 + 1])
            cutoffind = 1
            while abs(fft_data[cutoffind]) > abs(fnoise * fourier_noise_threshold):
                cutoffind += 1
            cutoffind = min(cutoffind, cutoffind_init)

            fixed_init = fft_init[:cutoffind]

            def fmodel(kk, sigma2):   # amplitude fixed to 1
                return fixed_init[:len(kk)] * np.exp(-kk ** 2 * sigma2 / 2)

            kk = k_fourier[:cutoffind]
            ydat = fft_data[:cutoffind] * fft_init.max() / fft_data.max()
            c, u, r2 = _fit(fmodel, kk, ydat, [fourier_fit_coeffs[k - 1, 1]],
                            lower=[lower_var], upper=[40e6])
            fourier_fit_coeffs[k, :] = [1.0, c[0]]
            fourier_fit_uncerts[k, :] = [0.0, u[0]]
            rsq[5, k] = r2; msds[5, k] = c[0]
        times[5] += time.perf_counter() - t0

        # ----- Method 7: Hankel fit -----
        # Cylindrical analogue of the Fourier fit. The (radial) Hankel transform
        # of the profile is fit to the initial transform times a Gaussian
        # falloff exp(-k^2*sigma2/2). Both transforms are normalized by their
        # own k = 0 value (hankel_k0), computed once above by quadrature.
        t0 = time.perf_counter()
        hankel_noise_factor = 7
        lower_var = -200.0 ** 2

        if k == 0:
            cut_length = len(k_hankel.ravel()) // 2
            hankel_init = hankel_all[:cut_length, 0]
            k_hankel_ab = k_hankel.ravel()[:cut_length]

            # Noise cutoff: keep k until the transform drops into the noise floor.
            nl = np.mean(np.abs(hankel_init[cut_length // 3:
                                            (2 * cut_length) // 3 + 1]))
            ci = 1
            while abs(hankel_init[ci]) > abs(nl * hankel_noise_factor):
                ci += 1
            k_hankel_cutoff = 0.03   # hard k ceiling (1/nm) for the Hankel fit
            max_k_ind = int(np.sum(k_hankel_ab <= k_hankel_cutoff))

            hk = dict(cut_length=cut_length, hankel_init=hankel_init,
                      max_k_ind=max_k_ind, k_hankel_ab=k_hankel_ab)
            rsq[6, k] = 1; msds[6, k] = 0
        else:
            cut_length = hk['cut_length']
            prof_hankel = hankel_all[:cut_length, k]
            k_hankel_ab = hk['k_hankel_ab']

            nl = np.mean(np.abs(prof_hankel[cut_length // 3:
                                            (2 * cut_length) // 3 + 1]))
            ci = 1
            while abs(prof_hankel[ci]) > abs(nl * hankel_noise_factor):
                ci += 1
            # Fit at least 5 k-points, but never beyond the noise/ceiling limits.
            fit_len = min(max(5, ci), hk['max_k_ind'])
            k_fit = k_hankel_ab[:fit_len]
            hinit = hk['hankel_init']

            def hmodel(kk, sigma2, a):
                return hinit[:len(kk)] / hankel_k0[0] \
                    * np.exp(-kk ** 2 * sigma2 / 2) * a

            ydat = prof_hankel[:len(k_fit)] / hankel_k0[k]
            c, u, r2 = _fit(hmodel, k_fit, ydat, [dx, 1],
                            lower=[lower_var, 0], upper=[4e6, np.inf])
            hankel_fit_coeffs[k, :] = c; hankel_fit_uncerts[k, :] = u
            rsq[6, k] = r2; msds[6, k] = c[0]   # sigma2 = MSD
        times[6] += time.perf_counter() - t0

    # Zero every MSD trace at t=0 so all methods start at 0 (removes the
    # initial-profile offset; the diffusive spread is what accumulates).
    msds = msds - msds[:, 0:1]
    keyValues['msds_allfits'] = msds
    keyValues['gauss_coeffs'] = gauss_coeffs; keyValues['gauss_uncerts'] = gauss_uncerts
    keyValues['voigt_coeffs'] = voigt_coeffs; keyValues['voigt_uncerts'] = voigt_uncerts
    keyValues['prof_coeffs'] = num_conv_coeffs; keyValues['prof_uncerts'] = num_conv_uncerts
    keyValues['num_int_coeffs'] = num_int_coeffs
    keyValues['fourier_fit_coeffs'] = fourier_fit_coeffs
    keyValues['fourier_fit_uncerts'] = fourier_fit_uncerts
    keyValues['hankel_fit_coeffs'] = hankel_fit_coeffs
    keyValues['hankel_fit_uncerts'] = hankel_fit_uncerts
    keyValues['rsq_allfits'] = rsq

    # ===================================================================== #
    #  R^2 vs time for the profile-shape methods
    # ===================================================================== #
    # (Skips the moment methods 4/5, which have no goodness-of-fit.)
    plt.figure()
    for i in [0, 1, 2, 5, 6]:
        plt.plot(tvec[:rsq.shape[1]], rsq[i, :], lw=1)
    plt.ylim([0.9, 1]); plt.xlim([0, analysis_end_time])
    plt.title('R^2'); plt.xlabel('Time (ns)'); plt.ylabel('R^2')
    plt.legend(["Gauss fit", "Voigt fit", "Num. conv.", "Fourier Fit", "Hankel Fit"])
    plt.savefig(os.path.join(figpath, 'r2_fits_comparison.png'))

    # ===================================================================== #
    #  Diffusion length
    # ===================================================================== #
    # Evaluate sqrt(MSD) at the 1/e point of the TRPL decay -> a diffusion length.
    fitnames = ["gauss", "voigt", "num conv", "num int 1d", "num int 2d",
                    "fourier", "hankel"]
    tvec_0p37 = int(np.argmin(np.abs(TRPL - np.exp(-1))))
    if tvec_0p37 < msds.shape[1]:
        DiffLength = np.sqrt(np.abs(msds[:, tvec_0p37]))
        print("DiffLengths (nm):")
        for i in range(7):
            print(f"  {fitnames[i]}: \t{DiffLength[i]:.6g}")
        keyValues['difflength_nm_allfits'] = DiffLength

    # ===================================================================== #
    #  MSD vs time for all seven methods
    # ===================================================================== #
    plt.figure()
    for i in range(7):
        plt.plot(tvec[:msds.shape[1]], msds[i, :] / 1e6, lw=1)  # /1e6: nm^2 -> um^2
    plt.xlim([0, analysis_end_time]); plt.title('MSD')
    plt.xlabel('Time (ns)'); plt.ylabel('MSD (um^2)')
    plt.legend(["Gauss fit", "Voigt fit", "Num. Conv.", "Num. Int. 1D",
                "Num. Int. 2D", "Fourier Fit", "Hankel Fit"])
    plt.savefig(os.path.join(figpath, 'msd_fits_comparison.png'))

    # ===================================================================== #
    #  Diffusivities
    # ===================================================================== #
    # Linear fit of MSD vs time; slope = 2*D (1D). D = slope/2, with 1e-5
    # converting nm^2/ns -> cm^2/s.
    Diffusivity_allFits = np.zeros(7)
    keyValues.setdefault('msd', {})
    msd_lin = dict(tstart=np.zeros(7), slope=np.zeros(7), slope_uncert=np.zeros(7),
                   yintercept=np.zeros(7), yintercept_uncert=np.zeros(7),
                   diff=np.zeros(7), diff_uncert=np.zeros(7))
    for i in range(7):
        fit_tvec = tvec - tvec[0]
        a = _find_last(tvec < diff_fit_time)   # end index of the linear-fit window
        xData, yData = prepare_curve_data(fit_tvec[:a], msds[i, :a])
        c, u, r2 = _fit(lambda xx, aa, bb: aa * xx + bb, xData, yData, [10000, 0])
        Diffusivity_allFits[i] = c[0] * 1e-5 / 2
        msd_lin['tstart'][i] = tvec[0]
        msd_lin['slope'][i] = c[0]; msd_lin['slope_uncert'][i] = u[0]
        msd_lin['yintercept'][i] = c[1]; msd_lin['yintercept_uncert'][i] = u[1]
        msd_lin['diff'][i] = Diffusivity_allFits[i]
        msd_lin['diff_uncert'][i] = u[0] / c[0] * Diffusivity_allFits[i]
        
    keyValues['msd_linear_tstart'] = msd_lin['tstart']
    keyValues['msd_linear_slope'] = msd_lin['slope']
    keyValues['msd_linear_slope_uncert'] = msd_lin['slope_uncert']
    keyValues['msd_linear_yintercept'] = msd_lin['yintercept']
    keyValues['msd_linear_yintercept_uncert'] = msd_lin['yintercept_uncert']
    keyValues['msd_linear_diff'] = msd_lin['diff']
    keyValues['msd']['linear_diff_uncert'] = msd_lin['diff_uncert']
    print("Diffusivities (cm^2/s):")
    for i in range(7):
        print(f"  {fitnames[i]}: \t{msd_lin['diff'][i]:.6g}")
    print("Uncertainties (cm^2/s):")
    for i in range(7):
        print(f"  {fitnames[i]}: \t{msd_lin['diff_uncert'][i]:.6g}")
    # ===================================================================== #
    #  Kurtosis
    # ===================================================================== #
    # Inspired by Rodriguez et al., J. Phys. Chem. Lett. 2026, 17, 8, 2479-2484.
    # Excess kurtosis of each spatial profile vs time. A pure Gaussian gives 0;
    # deviations indicate non-Gaussian profile shapes.
    kurtosis_calc = np.zeros(end_ind_fit)
    for k in range(end_ind_fit):
        xData, rawyData = prepare_curve_data(pos_cor, rawdata[:, k])
        kurtosis_calc[k] = calculate_kurtosis_excess(xData, rawyData - back_prof)
    keyValues['kurtosis'] = kurtosis_calc
    plt.figure()
    plt.plot(tvec[:len(kurtosis_calc)], kurtosis_calc, lw=1.5)
    plt.xlabel('Time (ns)'); plt.ylabel('Kurtosis')
    plt.xlim([0, analysis_end_time])
    plt.savefig(os.path.join(figpath, 'kurtosis.png'))

    # ===================================================================== #
    #  Save outputs
    # ===================================================================== #
    diffs = Diffusivity_allFits.reshape(-1, 1)
    diff_uncerts = msd_lin['diff_uncert'].reshape(-1, 1)
    with open(os.path.join(outfilepath, "keyValues.json"), "w") as f:
        json.dump(_to_jsonable(keyValues), f, indent=2)
        
    plt.close('all')
    return diffs, diff_uncerts, msds