# Transient Microscopy Reciprocal Space Analysis Toolkit

Extract carrier- and exciton-transport parameters (diffusivity, diffusion length,
mean-squared displacement, spatial-mode decay rates) from time- and space-resolved
microscopy data such as transient photoluminescence microscopy (TPLM).

The toolkit's distinguishing feature is its **reciprocal-space analysis**: instead of
relying solely on real-space profile widths, it decomposes each spatial profile into
Fourier (Cartesian) or Hankel (cylindrical) modes. Diffusion decouples these modes, so
each mode `k` decays exponentially at rate `Γ(k) = D·k² + 1/τ`. Fitting `Γ(k)` vs `k`
on a log-log scale yields a slope of 2 for ordinary diffusion, and deviations from 2
are a direct signature of anomalous transport.

Seven independent MSD estimators are computed in parallel so that model-dependent and
model-free results can be compared on the same dataset.

Both a **MATLAB** and a **Python** implementation are provided; they are feature-equivalent.

---

## Repository layout
src/
├── MATLAB/      MATLAB implementation + synthetic-data generators
├── python/      Python implementation
└── testdata/    Test dataset(s) for verifying an installation

---

## Quickstart

Only two files matter for getting started:

| File | Role |
| --- | --- |
| `diffimg_analyzer_reciprocal` | **Wrapper / driver.** Loads the test dataset, unpacks the acquisition metadata, reconstructs the position and time axes, and calls `analysis_loop`. |
| `analysis_loop` | **The analysis engine.** Does all of the heavy lifting: plots, fits, reciprocal-space analysis, MSD extraction, diffusivities, kurtosis, and saving. |

### MATLAB
diffimg_analyzer_reciprocal

### Python
cd src/python
python diffimg_analyzer_reciprocal.py

## Verifying your installation
diffimg_analyzer_reciprocal runs on src/testdata/testdata_cyl_0p05.mat, a synthetic
cylindrical-geometry dataset generated with a known diffusivity. The Hankel fit should
return a diffusivity of ≈ 0.05 cm²/s. If it does, the toolkit is working correctly.

## Using your own data
diffimg_analyzer_reciprocal is specific to one acquisition file format, so most users
will write their own short wrapper that loads their data, builds the position and
time vectors, and then calls analysis_loop directly. Use the provided wrapper as a
template. Everything downstream of the wrapper is format-agnostic.

## analysis_loop — the main entry point
### Call signature
MATLAB (name–value pairs):
	[diffs, diff_uncerts, msds] = analysis_loop(rawdata, xvec, tvec, analysis_end_time, ...
												'geometry', 'cylindrical', ...
												'background_subtract', 'estimate');

Python (keyword arguments):
	from analysis_loop import analysis_loop

	diffs, diff_uncerts, msds = analysis_loop(rawdata, xvec, tvec, analysis_end_time,
											  geometry="cylindrical",
											  background_subtract="estimate")

### Required inputs
Argument			Shape		Description
rawdata				[Nx × Nt]	Intensity matrix. Rows = spatial positions, columns = time bins. rawdata(i,j) is the intensity at position i, time j. Strip any metadata columns before passing it in.
xvec				[Nx]		Position vector, nm. Uniform spacing is assumed.
tvec				[Nt]		Time vector, ns. Uniform spacing is assumed.
analysis_end_time	scalar		Time (ns) beyond which data are ignored. Serves as the default cutoff for every sub-fit and as the x-limit of most plots.

### Optional parameters
Parameter			Default				Description
outfilepath			current dir			Output directory for figures and saved results.
diff_fit_time		analysis_end_time	End time (ns) of the linear MSD-vs-t diffusivity fit.
lifetime_fit_time	analysis_end_time	End time (ns) of the TRPL lifetime fit.
freq_fit_time		analysis_end_time	End time (ns) of the reciprocal-space modal decay fits.
geometry			"cartesian"			"cartesian" → 1D FFT. "cylindrical" → 2D radial, quasi-discrete Hankel transform. Use "cylindrical" for a line cut through a radially symmetric spot (the usual TPLM case).
background_subtract	false				false (none), a numeric constant, a numeric vector (per-position profile), or "estimate" to derive a background profile automatically from the pre-rise time window.
profile_center		0					"find" to locate the peak automatically (blur → coarse peak → spline refinement), or a numeric position used to center the spatial axis.
max_k_plot			0.02				Largest k (1/nm) included in the reciprocal-space plots and fits.
baseblur			0					Baseline instrument blur std-dev (nm), added and then removed again in the numerical-convolution MSD method.

### Outputs
Output			Shape			Description
diffs			[7 × 1]			Diffusivity from each MSD method, cm²/s.
diff_uncerts	[7 × 1]			Corresponding uncertainties.
msds			[7 × Nt_fit]	MSD vs time for each method, nm², zeroed at t = 0.
Order for diffusivities returned is always: gauss, voigt, num conv, num int 1d, num int 2d, fourier, hankel.

### What gets written to disk
Figures are saved as PNGs (MATLAB additionally saves editable .fig copies in a figs/
subfolder): spatially-resolved PL, normalized time slices, TRPL with the fitted decay,
2D intensity maps, modal intensity vs time per k, the spatial decay rate vs k power-law
fit, MSD comparison across methods, per-method MSD linear fits, R² vs time, and kurtosis
vs time.
Derived and intermediate quantities are also dumped to a single structured file:
keyValues.mat (MATLAB) or keyValues.json (Python).

### What analysis_loop does, in order
- Diagnostic plots of the raw dataset.
- Fits the spatially integrated decay (TRPL) to a monoexponential in log space to get a lifetime.
- Reciprocal-space spatial-mode decay analysis. Transforms every profile (FFT or QDHT), double-normalizes so each mode starts at 1, fits an exponential decay rate at each k, then fits Γ(k) vs k to a power law in log-log space. In the cylindrical case the k = 0 value is not returned by the QDHT and is instead computed separately by radial quadrature (hankel_k0), which is used for the normalization.
- Estimates the MSD at every time point using the seven methods below.
- Converts MSD-vs-time slopes into diffusivities (slope = 2D) and evaluates sqrt(MSD) at the 1/e point of the TRPL to get a diffusion length.
- Computes the excess kurtosis of each spatial profile vs time.
- Saves everything.

For more information about the 7 fitting methods, please see associated paper.

## Supporting files
Shared (both implementations)
File							Purpose
analysis_loop					The analysis engine described above.
diffimg_analyzer_reciprocal		Example driver/wrapper for the test dataset.
hankel_transformer				Quasi-discrete Hankel transform (QDHT) for radially symmetric data. Forward and inverse transforms; the sampling grids and transform matrices depend only on the position grid and are precomputed once and cached, so many time slices transform cheaply. A radial profile is split into a symmetric part (order-0 transform) and an antisymmetric part (order-1 transform).
gaussblur						Convolves a 1D profile with an area-normalized Gaussian kernel of a given variance. Several boundary-handling modes are available (zero_pad, valid_only, same_pad).
profilefit_1d					The model function behind MSD method 3: returns the initial profile spread by Gaussian diffusion of variance var. Handles var < 0 via a first-order deconvolution trick.
genvoigt						Generates a peak-normalized Voigt profile (Gaussian ⊛ Lorentzian) via the Faddeeva function.
calculate_kurtosis_excess		Excess kurtosis of a distribution defined by a weighted 1D profile. A Gaussian gives 0; positive values indicate heavier tails / a sharper peak, negative values a flatter top.
MATLAB only
File											Purpose
fadf.m											Faddeeva (complex error) function used by genvoigt.m. Third-party; see references.
bessel0s.mat									Precomputed table of Bessel-function zeros required by hankel_transformer.m. Contains 150,001 zeros for each order up to 4 — far more than any realistic dataset needs.
C.mat											Custom colormap for the 2D maps.
diffusion_data_generator_finite_differences.m	Generates synthetic datasets by solving the diffusion equation on a finite-difference grid. Supports two populations with interconversion, configurable lifetimes and PLQYs, Gaussian / Voigt / Airy initial profiles, and optional Poisson noise, pulse pileup, temporal IRF, and detector spatial IRF (the latter are present but commented out).
diffusion_data_generator_kmc.m					Generates synthetic datasets by kinetic Monte Carlo. Useful for ballistic transport and photon recycling simulations.
There are no Python equivalents of the generator scripts; generate test data in MATLAB and load the result from Python if needed.

Python notes
genvoigt.py evaluates the Faddeeva function with scipy.special.wofz directly, so no
fadf port is required. hankel_transformer.py computes its own Bessel zeros at runtime
and does not need bessel0s.mat. The Hankel transformer is exposed as the class
HankelTransformer. The custom colormap is loaded from C.mat if present and silently
falls back to viridis otherwise.

# Requirements
MATLAB
R2023b or newer (trimdata, used in the Fourier MSD fit, was introduced in R2023b).
Curve Fitting Toolbox — fit, fittype, fitoptions, prepareCurveData.
Image Processing Toolbox — padarray, used in gaussblur.m.
Optimization Toolbox — fmincon, only needed when profile_center = "find".
Statistics and Machine Learning Toolbox — poissrnd, only needed by the data generators.
Python
Python 3.9+ (developed and tested on 3.9).
numpy
scipy
matplotlib
tqdm (optional — progress bar; the code falls back gracefully if absent)
	pip install numpy scipy matplotlib tqdm
	
## License
Released under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

Note that `fadf.m` and the quasi-discrete Hankel transform implementation are derived
from third-party MATLAB File Exchange code; please preserve their original attributions
when redistributing (see [References](#references)).
	
# Citation
If you use this toolkit in published work, please cite:

<!-- Paper citation to be added upon publication. -->
and the repository:
https://github.com/tisdalelab/transient_microscopy_reciprocal_space_analysis_toolkit

# Authors and acknowledgements
Developed by Thomas J. Sheehan (tsheeh [at] mit [dot] edu) and William A. Tisdale (tisdale [at] mit [dot] edu), Department of Chemical
Engineering, Massachusetts Institute of Technology.

Thanks to Julia Im for writing the excess-kurtosis analysis method.

References
The quasi-discrete Hankel transform implementation follows:

M. Guizar-Sicairos and J. C. Gutiérrez-Vega, "Computation of quasi-discrete Hankel
transforms of integer order for propagating optical wave fields," J. Opt. Soc. Am. A
21, 53–58 (2004). doi:10.1364/JOSAA.21.000053

and the MATLAB File Exchange implementation (#6570) by Manuel Guizar. Please preserve
these citations when redistributing hankel_transformer.

The excess-kurtosis analysis was inspired by:

E. Arévalo Rodríguez, M. Meléndez, et al., "Quantifying Non-Gaussian Diffusion in
Transient Microscopy Using Excess Kurtosis," J. Phys. Chem. Lett. 17, 2479–2484
(2026). doi:10.1021/acs.jpclett.5c03961

Note that calculate_kurtosis_excess uses the one-dimensional convention
EK = k − 3, appropriate for the line-cut profiles analyzed here; the reference above
reports EK = k − 2 for its two-dimensional distributions.

fadf.m is the Faddeeva/complex error function implementation of S. M. Abrarov and
B. M. Quine (York University), distributed via MATLAB File Exchange; see the header of
that file for its full reference list.

