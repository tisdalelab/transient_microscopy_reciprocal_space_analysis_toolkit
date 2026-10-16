"""diffimg_analyzer_reciprocal.py -- example driver script.

Loads a test dataset, reconstructs the position and time axes from the
metadata packed into the raw data array, and calls analysis_loop on each
dataset. Adapt the indices to your own acquisition file format.
"""
import numpy as np
from scipy.io import loadmat
from analysis_loop import analysis_loop
from pathlib import Path

current_dir = Path(__file__).resolve().parent
datapath = current_dir.parent / "testdata/testdata_cyl_0p05.mat"

puredatasets = loadmat(datapath)["puredatasets"]  # [dataset,pos,time(+meta)]
Mag = 495                                        # optical magnification
outfilesuffix = ["gauss", "voigt", "airy"]

for i in range(3):                               # loop over profile shapes
    rawdata = np.squeeze(puredatasets[i, :, :]).astype(float)

    # Crop to a symmetric window of +/- spat_side_bins around the center row.
    spat_cent_ind = int(np.ceil(rawdata.shape[0] / 2)) - 1     # 0-based center
    spat_side_bins = 100
    rawdata = rawdata[spat_cent_ind - spat_side_bins:
                      spat_cent_ind + spat_side_bins + 1, :]

    # ---- Metadata packed into trailing columns (0-based indexing) --------
    Rep = rawdata[1, -1]              # repetition rate (Hz)
    AcqTime = rawdata[1, -2]          # acquisition time (s)
    bin_ns = rawdata[1, -3] / 1000.0  # time-bin width: ps -> ns
    lifetimes_sum = rawdata[:, -4]    # time-integrated counts per position

    # ---- Reconstruct the position axis from scan geometry ----------------
    posdiffs = np.flip(rawdata[1, -6:-4] - rawdata[0, -6:-4])   # per-step (x,y)
    step_size = np.linalg.norm(posdiffs) / 1e6                  # -> mm

    scan_angle = np.arctan(posdiffs[1] / posdiffs[0])
    if scan_angle < 0:
        scan_angle += np.pi / 2
    scan_angle = np.degrees(scan_angle)

    scan_points = rawdata.shape[0]
    scan_points_side = scan_points // 2
    scan_length_side = scan_points_side * step_size

    if scan_points % 2 != 0:
        pos_vec = np.linspace(-scan_length_side, scan_length_side, scan_points)
    else:
        pos_vec = np.linspace(-scan_length_side + step_size / 2,
                              scan_length_side - step_size / 2, scan_points)

    pos = pos_vec - 0.0
    pos_cor = pos / Mag * 1e6         # mm at stage -> nm at sample

    # ---- Reconstruct the time axis ---------------------------------------
    time_end = (rawdata.shape[1] - 7) * bin_ns
    tvec = np.arange(0, time_end + bin_ns / 2, bin_ns)

    # Run the full analysis (drops the 6 metadata columns from rawdata).
    analysis_loop(rawdata[:, :-6], pos_cor, tvec, 1, outfilepath = 'results/' + outfilesuffix[i], 
                  background_subtract=False, geometry='cylindrical')