% DIFFIMG_ANALYZER_RECIPROCAL  Example driver script.
%
% Loads a test dataset, reconstructs the position and time axes from the
% metadata packed into the raw data array, and calls analysis_loop on each
% dataset. Use this as a template for wiring your own data into the pipeline.
%
% NOTE: This script is specific to a particular acquisition file format.
% Adapt these indices to your own data layout.

clear
parent_dir = fileparts(pwd);
load(parent_dir + "\testdata\testdata_cyl_0p05.mat", "puredatasets");   % test data; dims: [dataset, position, time(+metadata)]
Mag = 495;                                        % optical magnification (used to convert stage -> sample nm)
for i = 1:3                                        % loop over selected datasets
    close all
    rawdata = squeeze(puredatasets(i, :, :));

    % Crop to a symmetric window of +/- spat_side_bins around the center row.
    spat_cent_ind = ceil(size(rawdata, 1)/2);
    spat_side_bins = 100;
    rawdata = rawdata(spat_cent_ind - spat_side_bins:spat_cent_ind + spat_side_bins, :);

    % ---- Metadata packed into trailing columns of rawdata ----
    Rep = rawdata(2, end);              % repetition rate (Hz)
    AcqTime = rawdata(2, end-1);        % acquisition time (s)
    bin = rawdata(2, end-2) / 1000;     % time-bin width: ps -> ns
    lifetimes_sum = rawdata(:, end-3);  % (time-integrated counts per position)

    %% Reconstruct the position axis from scan geometry --------------------
    posdiffs = flip(rawdata(2, end-5:end-4) - rawdata(1, end-5:end-4)); % per-step stage displacement (x,y)
    step_size = norm(posdiffs) / 1e6;   % step length -> mm

    % Scan angle (informational); folded into [0, 90) degrees.
    scan_angle = atan(posdiffs(2) / posdiffs(1));
    if scan_angle < 0
        scan_angle = scan_angle + pi/2;
    end
    scan_angle = scan_angle * 180 / pi;

    scan_points = length(rawdata(:, 1));
    scan_points_side = floor(scan_points / 2);
    scan_length_side = scan_points_side * step_size;

    % Build a symmetric position vector (handles odd/even point counts).
    pos_center = 0;
    if mod(scan_points, 2) ~= 0
        pos_vec = linspace(-scan_length_side, scan_length_side, scan_points);
    else
        pos_vec = linspace(-scan_length_side + step_size/2, scan_length_side - step_size/2, scan_points);
    end

    pos = pos_vec - pos_center;
    pos_cor = pos' / Mag * 1000000;      % mm at stage -> nm at sample (via magnification)

    %% Reconstruct the time axis -------------------------------------------
    time_end = (size(rawdata, 2) - 7) * bin;   % metadata columns excluded
    tvec = [0:bin:time_end]';

    % Run the full analysis (drops the 6 metadata columns from rawdata).
    analysis_loop(rawdata(:, 1:end-6), pos_cor, tvec, 1, 'background_subtract', 'estimate', 'geometry', 'cylindrical')
end