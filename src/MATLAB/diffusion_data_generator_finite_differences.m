%% overall parameters
trials = 2;
profiles_to_run = [1 2 3];
% D1 = 0.01 * 1e5; % nm^2/ns
signal_mult = 10000;
peak_snr = 30;
baseblur = 0;
back_prof_subtract = false;

%% setup parameters
% experiment parameters
pulse_time = 3; % ns
spot_std = 200; % nm
Mag = 495;

% simulation parameters
full_width = 15000; % nm
dt = 0.001; % ns
dr = 25; % nm

% collection parameters
t_bin_width = 0.01; % ns
%s_bin_width = 1e4/Mag; % nm
s_bin_width = dr;
s_col_width = 10e6/Mag;

% fit parameters
D1 =  0.1 * 1e5; % nm^2/ns
D2 = 0 * 1e5; % nm^2/ns
kd1_base = 1; % 1/ns
kd2 = 0; % 1/ns
k12 = 0; % 1/ns
k21 = 0; % 1/ns
plqy1_base = 1;
plqy2 = 1;
frac_1_init = 1;
% kd1bi = 0.01; % 1/ns * conc
% kd2bi = 0.01; % 1/ns * conc
max_photobleach = 0;

% finishing setup
spvec = -full_width:dr:full_width;

s_collect = -s_col_width:s_bin_width:s_col_width;

r = (-full_width:dr:full_width) + dr/2;
closest_space_bins = zeros(size(r));
full_collect = [s_collect(1) - s_bin_width : s_bin_width : s_collect(end)+s_bin_width];
for i = 1:length(closest_space_bins)
    [mini, ind] = min(abs(r(i) - full_collect));
    if mini <= s_bin_width/2 + s_bin_width/1E6
        closest_space_bins(i) = full_collect(ind);
    else
        closest_space_bins(i) = -1;
    end
end

profiles = zeros(3, length(r));
% gaussian profile
profiles(1, :) = exp(-(r.^2)./(2*spot_std^2));
% sort-of-dirac delta profile
% profiles(1, :) = zeros(1, length(r));
% profiles(1, floor(length(r)/2)) = 1;
% profiles(1, ceil(length(r)/2)) = 1;

profiles(2, :) = genvoigt(125, 135, 0, r);
% profiles(2, :) = genvoigt(260, 1, 0, r);
airyx = 2 * pi/405 * 1500000 * r/3400000;
profiles(3, :) = (2 * besselj(1, airyx)./airyx).^2;

puredatasets = zeros([3, length(s_collect), 6 + ceil(pulse_time/t_bin_width)]);

%% simulation

for j = profiles_to_run
    puredata = zeros([length(s_collect), 6 + ceil(pulse_time/t_bin_width)]);
    puredata(:, end) = 1/(pulse_time*1e-9); % rep rate
    puredata(:, end-1) = 1; % collection time
    puredata(:, end-2) = t_bin_width * 1000; % t bin width, ps
    puredata(:, end-4) = s_collect * Mag;
    % pulse
    % p1 = frac_1_init * exp(-(r.^2)./(2*spot_std^2));
    % p2 = (1-frac_1_init) * exp(-(r.^2)./(2*spot_std^2));
    
    % p1 = genvoigt(125, 125, 0, r);
    % p1 = (2 * besselj(1, airyx)./airyx).^2;
    p1 = frac_1_init * profiles(j, :);
    p2 = (1-frac_1_init) * profiles(j, :);
    % photobleach = max_photobleach .* profiles(j, :)/max(profiles(j, :));
    enhancement_func = @(x) max(0, log(x./max(x))./log(10) + 2)/2;
    % kd1 = kd1_base .* (1 + 0.75*enhancement_func(profiles(j, :)));
    kd1 = kd1_base;
    plqy1 = plqy1_base./(kd1).*kd1_base;

    for t = 0:dt:pulse_time
        % radial 1d
        % laplac_1 = r .*([p1(2:end), 0] - [0, p1(1:end-1)])/(2 * dr);
        % laplac_1 = 1./r .* ([laplac_1(2:end), 0] - [0, laplac_1(1:end-1)])/(2 * dr);
        % laplac_2 = r .* ([p2(2:end), 0] - [0, p2(1:end-1)])/(2 * dr);
        % laplac_2 = 1./r .* ([laplac_2(2:end), 0] - [0, laplac_2(1:end-1)])/(2 * dr);
        % -------

        % cartesian 1d
        laplac_1 = ([p1(2:end), 0] + [0, p1(1:end-1)] - 2 * p1)/dr^2;
        laplac_2 = ([p2(2:end), 0] + [0, p2(1:end-1)] - 2 * p2)/dr^2;
        % -------

        dp1 = dt * (D1 * laplac_1 - (kd1 + k12).*p1 + k21*p2);
        dp2 = dt * (D2 * laplac_2 - (kd2 + k21)*p2 + k12*p1);
        % dp2 = zeros(size(p2));

        pL = dt .* (kd1 .* p1 .* plqy1 + kd2 * p2 * plqy2);

        % for higher order recombination terms
        % pL = pL + dt * (kd1bi * p1.^2 * plqy1_bi + kd2bi * p2.^2 * plqy2_bi);
        % dp1 = dp1 - dt * kd1bi * p1.^2;
        % dp2 = dp2 - dt * kd2bi * p2.^2;
        % ----

        pL_binned = zeros(length(s_collect), 1);
        % bin in space
        for i = 1:length(s_collect)
            % convert data to format used for real data collection
            pL_binned(i) = sum(pL(abs(closest_space_bins-s_collect(i))<1e-9));       
        end
        t_index = floor((t + dt/1E6)/t_bin_width)+1;
        if t_index <= length(puredata(1, :))-6
            puredata(:, t_index) = puredata(:, t_index) + pL_binned * signal_mult;
        end
        % adding photobleaching?
        % dp1 = dp1 + dp1 .* photobleach;
        % dp2 = dp2 + dp2 .* photobleach;

        p1 = p1 + dp1;
        p2 = p2 + dp2;
    end
    puredatasets(j, :, :) = puredata;
    disp("Simulation " + j + " done!");
end

diffs_trials = zeros(trials, 3, 7);
diff_uncerts_trials = zeros(trials, 3, 7);
msds_trials = zeros(trials, 3, 7, size(puredatasets, 3)-6);
%% run trials
for trialcount = 1:trials
    datasets = puredatasets;
    
    % add noise
    % datasets(:, :, 1:end-6) = poissrnd(datasets(:, :, 1:end-6)./max(datasets(:, :, 1:end-6), [], "all").* peak_snr.^2);
    % datasets(:, :, 1:end-6) = round(datasets(:, :, 1:end-6)) + poissrnd(1, size(datasets(:, :, 1:end-6)));
    % datasets(:, :, 1:end-6) = round(datasets(:, :, 1:end-6));

    
    % add effect of pulses that haven't fully decayed yet
    % puredataset_count = length(datasets(:, 1, 1));
    % spat_bins = length(datasets(1, :, 1));
    % time_bins = length(datasets(1, 1, :))-6;
    % time_bins_overlap = floor(time_bins/2);
    % datasets_overlap = zeros(puredataset_count, spat_bins, time_bins_overlap + 6);
    % datasets_overlap(:, :, end-6:end) = datasets(:, :, end-6:end);
    % datasets_overlap(:, :, 1:time_bins_overlap) = datasets(:, :, 1:time_bins_overlap);
    % datasets_overlap(:, :, 1:time_bins_overlap) = datasets_overlap(:, :, 1:time_bins_overlap) + ...
    %                                               datasets(:, :, 1 + time_bins_overlap : 2 * time_bins_overlap);
    % datasets = datasets_overlap;

    % add temporal irf
    % bin = max(datasets(:, 2, end-2))/1000; % unit of ns
    % time_end=(size(puredatasets,3)-7)*bin; %end time of measurement
    % tvec = 0:bin:time_end; % time vector
    % % time_irf = 0.895*exp(-tvec/0.05)+0.1*exp(-tvec/0.3)+0.005*exp(-tvec/3);
    % time_irf = exp(-tvec/0.03);
    % time_irf = time_irf/sum(time_irf);
    % for i = 1:length(datasets(:, 1, 1))
    %     data = squeeze(datasets(i, :, 1:end-6));
    %     tpoints = length(data(1, :));
    %     conv_data = conv2(data, time_irf);
    %     conv_data = conv_data(:, 1:tpoints);
    %     datasets(i, :, 1:end-6) = conv_data;
    % end

    % add apd irf
    % apdIRFsamplesize = 50000/495;
    % fulldRincrements = floor(apdIRFsamplesize/dr);
    % residual = apdIRFsamplesize - dr * fulldRincrements;
    % apd_spat_irf = [residual/2; ones(fulldRincrements, 1); residual/2];
    % apd_spat_irf = apd_spat_irf/sum(apd_spat_irf);
    % for i = 1:length(datasets(:, 1, 1))
    %     data = squeeze(datasets(i, :, 1:end-6));
    %     spoints = length(data(:, 1));
    %     conv_data = conv2(data, apd_spat_irf);
    %     conv_data = conv_data(1:spoints, :);
    %     datasets(i, :, 1:end-6) = conv_data;
    % end

    % limit spatial extent of dataset
    spat_cent_ind = round(length(datasets(1, :, 1))/2);
    spat_side_bins = 100;
    datasets = datasets(:, spat_cent_ind - spat_side_bins : spat_cent_ind + spat_side_bins, :);


    datasets(:, :, end-3) = sum(datasets(:, :, 1:end-6), 3);
    
    lifetime_fit_times = [1 1 1] * 1;
    diff_fit_times = [1 1 1] * 1;
    datanames = ["fakedata_gauss", "fakedata_voigt", "fakedata_airy", "fakedata_asymmetric"];
    outprefix = pwd + "\results\";
    outfilepaths = outprefix + ["cart_gaussian\", "cart_voigt\", "cart_airy\"];

    for i = profiles_to_run
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
    
        % Run the full analysis.
        outfilepath = outfilepaths(i);
        [diffs, diff_uncerts, msds] = analysis_loop(rawdata(:, 1:end-6), pos_cor, tvec, 1, 'outfilepath', outfilepath, 'background_subtract', false, 'geometry', 'cartesian');
        diffs_trials(trialcount, i, :) = diffs;
        diff_uncerts_trials(trialcount, i, :) = diff_uncerts;
        msds_trials(trialcount, i, :, 1:size(msds, 2)) = msds;
        save(outfilepath + "output.mat");

    end
end    
    
%% helper functions
function voigt = genvoigt(gamma, sigma, center, xvec)
    zvector = (xvec - center + i * gamma)/(sigma * sqrt(2));
    voigt = real(fadf(zvector))/(sigma * sqrt(2 * pi));
    voigt = voigt/max(voigt);
end