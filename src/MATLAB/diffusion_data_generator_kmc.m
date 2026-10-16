%% overall parameters
trials = 1;
nexcitons = 100000;
profiles_to_run = [1];
% D1 = 0.01 * 1e5; % nm^2/ns
back_prof_subtract = false;
rng("default")

%% setup parameters
% experiment parameters
pulse_time = 3.1; % ns
spot_std = 200; % nm
Mag = 495;

% simulation parameters
full_width = 6000; % nm
dr = 5; % nm

% collection parameters
t_bin_width = 0.01; % ns
%s_bin_width = 1e4/Mag; % nm
s_bin_width = 5*dr;
s_col_width = full_width;

% material parameters
kd1 = 1; % 1/ns

%% finishing setup
spvec = -full_width:dr:full_width;
s_collect = -s_col_width:s_bin_width:s_col_width;
r = (-full_width:dr:full_width) + dr/2;

profiles = zeros(4, length(r));
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
profiles(4, :) = genvoigt(125, 130, 0, r) + 0.35 * genvoigt(150, 50, 400, r);

puredatasets = zeros([3, length(s_collect), 6 + ceil(pulse_time/t_bin_width)]);

%% simulation
for j = profiles_to_run
    puredata = zeros([length(s_collect), 6 + ceil(pulse_time/t_bin_width)]);
    puredata(:, end) = 1/(pulse_time*1e-9); % rep rate
    puredata(:, end-1) = 1; % collection time
    puredata(:, end-2) = t_bin_width * 1000; % t bin width, ps
    puredata(:, end-4) = s_collect * Mag;

    % pulse
    init_xs = initialize_population(r, profiles(j, :), nexcitons)';

    % --------------------- Parameters ---------------------
    max_hops   = pulse_time * kd1 * 10;           % safety cap on # hops

    t_eval  = (0:(ceil(pulse_time/t_bin_width)-1)) * t_bin_width;

    % --------- Inverse CDF table for PL wavelength sampling ---------
    lambda_grid = linspace(500, 900, 2001);   % [nm]
    P_vals      = PL_spectrum(lambda_grid);
    P_cdf       = cumtrapz(lambda_grid, P_vals);
    P_cdf       = P_cdf / P_cdf(end);
    % Make CDF strictly monotonic for interp1
    [P_cdf_u, iu] = unique(P_cdf);
    lambda_u      = lambda_grid(iu);

    % --------------------- KMC loop ---------------------
    all_t   = cell(nexcitons,1);
    all_pos = cell(nexcitons,1);

    for i = 1:nexcitons
        t   = 0;
        xpos = init_xs(i);
        ts  = zeros(1,max_hops+1);   ts(1)  = 0;
        ps  = zeros(max_hops+1,1);   ps(1) = xpos;
        nh  = 0;
        while true
            nh = nh + 1;

            % --- 1) decay time, rejection-free (inverse CDF, exponential)
            dt  = -1/kd1*log(rand);
            t   = t + dt;

            % --- 2) PL wavelength, rejection-free (inverse CDF)
            lam = interp1(P_cdf_u, lambda_u, rand, 'linear');

            % --- 3) propagation distance via Beer-Lambert absorption
            alpha_abs = absorption_spectrum(lam);    % [1/um]
            d         = -log(rand)/alpha_abs * 1000;        % [nm]

            % --- 4) pick x component of isotropic distribution function
            dir = 2*rand - 1;

            new_pos = xpos + d*dir;

            ts(nh+1)   = t;
            ps(nh+1,:) = new_pos;

            % --- 5) boundary check
            outside = (abs(new_pos)>full_width);
            if outside || nh >= max_hops || t > pulse_time
                break
            end
            xpos = new_pos;
        end
        all_t{i}   = ts(1:nh+1);
        all_pos{i} = ps(1:nh+1,:);
    end

    % -- Reconstruct spatial profile at evaluation times to make dataset --
    pos_at_t = nan(nexcitons, length(t_eval));

    for i = 1:nexcitons
        ts = all_t{i};
        ps = all_pos{i};
        for k = 1:length(t_eval)
            tk = t_eval(k);
            if tk < ts(end)
                idx = find(ts <= tk, 1, 'last');
                pos_at_t(i,k) = ps(idx,1);
            end
        end
    end
    
    hist_edges = [s_collect - s_bin_width/2, s_collect(end)+s_bin_width/2];

    for i = 1:length(t_eval)
        puredata(:, i) = histcounts(pos_at_t(:, i), hist_edges);
    end

    puredatasets(j, :, :) = puredata;
    disp("Simulation " + j + " done!");
end
%%
diffs_trials = zeros(trials, 3, 7);
diff_uncerts_trials = zeros(trials, 3, 7);
msds_trials = zeros(trials, 3, 7, size(puredatasets, 3)-6);

lifetime_fit_times = [1 1 1 1] * 1;
diff_fit_times = [1 1 1 1] * 1;
datanames = ["fakedata_gauss", "fakedata_voigt", "fakedata_airy", "fakedata_asymmetric"];
outprefix = pwd + "\results\";
outfilepaths = outprefix + ["pr_gaussian\", "pr_voigt\", "pr_airy\", "pr_asymmetric\"];

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
    spat_side_bins = 200;
    datasets = datasets(:, spat_cent_ind - spat_side_bins : spat_cent_ind + spat_side_bins, :);


    datasets(:, :, end-3) = sum(datasets(:, :, 1:end-6), 3);    

%% data analysis
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

% =====================================================================
% Spectra and helper functions
% =====================================================================
function x_start = initialize_population(x, pdf, nwalkers)
    cdf = cumtrapz(x, pdf);
    cdf = cdf/cdf(end);
    [cdf_unique, iunique] = unique(cdf);
    x_unique = x(iunique);

    u = rand([nwalkers, 1]);
    x_start = interp1(cdf_unique, x_unique,u, 'linear', 0);
end

function alpha_abs = absorption_spectrum(lambda)
% Absorption coefficient [1/um] vs wavelength [nm]
% Sigmoidal band edge plus a small Urbach-like tail.
    lam_edge = 725;
    width    = 25;
    a0       = 5;        % above-gap absorption [1/um]
    % % logistic
    % alpha_abs = a0 ./ (1 + exp((lambda - lam_edge)/width));

    % % gaussian
    alpha_abs = zeros(size(lambda));
    alpha_abs = a0 .* exp(-(lambda-lam_edge).^2/(2*width^2));
    alpha_abs(lambda <= lam_edge) = a0;

    % % complete redistribution
    % alpha_abs = PL_spectrum(lambda) * a0;
end

function p = PL_spectrum(lambda)
% Gaussian PL spectrum centered below the absorption edge (Stokes shift)
    lam0  = 775;
    sigma = 18;
    p = exp(-(lambda - lam0).^2/(2*sigma^2));
end