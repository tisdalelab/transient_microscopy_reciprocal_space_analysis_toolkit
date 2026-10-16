function [diffs, diff_uncerts, msds] = analysis_loop(rawdata, xvec, tvec, analysis_end_time, varargin)
% ANALYSIS_LOOP  Extract carrier-transport parameters from transient
%                (time- and space-resolved) microscopy data.
%
% This is the main analysis driver. Given a 2D dataset of intensity as a
% function of position and time, it:
%   1.  Produces diagnostic plots (spatially-resolved PL, time slices, 2D maps).
%   2.  Fits the spatially-integrated decay (TRPL) to a biexponential.
%   3.  Performs a reciprocal-space (Fourier / Hankel) spatial-mode-decay
%       analysis and fits decay-rate vs. k to a power law (diffusive -> slope 2).
%   4.  Estimates the mean-squared displacement (MSD) of the profile at every
%       time point using SEVEN independent methods (see below).
%   5.  Converts MSD-vs-time slopes into diffusivities and diffusion lengths.
%   6.  Computes the excess kurtosis of the profile vs. time.
%   7.  Saves all intermediate/derived quantities to <outfilepath>keyValues.mat.
%
% ---------------------------------------------------------------------------
% REQUIRED INPUTS
%   rawdata           [Nx x Nt] matrix. Rows = spatial positions, columns =
%                     time bins. rawdata(i,j) is intensity at position i, time j.
%   xvec              [Nx x 1] vector of positions (nm).
%   tvec              [Nt x 1] vector of times (ns).
%   analysis_end_time scalar (ns). Data beyond this time are not analyzed
%                     (used as the default cutoff for every sub-fit and as the
%                     x-limit of most plots).
%
% OPTIONAL NAME-VALUE PARAMETERS
%   'outfilepath'         Output directory for figures / .mat  (default: pwd).
%   'diff_fit_time'       End time (ns) for the linear MSD-vs-t diffusivity fit.
%   'lifetime_fit_time'   End time (ns) for the TRPL biexponential fit.
%   'freq_fit_time'       End time (ns) for the reciprocal-space decay fits.
%   'geometry'            "cartesian" (1D, use FFT) or "cylindrical"
%                         (2D radial, use Hankel transform). Default "cartesian".
%   'background_subtract' false | numeric constant | numeric vector | "estimate".
%                         false applies no background subtraction to the
%                         profiles. 
%                         numeric constant or numeric vector will subtract the 
%                         given constant/vector from all profiles
%                         "estimate" derives a background profile from the pre-rise
%                         time window automatically.
%   'profile_center'      "find" (locate the peak automatically) or a numeric
%                         position used to center the spatial axis. Default 0.
%   'max_k_plot'          Largest k (1/nm) to include in reciprocal-space plots.
%   'baseblur'            Baseline blur std-dev (nm) removed in the numerical-
%                         convolution MSD method.
%
% OUTPUTS
%   diffs         [7x1] diffusivity from each MSD method (cm^2/s).
%   diff_uncerts  [7x1] corresponding uncertainties.
%   msds          [7 x Nt_fit] MSD vs time for each method (nm^2), zeroed at t=0.
%
% EXAMPLE
%   analysis_loop(data, pos_nm, time_ns, 1, ...
%                 'background_subtract','estimate','geometry','cylindrical');
%
% DEPENDENCIES: gaussblur, genvoigt (-> fadf), profilefit_1d,
%               calculate_kurtosis_excess, hankel_transformer, C.mat (colormap).
% ---------------------------------------------------------------------------

    % ---- Parse optional name-value arguments -----------------------------
    p = inputParser;
    addParameter(p,  'outfilepath',  pwd);                    % output directory
    addParameter(p,  'diff_fit_time',  analysis_end_time);    % positive float (ns)
    addParameter(p,  'lifetime_fit_time',  analysis_end_time);% positive float (ns)
    addParameter(p,  'freq_fit_time',  analysis_end_time);    % positive float (ns)
    addParameter(p,  'geometry', "cartesian");                % "cartesian" | "cylindrical"
    addParameter(p,  'background_subtract', false);           % false|const|vector|"estimate"
    addParameter(p,  'profile_center', 0);                    % "find" | numeric center
    addParameter(p,  'max_k_plot', 0.02);                     % positive float (1/nm)
    addParameter(p,  'baseblur',  0);                         % positive float (nm)

    parse(p, varargin{:})

    outfilepath         = p.Results.outfilepath;
    diff_fit_time       = p.Results.diff_fit_time;
    lifetime_fit_time   = p.Results.lifetime_fit_time;
    freq_fit_time       = p.Results.freq_fit_time;
    geometry            = p.Results.geometry;
    background_subtract = p.Results.background_subtract;
    profile_center      = p.Results.profile_center;
    max_k_plot          = p.Results.max_k_plot;
    baseblur            = p.Results.baseblur;

    % Force column vectors so indexing/broadcasting is consistent.
    tvec = tvec(:);
    xvec = xvec(:);

    % Create output folder + a subfolder for editable .fig files.
    mkdir(outfilepath);
    mkdir(fullfile(outfilepath, "figs\"))

    % Time-integrated intensity at each spatial position (sum over time bins).
    lifetimes_sum = sum(rawdata, 2);

    pulse_time = max(tvec); % time between excitation pulses (ns) = length of the trace

    % Load the custom colormap used for the 2D maps.
    C=load('C.mat');
    Cmap=C(1).C;

    % keyValues accumulates every derived quantity; saved to disk at the end.
    keyValues = struct();

    %% Preliminary looks at dataset -----------------------------------------
    % Spatially-resolved PL: counts vs position (integrated over all time).
    [xData, yData] = prepareCurveData(xvec, lifetimes_sum);
    keyValues.summed_space_x_pos_nm = xData;
    keyValues.summed_space_y_cts_arb = yData;

    % Plot counts-vs-position.
    figure
    h = plot(xData, yData );
    xlabel('position (nm)')
    ylabel('counts')
    axis([min(xvec) max(xvec) 0 max(yData)*1.2])
    title('spatially-resolved PL')
    print(gcf, fullfile(outfilepath, 'cts vs pos.png'),'-dpng')

    % ---- Optionally locate the profile center automatically --------------
    % Blur the time-integrated profile, find the coarse peak, then refine the
    % peak position with a spline + constrained optimizer.
    if string(profile_center) == "find"
        blurred_prof_total = gaussblur(xvec, lifetimes_sum, 10 * (xvec(2)-xvec(1))^2);
        opts = optimoptions('fmincon', 'Display', 'off');
        [~, max_ind] = max(blurred_prof_total);
        profile_center = fmincon(@(xprobe) -spline(xvec, blurred_prof_total, xprobe), xvec(max_ind), [], [], [], [], min(xvec), max(xvec), [], opts);
    end
    % Center the spatial axis on the profile center.
    pos_cor = xvec - profile_center;


    % ---- Plot five normalized spatial profiles at evenly spaced times ----
    end_ind_fit = find(tvec <analysis_end_time, 1, "last"); % last time index we analyze
    profiles_to_plot = round(linspace(1, end_ind_fit,5));
    colors = [0.1 0.1 0.8;
        0.1 0.3 0.6;
        0.1 0.5 0.4;
        0.1 0.7 0.2;
        0.1 1.0 0.0];

    figure
    hold on
    for i = 1:5
        plot(pos_cor, rawdata(:, profiles_to_plot(i))./max(rawdata(:, profiles_to_plot(i))) , 'LineWidth', 1.5, 'Color', colors(i, :), ...
            'DisplayName', sprintf("%.2f ns", tvec(profiles_to_plot(i))));
    end
    hold off
    ax = gca;
    ax.FontSize = 14;

    legend('Location', 'northeast', 'FontSize', 14, 'Color', 'w', 'Box', 'off')
    xlabel('position (nm)', 'FontSize', 15)
    ylabel('normalized counts', 'FontSize', 15)
    axis([min(pos_cor) max(pos_cor) 0 1.2])
    set(gca, 'FontName', 'Arial', 'LineWidth', 2, 'Box', 'on')
    set(gcf, 'Color', 'w');
    box on
    print(gcf,fullfile( outfilepath, 'emission_time_slices.png'),'-dpng')
    savefig(gcf, fullfile(outfilepath,  'figs\emission_time_slices.fig'))


    % ---- Spatially integrated decay (TRPL) -------------------------------
    TRPL = [sum(rawdata, 1)]';   % sum over positions -> intensity vs time
    TRPL = TRPL / max(TRPL);     % normalize to peak
    keyValues.summed_time_norm_intensities = TRPL;

    % Fit TRPL to a normalized biexponential (+ constant background k) in log
    % space. Coeffs: a = fast-component fraction, b = tau1, d = tau2, k = bkg.
    maxfittindex = find(tvec < lifetime_fit_time, 1, "last");
    [xData, yData] = prepareCurveData(tvec(1:maxfittindex), log(TRPL(1:maxfittindex)));
    ft = fittype( '-x/b', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Lower = 0;
    opts.Upper = 100;       % NOTE: background k is pinned to 0 here.
    opts.StartPoint = 1;
    [fitresult, gof] = fit( xData, yData, ft, opts );
    coeffs = coeffvalues(fitresult);
    confs = confint(fitresult);
    uncert = confs(2,:)-coeffs;       % 95% CI half-width as an uncertainty
    keyValues.lifetime = coeffs;
    keyValues.lifetime_uncert = uncert;

    disp("lifetime: " + keyValues.lifetime + " ns")


    % Evaluate the fitted decay on a fine grid for overlay plotting.
    x1 = 0.01:0.1:(pulse_time-2);
    y1= exp(-x1./coeffs);

    figure
    hold on
    plot(tvec, TRPL, '.k')
    plot(x1,y1,'r-','LineWidth',1)
    hold off
    xlabel('time (ns)')
    ylabel('normalized PL')
    set(gca, 'YScale', 'log')
    axis([0 max(1, round(pulse_time-2)) 0.0001 1.2])
    box on
    title('TRPL')
    print(gcf, fullfile(outfilepath, 'cts vs time.png'),'-dpng')

    %% 2D plots (intensity vs position and time) ---------------------------
    figure;
    surface(pos_cor, tvec, rawdata','EdgeColor','none','CDataMapping','scaled');
    colorbar; colormap(Cmap);
    axis([min(pos_cor) max(pos_cor) 0 analysis_end_time])
    xlabel('position (nm)')
    ylabel('time (ns)')
    print(gcf, fullfile(outfilepath , '2D_map.png'),'-dpng')

    % Same map but each time slice normalized to its own max (shows spreading).
    lifetimes_normtomax = rawdata ./ max(rawdata, [], 1);
    figure;
    surface(pos_cor, tvec, lifetimes_normtomax','EdgeColor','none','CDataMapping','scaled');
    colorbar; colormap(Cmap);
    set(gca, 'CLim', [0, 1]);
    axis([min(pos_cor) max(pos_cor) 0 analysis_end_time])
    xlabel('position (nm)')
    ylabel('time (ns)')
    print(gcf, fullfile(outfilepath ,  '2D_normtomax.png'),'-dpng')
    savefig(gcf, fullfile(outfilepath , 'figs\2D_normtomax.fig'))

    %% Data processing: background subtraction -----------------------------
    back_prof = zeros(size(rawdata, 1), 1);
    if string(background_subtract) == "estimate"
        % Walk backwards from the end of the trace to find where the signal
        % starts rising above the local mean (the pre-pulse rise onset), then
        % average a window before that as the background spatial profile.
        rise_begin = length(TRPL);
        while TRPL(rise_begin) > mean(TRPL(rise_begin - 5: rise_begin-1))
            rise_begin = rise_begin-1;
        end
        back_prof = mean(rawdata(:, rise_begin-250:rise_begin - 50), 2);
    elseif background_subtract
        % A supplied constant or vector background.
        back_prof = back_prof + background_subtract;
    end

    %% Reciprocal-space spatial-mode-decay analysis ------------------------
    % Diffusion decouples spatial Fourier/Hankel modes: each mode k should
    % decay exponentially at rate Gamma(k) = D*k^2 + 1/tau. Fitting Gamma(k)
    % vs k (log-log) gives slope 2 for pure diffusion and prefactor ~ D.
    dx = pos_cor(2) - pos_cor(1);
    k_fourier = (0 : 2 * pi/(length(pos_cor) * dx) : 2 * pi/(length(pos_cor) * dx) * (length(pos_cor) - 1))';
    fft_all = real(fft(ifftshift(rawdata - back_prof, 1), [], 1));  % FFT along position

    % Quasi-discrete Hankel transform (for radially symmetric / cylindrical data).
    hankel_txer = hankel_transformer(pos_cor);
    [hankel_all, k_hankel] =  hankel_transformer.ht(rawdata - back_prof, hankel_txer);
    % calculate k = 0 value separately
    xData_left = pos_cor(pos_cor<=0);
    xData_right = pos_cor(pos_cor>=0);
    quad_weights_right = pi./8.*[sqrt(0.5).*(xData_right(1) + xData_right(2)).^2 ; ...
        (xData_right(3:end) - xData_right(1:end-2)).*(xData_right(1:end-2) + 2.*xData_right(2:end-1) + xData_right(3:end)) ; ...
        4.*xData_right(end).^2 - (xData_right(end-1) + xData_right(end)).^2];
    xData_left_quadest = -flip(xData_left);
    quad_weights_left = pi./8.*[sqrt(0.5).*(xData_left_quadest(1) + xData_left_quadest(2)).^2 ; ...
        (xData_left_quadest(3:end) - xData_left_quadest(1:end-2)).*(xData_left_quadest(1:end-2) + 2.*xData_left_quadest(2:end-1) + xData_left_quadest(3:end)) ; ...
        4.*xData_left_quadest(end).^2 - (xData_left_quadest(end-1) + xData_left_quadest(end)).^2];
    quad_weights_left = flip(quad_weights_left);

    prof_left = rawdata(pos_cor<=0, :) - back_prof(pos_cor <= 0);
    prof_right = rawdata(pos_cor>=0, :) - back_prof(pos_cor >= 0);

    hankel_k0 = sum(quad_weights_right.*prof_right) + sum(quad_weights_left.*prof_left);



    % Double-normalize: divide out the t=0 spectrum AND the k=0 decay so that
    % every curve starts at 1, isolating the k-dependent decay.
    if string(geometry) == "cartesian"
        recip_double_normalized = (fft_all ./ fft_all(1, :)) ./ fft_all(:, 1) * fft_all(1, 1) ;
        k_vec = k_fourier;
    elseif string(geometry) == "cylindrical"
        recip_double_normalized = (hankel_all ./ hankel_k0) ./ hankel_all(:, 1) * hankel_k0(1) ;
        k_vec = k_hankel;
    end

    fourier_colors = @(n) hsv(n);

    % ---- Fit an exponential decay rate at each spatial frequency k -------
    fourier_end_ind_fit = find(tvec < freq_fit_time, 1, "last");
    tvec_fourier_fit = tvec(1:fourier_end_ind_fit);
    if (max(k_vec) > max_k_plot )
        k_max_ind_plot = find(k_vec>max_k_plot, 1)-1;
    else
        k_max_ind_plot = length(k_vec);
    end

    ft = fittype('a*x + b', 'independent', 'x', 'dependent', 'y');          % linear fit in log-space
    ft_noisy = fittype('exp(x.*a) * b', 'independent', 'x', 'dependent', 'y'); % fallback for noisy (sign-changing) data
    opts = fitoptions('Method', 'NonlinearLeastSquares');
    opts.StartPoint = [0 0];

    spatial_decay_rate = zeros(k_max_ind_plot, 1);
    spatial_decay_rate_uncerts = zeros(k_max_ind_plot, 1);
    spatial_decay_fit_r2s = zeros(k_max_ind_plot, 1);
    all_valid = true(k_max_ind_plot, 1);          % track which k had strictly positive data
    decay_fit_coeffs = zeros(k_max_ind_plot, 2);

    for i = 2:k_max_ind_plot   % skip i=1 (k=0, the DC/global-decay mode)
        yData = recip_double_normalized(i, 1:fourier_end_ind_fit)';
        % If any value is <= 0, log() is complex -> that k is flagged "not valid"
        % and fit in linear space instead of log space.
        valid_inds = imag(log(yData)) == 0;
        if any(~valid_inds)
            all_valid(i) = false;
        end
        if all_valid(i)
            % Clean case: linear fit of log(intensity) vs time; slope = -Gamma.
            [fitresult, gof] = fit(tvec_fourier_fit, real(log(yData)), ft, opts);
            coeffs = coeffvalues(fitresult);
            confs = confint(fitresult);
            uncert = confs(2, :) - coeffs;
            rsquared = struct2cell(gof);

            spatial_decay_rate(i) = -coeffs(1);
            spatial_decay_rate_uncerts(i) = uncert(1);
            spatial_decay_fit_r2s(i) = cell2mat(rsquared(2));  % adjusted R^2
            decay_fit_coeffs(i, :) = coeffs;
        else
            % Noisy case: nonlinear exponential fit directly on the data.
            [fitresult, gof] = fit(tvec_fourier_fit, yData, ft_noisy, opts);
            coeffs = coeffvalues(fitresult);
            confs = confint(fitresult);
            uncert = confs(2, :) - coeffs;
            rsquared = struct2cell(gof);

            spatial_decay_rate(i) = -coeffs(1);
            spatial_decay_rate_uncerts(i) = uncert(1);
            spatial_decay_fit_r2s(i) = cell2mat(rsquared(2));
            decay_fit_coeffs(i, :) = coeffs;
        end
    end

    % ---- Choose a subset of k values to display -------------------------
    r2_plot_threshold = 0;   % keep all fits by default; raise to filter poor fits
    good_k_mask = false(k_max_ind_plot, 1);
    good_k_mask(2:end) = (spatial_decay_fit_r2s(2:end) > r2_plot_threshold) & (k_vec(2:k_max_ind_plot) <= max_k_plot);
    good_k_inds = find(good_k_mask);

    nlines = min(length(good_k_inds), 10);  % plot at most 10 curves

    if nlines > 1
        k_inds_plot = good_k_inds(round(linspace(1, length(good_k_inds), nlines)));
    else
        k_inds_plot = good_k_inds;
        nlines = length(k_inds_plot);
    end

    % ---- Plot: normalized modal intensity vs time (per k) ----------------
    figure(Name="intensity vs k norm");
    hold on;

    colors_norm = fourier_colors(nlines);
    labels_norm = cell(1, nlines);
    h_data_norm = gobjects(nlines, 1);

    for i = 1:nlines
        ki = k_inds_plot(i);

        data_to_plot = recip_double_normalized(ki, :);
        h_data_norm(i) = semilogy(tvec, data_to_plot, '-', 'Color', colors_norm(i, :), 'MarkerSize', 5);
        labels_norm{i} = sprintf('k = %.4f nm^{-1} (R^2=%.3f)', k_vec(ki), spatial_decay_fit_r2s(ki));
    end

    set(gca, 'YScale', 'log');
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1.5, 'Box', 'on');
    xlabel('Time (ns)', 'FontSize', 14);
    ylabel('Normalized Intensity', 'FontSize', 14);
    title(sprintf('Fourier Intensity vs Time (k-normalized)'), 'FontSize', 13);
    legend(h_data_norm, labels_norm, 'Location', 'southwest', 'FontSize', 7, 'Box', 'off');
    xlim([0 analysis_end_time]);
    ax = gca;
    ax.YLim(2) = 1.1;
    print(gcf, fullfile(outfilepath, 'fourier_intensity_vs_t_norm.png'), '-dpng');
    savefig(gcf, fullfile(outfilepath, "\figs\fourier_intensity_vs_t_norm.fig"));

    % ---- Fit decay-rate vs k to a power law: Gamma = prefactor * k^exp ----
    first_invalid = find(~all_valid, 1);
    if isempty(first_invalid)
        first_invalid = k_max_ind_plot + 1;
    end

    k_vec_fits = k_vec(1:k_max_ind_plot);
    kvec_fit_valid = k_vec(all_valid);
    decay_rate_fit_valid = spatial_decay_rate(all_valid);

    % Only fit the low-k (long-wavelength) regime with highest signal.
    k_max_slope_fit = 0.01001;
    k_max_slope_fit_ind = min(k_max_ind_plot, find(k_vec>k_max_slope_fit, 1)-1);

    fit_range = 1:min(first_invalid - 1, k_max_slope_fit_ind);

    kvec_for_fit = kvec_fit_valid(fit_range);
    decay_for_fit = decay_rate_fit_valid(fit_range);

    % Require positive, finite k and decay rate for a log-log fit.
    valid_for_fit = ~isnan(decay_for_fit) & decay_for_fit > 0 & kvec_for_fit > 0;
    kvec_for_fit = kvec_for_fit(valid_for_fit);
    decay_for_fit = decay_for_fit(valid_for_fit);

    if length(kvec_for_fit) >= 3
        % Power law in log-log space: log(Gamma) = exp*log(k) + log(prefactor).
        [fitresult_pl, gof_pl] = fit(log(kvec_for_fit), log(decay_for_fit), ft, opts);
        coeffs_pl = coeffvalues(fitresult_pl);
        confs_pl = confint(fitresult_pl);
        uncert_pl = confs_pl(2, :) - coeffs_pl;
        rsquared_pl = struct2cell(gof_pl);
        r2fit_decay_rates = cell2mat(rsquared_pl(2));
        has_power_law_fit = true;
    else
        % Not enough points: assume ideal diffusive slope of 2, flag no fit.
        coeffs_pl = [2 0];
        r2fit_decay_rates = 0;
        has_power_law_fit = false;
    end

    % Reference lines for the plot: the fitted power law and an ideal slope-2 line.
    fit_line_k = linspace(0.8 * k_vec_fits(2), max(k_vec_fits(all_valid)) * 1.25, 50);
    fit_line_func = @(x) exp(coeffs_pl(1) .* log(x) + coeffs_pl(2));
    fit_line_values = fit_line_func(fit_line_k);

    slope_2_line_func = @(x) exp(2 .* (log(x) - log(k_vec_fits(2))) + log(spatial_decay_rate(2)));
    slope_2_line_values = slope_2_line_func(fit_line_k);

    % Quick diffusivity estimate from the lowest usable k: D = Gamma/k^2.
    % (1e-5 converts nm^2/ns -> cm^2/s.)
    D_from_lowest_k = spatial_decay_rate(2) / k_vec_fits(2).^2 * 1e-5;

    figure(Name="spatial decay lifetimes vs. k");
    hold on;

    % Black = valid (log-space) fits, red = flagged/noisy fits.
    errorbar(k_vec_fits(all_valid), spatial_decay_rate(all_valid), ...
        spatial_decay_rate_uncerts(all_valid), '.k', 'LineWidth', 1, 'MarkerSize', 12);
    errorbar(k_vec_fits(~all_valid), spatial_decay_rate(~all_valid), ...
        spatial_decay_rate_uncerts(~all_valid), '.r', 'LineWidth', 1, 'MarkerSize', 12);

    if has_power_law_fit
        plot(fit_line_k, fit_line_values, '-b', 'LineWidth', 1.5);
    end

    set(gca, 'XScale', 'log', 'YScale', 'log');
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1.5, 'Box', 'on', 'MinorGridLineWidth', 1);
    grid on;

    if has_power_law_fit
        fit_text = { ...
            sprintf('$$\\Gamma = k^{%.2f} \\times %.3g$$', coeffs_pl(1), exp(coeffs_pl(2))), ...
            sprintf('$$R^2 = %.4f$$', r2fit_decay_rates), ...
            sprintf('$$D = %.4f cm^2/s$$', D_from_lowest_k)};
        annotation('textbox', [0.15 0.75 0.3 0.15], 'String', fit_text, ...
            'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'none', 'FontSize', 14);
    end

    xlabel('k (1/nm)', 'FontSize', 14);
    ylabel('Spatial decay rate (1/ns)', 'FontSize', 14);
    title('Spatial Mode Decay Rate vs. k', 'FontSize', 14);
    hold off;
    print(gcf, fullfile(outfilepath, 'hankel_spatial_decay_vs_rho.png'), '-dpng');
    savefig(gcf, fullfile(outfilepath, "\figs\hankel_spatial_decay_vs_rho.fig"));

    % Store the reciprocal-space results.
    keyValues.k_vec = k_vec;
    keyValues.freq_analysis.spatial_decay_rate = spatial_decay_rate;
    keyValues.freq_analysis.spatial_decay_rate_uncerts = spatial_decay_rate_uncerts;
    keyValues.freq_analysis.fit_r2s = spatial_decay_fit_r2s;
    keyValues.freq_analysis.all_valid = all_valid;
    keyValues.freq_analysis.k_max_ind = k_max_ind_plot;
    keyValues.freq_analysis.r2_plot_threshold = r2_plot_threshold;
    keyValues.freq_analysis.decay_fit_coeffs = decay_fit_coeffs;

    keyValues.freq_analysis.power_law_exponent = coeffs_pl(1);
    keyValues.freq_analysis.power_law_prefactor = exp(coeffs_pl(2));
    keyValues.freq_analysis.power_law_r2 = r2fit_decay_rates;
    keyValues.freq_analysis.has_power_law_fit = has_power_law_fit;

    %% Setup for MSD extraction (seven methods) ----------------------------
    % Rows of `msds`/`rsq` correspond to:
    %   1 Gaussian fit          (variance c)
    %   2 Voigt fit             (Gaussian part of a Voigt; Lorentzian gamma fixed)
    %   3 Numerical convolution (blur the initial profile to match later ones)
    %   4 Numerical integration, 1D (2nd central moment)
    %   5 Numerical integration, 2D/radial (moment estimate for cylindrical case)
    %   6 Fourier fit           (Gaussian falloff of the FFT)
    %   7 Hankel fit            (Gaussian falloff of the Hankel transform)
    clear gauss_coeffs gauss_confs voigt_coeffs voigt_confs num_conv_coeffs num_conv_confs
    clear num_int_coeffs fourier_fit_coeffs fourier_fit_uncerts
    clear hankel_fit_coeffs hankel_fit_uncerts uncert rsq msds
    clear gamma init_prof

    msds = zeros(7, end_ind_fit);

    gauss_coeffs = zeros([end_ind_fit, 3]);
    gauss_uncerts = zeros([end_ind_fit, 3]);

    voigt_coeffs = zeros([end_ind_fit, 3]);
    voigt_uncerts = zeros([end_ind_fit, 3]);

    num_conv_coeffs = zeros([end_ind_fit, 2]);
    num_conv_uncerts = zeros([end_ind_fit, 2]);

    num_int_coeffs = zeros([2, end_ind_fit]);

    fourier_fit_coeffs = zeros([end_ind_fit, 2]);
    fourier_fit_uncerts = zeros([end_ind_fit, 2]);

    hankel_fit_coeffs = zeros([end_ind_fit, 2]);
    hankel_fit_uncerts = zeros([end_ind_fit, 2]);

    rsq = zeros(7, end_ind_fit);

    times = zeros(1, 7);   % accumulate wall-clock time per method (profiling)
    wb = waitbar(0, 'Fitting profiles...');
    for k = 1:end_ind_fit % loop over time index
        % Background-subtracted spatial profile at this time.
        spec = rawdata(:, k) - back_prof;
        spec = spec(:);
        [xData, rawyData] = prepareCurveData(pos_cor, spec);

        % ----- Method 1: Gaussian fit -----
        % a*exp(-(x-b)^2 / (2c)); MSD is the variance c.
        gausstime = tic;
        ft = fittype( 'a*exp(-((x-b)^2)/(2*c))', 'independent', 'x', 'dependent', 'y' );
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.Display = 'Off';
        opts.Lower = [0 -100 0];
        opts.Upper = [Inf 100 40e6];
        if k == 1
            opts.StartPoint = [max(rawyData) 0 250000];
        else
            % Warm-start from the previous time point's fit for stability/speed.
            opts.StartPoint = reshape(gauss_coeffs(k-1, :), [3 1]);
        end
        [fitresult, gof] = fit( xData, rawyData, ft, opts );
        gauss_coeffs(k,:) = coeffvalues(fitresult);
        confs = confint(fitresult);
        gauss_uncerts(k,:) = reshape(confs(2,:), [1 3])-gauss_coeffs(k,:);
        rsquared = struct2cell(gof);
        rsq(1, k) = cell2mat(rsquared(2));
        msds(1, k) = gauss_coeffs(k, 3);
        times(1) = times(1) + toc(gausstime);

        % ----- Method 2: Voigt fit -----
        % A Voigt is a Gaussian convolved with a Lorentzian. The Lorentzian
        % width gamma is treated as a fixed instrument/intrinsic parameter
        % estimated from the FIRST (t=0) profile, then held constant so that
        % only the Gaussian variance (the diffusive part) varies with time.
        voigttime = tic;
        if k == 1
            % First frame: fit gamma, sigma^2, center and constant together.
            ft = fittype( @(a, b, c2, d, x) a*genvoigt(b, sqrt(c2), d, x), 'independent', 'x', 'dependent', 'y' );
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            sigma2_guess = 250000;
            gamma_guess = 500;
            constant_guess = 1;
            opts.Display = 'Off';
            opts.Lower = [0 0 0 -100];
            opts.StartPoint = [constant_guess  gamma_guess sigma2_guess 0];
            opts.Upper = [Inf 2000 40e6 100];
            [fitresult, gof] = fit( xData, (rawyData)/max(rawyData), ft, opts);
            all_coeffs = coeffvalues(fitresult);
            gamma = all_coeffs(2);              % fixed Lorentzian width for all later frames
            voigt_coeffs(k,:) = all_coeffs([1 3 4]);
            confs = confint(fitresult);
            voigt_uncerts(k,:) = reshape(confs(2,[1 3 4]), [1 3])-voigt_coeffs(k,:);
            rsquared = struct2cell(gof);
            rsq(2, k) = cell2mat(rsquared(2));
            msds(2, k) = voigt_coeffs(k, 2);    % Gaussian variance = MSD contribution
        else
            % Later frames: gamma is a fixed "problem" parameter.
            ft = fittype( @(a, b2, c, gamma, x) a*genvoigt(gamma, sqrt(b2), c, x), 'independent', 'x', 'dependent', 'y', 'problem', 'gamma');
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            opts.Lower = [0 0 -100];
            opts.Upper = [Inf 40e6 100];
            opts.StartPoint = reshape(voigt_coeffs(k-1, :), [3 1]);
            [fitresult, gof] = fit( xData, (rawyData)/max(rawyData), ft, opts, 'problem', gamma);
            voigt_coeffs(k,:) = coeffvalues(fitresult);
            confs = confint(fitresult);
            voigt_uncerts(k,:) = reshape(confs(2,:), [1 3]) - voigt_coeffs(k,:);
            rsquared = struct2cell(gof);
            rsq(2, k) = cell2mat(rsquared(2));
            msds(2, k) = voigt_coeffs(k, 2);
        end
        times(2) = times(2) + toc(voigttime);

        % ----- Method 3: Numerical convolution -----
        % Model each later profile as the INITIAL profile blurred by a Gaussian
        % of variance sigma2. That added variance IS the MSD (diffusive spread),
        % with no assumption about the profile shape. baseblur accounts for a
        % baseline instrument blur and is subtracted back out afterward.
        proftime = tic;
        if k == 1
            init_prof = rawyData;              % reference profile at t=0
            num_conv_coeffs(k, :) = [1, 0];
            num_conv_uncerts(k, :) = [0, 0];
            rsq(3, k) = 1;
            msds(3, k) = 0;
        else
            ft = fittype(@(a, sigma2, x) a * profilefit_1d(x, init_prof, rawyData, sigma2)/max(init_prof), 'independent', 'x', 'dependent', 'y');
            lower_var_bound = -200^2;          % allows slight negative (sharpening) variance
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'off';
            opts.Lower = [0 lower_var_bound];
            opts.StartPoint = [num_conv_coeffs(k-1, 1),  baseblur^2 + max([(xData(2)-xData(1))^2, num_conv_coeffs(k-1, 2)])];
            opts.Upper = [Inf 40e6];
            [fitresult, gof] = fit( xData, gaussblur(xData, rawyData, baseblur^2)/max(rawyData), ft, opts);
            num_conv_coeffs(k,:) = coeffvalues(fitresult);
            confs = confint(fitresult);
            num_conv_uncerts(k,:) = reshape(confs(2,:), [1 2]) - num_conv_coeffs(k,:);
            num_conv_coeffs(k, 2) = num_conv_coeffs(k, 2) - baseblur^2; % remove baseline blur
            rsquared = struct2cell(gof);
            rsq(3, k) = cell2mat(rsquared(2));
            msds(3, k) = num_conv_coeffs(k, 2);
        end
        times(3) = times(3) + toc(proftime);

        % ----- Methods 4 & 5: Numerical integration (moments) -----
        % Model-free MSD from the data moments directly.
        msdtime = tic;
        mean_x_1d = sum(xData .* (rawyData))/sum(rawyData);                        % centroid
        num_int_coeffs(1, k) = (sum((xData-mean_x_1d).^2 .* rawyData))/(sum(rawyData)); % 1D 2nd central moment
        % Radial (cylindrical) estimate of the 2D profile variance <r^2>.
        % Divided by 2 to express as a per-dimension (1D) MSD
        num_int_coeffs(2, k) = (sum(rawyData .*abs(xData.^3)))/(2 * sum(rawyData .*abs(xData)));

        msds(4, k) = num_int_coeffs(1, k);
        msds(5, k) = num_int_coeffs(2, k);
        times(4) = times(4) + toc(msdtime);

        % ----- Method 6: Fourier fit -----
        % In k-space the profile is the initial spectrum times a Gaussian
        % falloff exp(-k^2*sigma2/2). Fit sigma2 (= MSD) over the k-range where
        % the signal is above a noise threshold.
        fourier_noise_threshold = 3;
        fourierfittime = tic;
        lower_var_bound = -200^2;
        if k == 1
            fft_init = fft_all(:, 1);
            % Estimate noise from the mid-band of the spectrum.
            fouriernoise_init = std(fft_init(floor(length(fft_init)/3):ceil(2*length(fft_init)/3)));
            cutoffind_init = 1;
            while abs(fft_init(cutoffind_init)) > abs(fouriernoise_init*fourier_noise_threshold)
                cutoffind_init = cutoffind_init + 1;
            end
            rsq(6, k) = 1;
            msds(6, k) = 0;
        else
            fft_data = fft_all(:, k);
            fouriernoise = std(fft_data(floor(length(fft_data)/3):ceil(2*length(fft_data)/3)));
            cutoffind_fourier = 1;
            while abs(fft_data(cutoffind_fourier)) > abs(fouriernoise*fourier_noise_threshold)
                cutoffind_fourier = cutoffind_fourier + 1;
            end
            cutoffind_fourier = min(cutoffind_fourier, cutoffind_init);

            ft = fittype(@(a, sigma2, k) a .* trimdata(fft_init, numel(k)) .*  exp(-k.^2 .* sigma2/2), 'independent', 'k', 'dependent', 'y');
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'off';
            opts.Lower = [1 lower_var_bound];
            opts.StartPoint = [1, fourier_fit_coeffs(k-1, 2)];
            opts.Upper = [1 40e6];          % amplitude fixed to 1 (bounds equal)

            [fitresult, gof] = fit(k_fourier(1:cutoffind_fourier), fft_data(1:cutoffind_fourier) .* max(fft_init)/max(fft_data), ft, opts);

            fourier_fit_coeffs(k,:) = coeffvalues(fitresult);
            confs = confint(fitresult);
            fourier_fit_uncerts(k,:) = reshape(confs(2,:), [1 2])-fourier_fit_coeffs(k,:);
            fourier_fit_coeffs(k, 2) = fourier_fit_coeffs(k, 2); % (no-op; kept for parallel structure)
            rsquared = struct2cell(gof);
            rsq(6, k) = cell2mat(rsquared(2));
            msds(6, k) = fourier_fit_coeffs(k, 2);
        end
        times(6) = times(6) + toc(fourierfittime);

        % ----- Method 7: Hankel fit -----
        % Cylindrical analogue of the Fourier fit. The (radial) Hankel transform
        % of the profile is fit to the initial transform times a Gaussian
        % falloff exp(-k^2*sigma2/2). Extra bookkeeping builds quadrature weights
        % to normalize the transforms and finds a per-frame noise cutoff.
        hankel_noise_threshold_factor = 7;
        hankel_k_start_index = 1;   % first k index used in the Hankel fit
        hankelfittime = tic;
        lower_var_bound = -200^2;
        if k == 1
            init_prof_hankel = rawyData;
            hankel_init = hankel_all(:, 1);

            cut_length = floor(length(k_hankel)/2);
            hankel_init = hankel_init(1:cut_length);
            k_hankel_abridged = k_hankel(1:cut_length);

            % Noise cutoff: keep k until the transform drops into the noise floor.
            hankel_noise_level = mean(abs(hankel_init(floor(cut_length/3):ceil(2*cut_length/3))), 1);
            hankel_cutoff_ind_init = 1;
            while abs(hankel_init(hankel_cutoff_ind_init + 1)) > abs(hankel_noise_level*hankel_noise_threshold_factor)
                hankel_cutoff_ind_init = hankel_cutoff_ind_init + 1;
            end

            k_hankel_cutoff = 0.03;   % hard k ceiling (1/nm) for the Hankel fit
            max_k_ind = sum(k_hankel_abridged <= k_hankel_cutoff);

            hankel_fit_coeffs(k, :) = 0;
            hankel_fit_uncerts(k, :) = 0;
            rsq(7, k) = 1;
            msds(7, k) = 0;
        else
            prof_hankel = hankel_all(1:cut_length, k);
            k_hankel_abridged = k_hankel(1:cut_length);

            hankel_noise_level = mean(abs(prof_hankel(floor(cut_length/3):ceil(2*cut_length/3))), 1);
            hankel_cutoff_ind = 1;
            while abs(prof_hankel(hankel_cutoff_ind + 1)) > abs(hankel_noise_level*hankel_noise_threshold_factor)
                hankel_cutoff_ind = hankel_cutoff_ind + 1;
            end

            % Fit at least 5 k-points, but never beyond the noise/ceiling limits.
            fit_length_hankel = min(max(5, hankel_cutoff_ind), max_k_ind);
            k_fit_hankel = k_hankel_abridged(1:fit_length_hankel);

            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'off';
            opts.Lower = [lower_var_bound 0];
            opts.StartPoint = [dx 1];
            opts.Upper = [4000000 Inf];
            ft = fittype(@(sigma2, a, kvec) hankel_init(1:length(kvec))/hankel_k0(1) .* exp(-kvec.^2 .* sigma2/2) * a, 'independent', 'kvec', 'dependent', 'y');
            [fitresult, gof] = fit( k_fit_hankel, prof_hankel(1:length(k_fit_hankel))/hankel_k0(k), ft, opts);

            hankel_fit_coeffs(k, :) = coeffvalues(fitresult);
            confs = confint(fitresult);
            hankel_fit_uncerts(k, :) = reshape(confs(2, :), [1 2])-hankel_fit_coeffs(k,:);
            rsquared = struct2cell(gof);
            rsq(7, k) = cell2mat(rsquared(2));
            msds(7, k) = hankel_fit_coeffs(k, 1);   % sigma2 = MSD
        end
        times(7) = times(7) + toc(hankelfittime);

        waitbar(k/end_ind_fit, wb);   % progress bar
    end
    close(wb)
    % Zero every MSD trace at t=0 so all methods start at 0 (removes the
    % initial-profile offset; the diffusive spread is what accumulates).
    msds = msds - msds(:, 1);
    keyValues.msds_allfits = msds;

    keyValues.gauss_coeffs = gauss_coeffs;
    keyValues.gauss_uncerts = gauss_uncerts;

    keyValues.voigt_coeffs = voigt_coeffs;
    keyValues.voigt_uncerts = voigt_uncerts;

    keyValues.prof_coeffs = num_conv_coeffs;
    keyValues.prof_uncerts = num_conv_uncerts;

    keyValues.num_int_coeffs = num_int_coeffs;

    keyValues.fourier_fit_coeffs = fourier_fit_coeffs;
    keyValues.fourier_fit_uncerts = fourier_fit_uncerts;

    keyValues.hankel_fit_coeffs = hankel_fit_coeffs;
    keyValues.hankel_fit_uncerts = hankel_fit_uncerts;

    keyValues.rsq_allfits = rsq;

    %% Plot R^2 vs time for the profile-shape methods -----------------------
    % (Skips the moment methods 4/5, which have no goodness-of-fit.)
    figure
    for i = [1 2 3 6 7]
        plot(tvec(1:size(rsq, 2)), rsq(i, :), 'LineWidth', 1)
        hold on
    end

    ax = gca;
    ax.LineWidth = 1;
    ax.Box = 'on';
    ylim([0.9 1]);
    set(gca,'FontSize',11)
    xlim([0, analysis_end_time])
    title("R^2", 'FontSize', 20)
    xlabel("Time (ns)", 'FontSize', 18);
    ylabel("R^2", 'FontSize', 18);
    legend(["Gauss fit", "Voigt fit", "Num. conv.", "Fourier Fit", "Hankel Fit"], 'Location', 'southwest', 'FontSize', 13, 'Box', 'off');
    print(gcf, fullfile(outfilepath , 'r2_fits_comparison.png'),'-dpng')

    %% Diffusion length ----------------------------------------------------
    % Evaluate sqrt(MSD) at the 1/e point of the TRPL decay -> a diffusion length.
    [~,tvec_0p37] = min(abs(TRPL - exp(-1))); % index where TRPL ~ 1/e
    if tvec_0p37 <= length(msds(1, :))
        DiffLength = msds(:, tvec_0p37).^0.5;
        fprintf("DiffLengths: ")
        for i = 1:7
            fprintf(DiffLength(i) + ",")
        end
        fprintf("\n");
        keyValues.difflength_nm_allfits = DiffLength(:)';
    end

    %% Plot MSD vs time for all seven methods ------------------------------
    figure
    for i = 1:7
        plot(tvec(1:length(msds(1, :))), msds(i, :)/1e6, 'LineWidth', 1) % /1e6: nm^2 -> um^2
        hold on
    end

    ax = gca;
    ax.LineWidth = 1;
    ax.Box = 'on';
    set(gca,'FontSize',11)
    xlim([0, analysis_end_time])
    title("MSD", 'FontSize', 15)
    xlabel("Time (ns)", 'FontSize', 13);
    ylabel("MSD (\mum^2)", 'FontSize', 13);
    legend(["Gauss fit", "Voigt fit", "Num. Conv.", "Num. Int. 1D", "Num. Int. 2D", "Fourier Fit", "Hankel Fit"], 'Location', 'northwest', 'FontSize', 13, 'Box', 'off');
    print(gcf, fullfile(outfilepath, 'msd_fits_comparison.png'),'-dpng')
    savefig(gcf, fullfile(outfilepath, 'figs\msd_fits_comparison.fig'))

    %% Diffusivities -------------------------------------------------------
    % Linear fit of MSD vs time; slope = 2*D (1D). D = slope/2, with 1e-5
    % converting nm^2/ns -> cm^2/s.
    Diffusivity_allFits = zeros(1, 7);
    fitnames = ["gauss", "voigt", "num conv", "num int 1d", "num int 2d", "fourier", "hankel"];
    rsq_msdfit = zeros(1, 7);
    for i = 1:7
        tstartind = 1;
        fit_tvec = tvec(tstartind:end) - tvec(tstartind);

        a = find(tvec < diff_fit_time, 1, "last");     % end index of the linear-fit window
        [xData, yData] = prepareCurveData(fit_tvec(1:a), msds(i, 1:a)');
        ft = fittype( 'a*x + b', 'independent', 'x', 'dependent', 'y');
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.StartPoint = [10000 0];
        [fitresult, gof] = fit( xData, yData, ft, opts );
        coeffs = coeffvalues(fitresult);
        confs = confint(fitresult);
        uncert = confs(2,:)-coeffs;
        rsquared = struct2cell(gof);
        rsq_msdfit(i) = cell2mat(rsquared(2));
        fitline = coeffs(1) * fit_tvec(1:min(a*3, length(fit_tvec))) + coeffs(2);

        Diffusivity_allFits(i) = coeffs(1) * 10^(-5) / 2;

        keyValues.msd_linear_tstart(i) = tvec(tstartind);
        keyValues.msd_linear_slope(i) = coeffs(1);
        keyValues.msd_linear_slope_uncert(i) = uncert(1);
        keyValues.msd_linear_yintercept(i) = coeffs(2);
        keyValues.msd_linear_yintercept_uncert(i) = uncert(2);
        keyValues.msd_linear_diff(i) = Diffusivity_allFits(i);
        keyValues.msd.linear_diff_uncert(i) = uncert(1)/coeffs(1)*Diffusivity_allFits(i);

        % Per-method MSD-vs-time plot with the linear fit overlaid.
        figure
        hold on
        plot(tvec(1:length(msds(i, :))), msds(i, :), 'ok')
        plot(fit_tvec(1:min(a*3, length(fit_tvec))),fitline, 'r', 'LineWidth', 2)
        ax = gca;
        ax.FontSize = 14;
        xlabel('time (ns)', 'FontSize', 15)
        xlim([0 analysis_end_time])
        ylabel('MSD (nm^2)', 'FontSize', 15)
        title(fitnames(i));
        set(gca, 'FontName', 'Arial', 'LineWidth', 2, 'Box', 'on')
        set(gcf, 'Color', 'w');
        box on
        hold off
        print(gcf, sprintf("%sMSDChangeLinearFit_%s.png", [outfilepath, fitnames(i)]),'-dpng')
    end
    fprintf('%s, ', fitnames)
    fprintf("\n");

    fprintf("Diffusivities: ")
    for i = 1:7
        fprintf(keyValues.msd_linear_diff(i) + ",")
    end
    fprintf("\n");
    fprintf("Uncertainties: ")
    for i = 1:7
        fprintf(keyValues.msd.linear_diff_uncert(i) + ",")
    end
    fprintf("\n");

    %% Kurtosis ------------------------------------------------------------
    % Inspired by Rodriguez et al., J. Phys. Chem. Lett. 2026, 17, 8, 2479–2484
    % Excess kurtosis of each spatial profile vs time. A pure Gaussian has 0;
    % deviations indicate non-Gaussian profile shapes.
    kurtosis_calc = zeros([1,end_ind_fit]);

    for k = 1:end_ind_fit
        spec = rawdata(:, k);
        spec = spec(:);
        [xData, rawyData] = prepareCurveData(pos_cor, spec);
        kurtosis_calc(1,k) = calculate_kurtosis_excess(xData,rawyData-back_prof);
    end

    keyValues.kurtosis = kurtosis_calc;
    figure('Name','Kurtosis','NumberTitle','off');
    plot(tvec(1:length(kurtosis_calc)),kurtosis_calc,'Linewidth',1.5);
    axis square;
    xlabel("Time (ns)");
    ylabel("Kurtosis");
    xlim([0, analysis_end_time]);
    ax = gca;
    ax.FontSize = 14;
    ax.LineWidth = 2;
    print(gcf, fullfile(outfilepath , 'kurtosis.png'),'-dpng');

    %% Save outputs --------------------------------------------------------
    diffs = Diffusivity_allFits(:);
    diff_uncerts = keyValues.msd.linear_diff_uncert(:);
    save(fullfile(outfilepath, "keyValues.mat"), "keyValues");
end