function fit_profile = profilefit_1d(x, init_profile, later_profile, var, conv_method, trim)
% PROFILEFIT_1D  Model a later profile as the initial profile spread by
% Gaussian diffusion of variance `var`.
%
% This is the model function behind the "numerical convolution" MSD method in
% analysis_loop. It supports both spreading (var >= 0) and — via a first-order
% deconvolution trick — apparent sharpening (var < 0).
%
% INPUTS
%   x              position vector.
%   init_profile   reference (t=0) profile.
%   later_profile  measured profile at the later time (used only when var<0).
%   var            Gaussian variance to apply (nm^2); may be negative.
%   conv_method    passed through to gaussblur (default "zero_pad").
%   trim           passed through to gaussblur (default 0).
% OUTPUT
%   fit_profile    modeled profile at the later time.

    if nargin == 4
        conv_method = "zero_pad";
        trim = 0;
    elseif nargin == 5
        conv_method = "zero_pad";
        trim = 0;
    elseif nargin == 6
        trim = 0;
    end

    if var >= 0
        % Normal case: blur the initial profile by the diffusive variance.
        fit_profile = gaussblur(x, init_profile, var, conv_method, trim);
    else
        % Negative variance: approximate "un-blurring" by adding the difference
        % between the (area-matched) later profile and its re-blurred version.
        later_profile_rescaled = later_profile/sum(later_profile) * sum(init_profile);
        fit_profile = later_profile_rescaled + init_profile - gaussblur(x, later_profile_rescaled, -var, conv_method, trim);
    end
end