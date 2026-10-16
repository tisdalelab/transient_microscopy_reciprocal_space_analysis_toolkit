function blurred = gaussblur(x, profile, var, conv_method, trim)
% GAUSSBLUR  Convolve a 1D profile with a Gaussian kernel of variance `var`.
%
% Used to model diffusive spreading (and instrument blur) of a spatial profile.
%
% INPUTS
%   x            position vector (uniform spacing assumed).
%   profile      profile values sampled at x.
%   var          Gaussian variance (nm^2). var == 0 returns the profile unblurred.
%   conv_method  boundary handling (default "zero_pad"):
%                  "zero_pad"   - conv(...,'same'); implicit zero padding.
%                  "valid_only" - only fully-overlapping region, then re-padded.
%                  "same_pad"   - replicate-pad edges before convolving (may reduce
%                                 edge artifacts).
%   trim         number of edge samples to trim then zero-pad back (default 0);
%                pass 'adaptive' to size it from the kernel width.
% OUTPUT
%   blurred      blurred profile, resized to length(x).

if nargin == 3
    conv_method = "zero_pad";
    trim = 0;
elseif nargin == 4
    trim = 0;
end

% Zero-variance shortcut: no blurring, just apply trim/repad bookkeeping.
if var == 0
    blurred = profile;
    blurred = blurred(1+trim:end-trim);
    blurred = padarray(blurred, trim);
    blurred = blurred(1:length(x));
    return
end

dx = x(2) - x(1);

if ischar(trim) && string(trim) == 'adaptive'
    trim = round(5*sqrt(var)/dx);
end
% Build the kernel spanning +/- 5 sigma, on the same grid spacing as x.
gauss_start = dx * round(5*sqrt(var)/dx);
gauss_xvec = (-gauss_start:dx:gauss_start)';

% continuous Gaussian kernel, area-normalized to conserve total signal:
gauss_profile = exp(-(gauss_xvec).^2./(2 * var));
gauss_profile = gauss_profile/sum(gauss_profile);
gauss_len = length(gauss_profile);

if conv_method == "zero_pad"
    blurred = conv(profile, gauss_profile, 'same');
elseif conv_method == "valid_only"
    if length(gauss_xvec) > 0.2 * length(profile)
        disp("Warning: cutting off " + 2 * length(gauss_xvec)/length(profile) + " of profile")
    end
    blurred = conv(profile, gauss_profile, 'valid');
    blurred = padarray(blurred, 0.5*(length(profile) - length(blurred)));
elseif conv_method == "same_pad"
    % Replicate edges so the convolution doesn't pull the profile toward zero.
    blurred_expanded = conv(padarray(profile, gauss_len, "replicate"), gauss_profile, 'same');
    blurred = blurred_expanded(1 + gauss_len : end - gauss_len);
else
    disp("Invalid method, defaulting to same_pad");
    blurred_expanded = conv(padarray(profile, gauss_len, "replicate"), gauss_profile, 'same');
    blurred = blurred_expanded(1 + gauss_len : end - gauss_len);
end
% Optional edge trim + re-pad, then force output length to match x.
blurred = blurred(1+trim:end-trim);
blurred = padarray(blurred, trim);
blurred = blurred(1:length(x));
end