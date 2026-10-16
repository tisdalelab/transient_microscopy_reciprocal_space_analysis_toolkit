function voigt = genvoigt(gamma, sigma, center, xvec)
% GENVOIGT  Generate a (peak-normalized) Voigt profile.
%
% A Voigt profile is the convolution of a Gaussian (std-dev sigma) and a
% Lorentzian (half-width gamma). Computed via the Faddeeva/complex error
% function (see fadf.m).
%
% INPUTS
%   gamma   Lorentzian half-width.
%   sigma   Gaussian standard deviation.
%   center  peak center position.
%   xvec    positions at which to evaluate the profile.
% OUTPUT
%   voigt   profile values, normalized so the peak = 1.
%
% NOTE: uses `i` as the imaginary unit; ensure `i` is not shadowed by a loop
% variable in the calling scope.

    zvector = (xvec - center + i * gamma)/(sigma * sqrt(2));
    voigt = real(fadf(zvector))/(sigma * sqrt(2 * pi));
    voigt = voigt/max(voigt);   % peak-normalize
end