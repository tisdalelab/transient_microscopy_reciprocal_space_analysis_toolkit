function k = calculate_kurtosis_excess(x,y)
% CALCULATE_KURTOSIS_EXCESS  Excess kurtosis of a distribution defined by a
% weighted 1D profile y(x).
%
% Treats y(x) as an (unnormalized) probability density over positions x and
% returns the excess kurtosis (kurtosis - 3). A Gaussian gives 0; positive
% values indicate heavier tails / sharper peak, negative values a flatter top.
% 
% Thanks to Julia Im for writing this method!
%
% INPUTS
%   x  position vector.
%   y  intensity/weight at each position (need not be normalized).
% OUTPUT
%   k  excess kurtosis (dimensionless).

    total_y = trapz(x,y);        % integral of the profile (normalization constant)
    y_norm = y/total_y;          % normalized density

    mu = trapz(x,x.* y_norm);    % mean position (1st moment)

    m2 = trapz(x,(((x-mu).^2).*y_norm));  % 2nd central moment (variance)

    m4 = trapz(x,(((x-mu).^4 .* y_norm))); % 4th central moment

    k = (m4/(m2^2))-3;           % excess kurtosis
end