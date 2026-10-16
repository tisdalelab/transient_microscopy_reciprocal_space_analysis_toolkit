classdef hankel_transformer
% HANKEL_TRANSFORMER  Quasi-discrete Hankel transform (QDHT) helper for
% radially symmetric / cylindrical diffusion-imaging data.
%
% Provides the forward and inverse Hankel transforms used by analysis_loop
% when geometry == "cylindrical". A line profile through the center of a
% radially symmetric 2D distribution is transformed to k-space, where a
% diffusive Gaussian falloff can be fit exactly as in the Cartesian FFT case.
%
% The heavy matrices (which depend only on the sampling grid) are precomputed
% once and cached in `precalc`, so the same object can transform many time
% slices cheaply.
%
% USAGE
%   tx = hankel_transformer(x_centered);          % build (caches matrices)
%   [H, k] = hankel_transformer.ht(y, tx);        % forward transform
%   [y2, x] = hankel_transformer.iht(H, tx);      % inverse transform
%
%   Note: ht/iht are STATIC methods; pass the object (or a raw x vector) as
%   the second argument.
%
% ALGORITHM / CITATIONS (preserve when redistributing)
%   M. Guizar-Sicairos & J. C. Gutierrez-Vega, "Computation of quasi-discrete
%   Hankel transforms of integer order...", J. Opt. Soc. Am. A 21, 53-58 (2004),
%   and the MATLAB File Exchange implementation #6570 by Manuel Guizar.
%
% DEPENDENCY: a precomputed table of Bessel-function zeros, bessel0s.mat

    properties
        x         % centered position vector (0 = center of radial symmetry)
        precalc   % cached transform matrices and k/x grids (see get_precalc)
    end
    methods
        % x_centered must be shifted so that 0 corresponds to the center of
        % radial symmetry.
        function obj = hankel_transformer(x_centered)
            obj.x = x_centered(:);
            obj.precalc = hankel_transformer.get_precalc(x_centered);
        end
    end

    methods(Static)
        function [tx, k] = ht(y, x_or_hankel_transformer)
            % Forward QDHT of columns of y.
            % The second argument may be either a raw x vector (matrices are
            % recomputed) or a prebuilt hankel_transformer (matrices reused).
            %
            % A radial profile is split into a symmetric part (order-0 Hankel)
            % and an antisymmetric part (order-1 Hankel); the two transforms
            % are stacked into the returned column(s) tx, with matching k.
            if isnumeric(x_or_hankel_transformer)
                x = x_or_hankel_transformer(:);
                precalc = hankel_transformer.get_precalc(x);
            elseif isa(x_or_hankel_transformer, "hankel_transformer")
                x = x_or_hankel_transformer.x;
                precalc = x_or_hankel_transformer.precalc;
            else
                error("invalid second argument form, it should be a vector of x values or a hankel_transformer object");
            end

            zero_pad_length = round(length(x)/2);
            max_x = precalc.max_x;
            x_hankel_n0 = precalc.x_hankel_n0;   % order-0 sample radii
            x_hankel_n1 = precalc.x_hankel_n1;   % order-1 sample radii

            left_profs_n0 = zeros(zero_pad_length, size(y, 2));
            right_profs_n0 = zeros(zero_pad_length,  size(y, 2));
            left_profs_n1 = zeros(zero_pad_length,  size(y, 2));
            right_profs_n1 = zeros(zero_pad_length,  size(y, 2));

            % Resample the profile onto the Bessel-zero radial grids (both
            % sides of center) via spline interpolation.
            left_profs_n0(x_hankel_n0 <= max_x, :) = spline(x, y', -x_hankel_n0(x_hankel_n0 <= max_x))';
            right_profs_n0(x_hankel_n0 <= max_x, :) = spline(x, y', x_hankel_n0(x_hankel_n0 <= max_x))';

            left_profs_n1(x_hankel_n1 <= max_x, :) = spline(x, y', -x_hankel_n1(x_hankel_n1 <= max_x))';
            right_profs_n1(x_hankel_n1 <= max_x, :) = spline(x, y', x_hankel_n1(x_hankel_n1 <= max_x))';

            % Symmetric (even) and antisymmetric (odd) components.
            sym_profs = 1/2 * (left_profs_n0 + right_profs_n0);
            asym_profs = 1/4 * (right_profs_n1 - left_profs_n1);

            % Apply the cached forward transform matrices.
            h0 = precalc.transform_matrix_forward_n0 * sym_profs;
            h1 = precalc.transform_matrix_forward_n1 * asym_profs;

            k = [precalc.k_hankel_n0; precalc.k_hankel_n1];
            tx = [h0; h1]; % stack order-0 and order-1 transforms
        end

        function [itx, x] = iht(y, hankel_transformer, interp_method)
            % Inverse QDHT: reconstruct the real-space profile on the original
            % x grid. interp_method is "linear" (default) or "spline".
            %
            % NOTE: the "spline" branch references a variable `hankel_data`
            % that is never defined in this file — it appears to be dead/legacy
            % code. Use interp_method = "linear" unless you fix that branch.
            if nargin == 2
                interp_method = "linear";
            end
            x_hankel_n0 = hankel_transformer.precalc.x_hankel_n0(:);
            x_hankel_n1 = hankel_transformer.precalc.x_hankel_n1(:);

            k_hankel_n0 = hankel_transformer.precalc.k_hankel_n0;
            k_hankel_n1 = hankel_transformer.precalc.k_hankel_n1;

            % Split the stacked transform back into symmetric / antisymmetric parts.
            tx_sym = y(1:length(k_hankel_n0), :);
            tx_asym = y(length(k_hankel_n1)+1:end, :);

            % Reconstruct the even part and mirror it about the center.
            real_sym = zeros(length(x_hankel_n0) * 2, size(y, 2));
            real_sym(length(x_hankel_n0)+1:end, :) = hankel_transformer.precalc.transform_matrix_reverse_n0 * tx_sym;
            real_sym(1:length(x_hankel_n0), :) = flip(real_sym(length(x_hankel_n0)+1:end, :), 1);

            % Reconstruct the odd part and anti-mirror it about the center.
            real_asym = zeros(size(real_sym, 1) + 1, size(real_sym, 2));
            real_asym(length(x_hankel_n0)+2:end, :) = hankel_transformer.precalc.transform_matrix_reverse_n1 * tx_asym;
            real_asym(1:length(x_hankel_n0), :) = -flip(real_asym(length(x_hankel_n0)+2:end, :), 1);

            x = hankel_transformer.x;

            if interp_method == "spline"
                if isvector(hankel_data)
                    itx = spline([-flip(x_hankel_n0); x_hankel_n0], real_sym', abs(x));
                else
                    itx = spline([-flip(x_hankel_n0); x_hankel_n0], real_sym', abs(x))';
                end

                if isvector(hankel_data)
                    itx(x >= 0, :) = itx(x >= 0, :) + 2 * spline([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym', abs(x(x >= 0)));
                    itx(x < 0, :) = itx(x < 0, :) - 2 * spline([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym', abs(x(x < 0)));
                else
                    itx(x >= 0, :) = itx(x >= 0, :) + 2 * spline([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym', abs(x(x >= 0)))';
                    itx(x < 0, :) = itx(x < 0, :) - 2 * spline([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym', abs(x(x < 0)))';
                end
            elseif interp_method == "linear"
                % Combine even + odd radial parts, mapping |x| back to signed x.
                itx = interp1([-flip(x_hankel_n0); x_hankel_n0], real_sym, abs(x));
                itx(x >= 0, :) = itx(x >= 0, :) + 2 * interp1([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym, abs(x(x >= 0)));
                itx(x < 0, :) = itx(x < 0, :) - 2 * interp1([-flip(x_hankel_n1); 0; x_hankel_n1], real_asym, abs(x(x < 0)));
            end
        end
    end

    methods(Static, Access = private)
        function precalc = get_precalc(x)
            % Precompute the QDHT sampling grids (x_hankel, k_hankel) and the
            % forward/reverse transform matrices for orders 0 and 1.
            %
            % Loads precomputed Bessel-function zeros from bessel0s.mat.
            bessel_0s = load("bessel0s.mat");
            bessel_0s = bessel_0s.c;

            zero_pad_length = ceil(length(x)/2);

            % Split about center; the smaller half sets the usable radius.
            x = x(:);
            left_xData = x(x<=0);
            right_xData = x(x>=0);
            left_length = length(left_xData);
            right_length = length(right_xData);

            if left_length < right_length
                max_x_padded = -min(left_xData) * zero_pad_length/length(left_xData);
                max_x = -min(left_xData);
            else
                max_x_padded = max(right_xData) * zero_pad_length/length(right_xData);
                max_x = max(right_xData);
            end

            % Radial sample points = Bessel zeros scaled to the padded radius.
            x_hankel_n0 = bessel_0s(1, 1:zero_pad_length) * max_x_padded/bessel_0s(1, zero_pad_length);
            x_hankel_n1 = bessel_0s(2, 1:zero_pad_length) * max_x_padded/bessel_0s(2, zero_pad_length);

            %% --------------- build QDHT transform matrices --------------- %
            max_k_hankel_n0 = bessel_0s(1, zero_pad_length)/max_x_padded;
            max_k_hankel_n1 = bessel_0s(2, zero_pad_length)/max_x_padded;

            k_hankel_n0 = bessel_0s(1, 1:zero_pad_length)'/max_x_padded;
            k_hankel_n1 = bessel_0s(2, 1:zero_pad_length)'/max_x_padded;

            % Symmetric transform kernels (Guizar-Sicairos matrix C_mn).
            [Jn0, Jm0] = meshgrid(bessel_0s(1, 1:zero_pad_length), bessel_0s(1, 1:zero_pad_length));
            [Jn1, Jm1] = meshgrid(bessel_0s(2, 1:zero_pad_length), bessel_0s(2, 1:zero_pad_length));
            transform_matrix_n0 = (2/bessel_0s(1, zero_pad_length))*besselj(0,Jn0.*Jm0/bessel_0s(1,zero_pad_length))./(abs(besselj(1,Jn0)).*abs(besselj(1,Jm0)));
            transform_matrix_n1 = (2/bessel_0s(2, zero_pad_length))*besselj(1,Jn1.*Jm1/bessel_0s(2,zero_pad_length))./(abs(besselj(2,Jn1)).*abs(besselj(2,Jm1)));
            % Scaling vectors that convert the symmetric kernel into properly
            % normalized forward/reverse transforms in physical units.
            m1_n0 = (abs(besselj(1,bessel_0s(1, 1:zero_pad_length)))/max_x_padded)';
            m2_n0 = (abs(besselj(1,bessel_0s(1, 1:zero_pad_length)))* 2 * pi/max_k_hankel_n0)';
            m1_n1 = (abs(besselj(2,bessel_0s(2, 1:zero_pad_length)))/max_x_padded)';
            m2_n1 = (abs(besselj(2,bessel_0s(2, 1:zero_pad_length)))* 2 * pi/max_k_hankel_n1)';
            transform_matrix_forward_n0 = diag(m2_n0) * transform_matrix_n0 * diag(1./m1_n0);
            transform_matrix_reverse_n0 = diag(m1_n0) * transform_matrix_n0 * diag(1./m2_n0);
            transform_matrix_forward_n1 = diag(m2_n1) * transform_matrix_n1 * diag(1./m1_n1);
            transform_matrix_reverse_n1 = diag(m1_n1) * transform_matrix_n1 * diag(1./m2_n1);

            % Cache everything.
            precalc = struct;
            precalc.max_x = max_x;
            precalc.x_hankel_n0 = x_hankel_n0;
            precalc.x_hankel_n1 = x_hankel_n1;
            precalc.k_hankel_n0 = k_hankel_n0;
            precalc.k_hankel_n1 = k_hankel_n1;
            precalc.transform_matrix_forward_n0 = transform_matrix_forward_n0;
            precalc.transform_matrix_reverse_n0 = transform_matrix_reverse_n0;
            precalc.transform_matrix_forward_n1 = transform_matrix_forward_n1;
            precalc.transform_matrix_reverse_n1 = transform_matrix_reverse_n1;
        end
    end
end