"""hankel_transformer.py -- Quasi-discrete Hankel transform (QDHT) helper for
radially symmetric / cylindrical diffusion-imaging data.

Python port of the MATLAB `hankel_transformer` class. Bessel-function zeros
are computed on the fly with scipy.special.jn_zeros (so the precomputed
my_bessel0s.mat / bessel0s.mat table is no longer required).

ALGORITHM / CITATIONS (preserve when redistributing)
  M. Guizar-Sicairos & J. C. Gutierrez-Vega, "Computation of quasi-discrete
  Hankel transforms of integer order...", J. Opt. Soc. Am. A 21, 53-58 (2004),
  and MATLAB File Exchange implementation #6570 by Manuel Guizar.
"""
import numpy as np
from scipy.special import jv, jn_zeros
from scipy.interpolate import CubicSpline


class HankelTransformer:
    def __init__(self, x_centered):
        # x_centered must be shifted so 0 = center of radial symmetry.
        self.x = np.asarray(x_centered, dtype=float).ravel()
        self.precalc = self._get_precalc(self.x)

    # ------------------------------------------------------------------ #
    @staticmethod
    def ht(y, x_or_transformer):
        """Forward QDHT of the columns of `y`.

        The second argument may be a raw x-vector (matrices recomputed) or a
        prebuilt HankelTransformer (matrices reused). Returns (tx, k).
        """
        if isinstance(x_or_transformer, HankelTransformer):
            x = x_or_transformer.x
            precalc = x_or_transformer.precalc
        else:
            x = np.asarray(x_or_transformer, dtype=float).ravel()
            precalc = HankelTransformer._get_precalc(x)

        y = np.asarray(y, dtype=float)
        squeeze = (y.ndim == 1)
        if squeeze:
            y = y[:, None]
        ncols = y.shape[1]

        zero_pad_length = int(np.ceil(len(x) / 2.0))
        max_x = precalc['max_x']
        x_hankel_n0 = precalc['x_hankel_n0']
        x_hankel_n1 = precalc['x_hankel_n1']

        left_profs_n0 = np.zeros((zero_pad_length, ncols))
        right_profs_n0 = np.zeros((zero_pad_length, ncols))
        left_profs_n1 = np.zeros((zero_pad_length, ncols))
        right_profs_n1 = np.zeros((zero_pad_length, ncols))

        # Resample the profile onto the Bessel-zero radial grids (both sides
        # of center) via spline interpolation.
        cs = CubicSpline(x, y, axis=0, extrapolate=True)
        m0 = x_hankel_n0 <= max_x
        m1 = x_hankel_n1 <= max_x
        left_profs_n0[m0, :] = cs(-x_hankel_n0[m0])
        right_profs_n0[m0, :] = cs(x_hankel_n0[m0])
        left_profs_n1[m1, :] = cs(-x_hankel_n1[m1])
        right_profs_n1[m1, :] = cs(x_hankel_n1[m1])

        # Symmetric (even) and antisymmetric (odd) components.
        sym_profs = 0.5 * (left_profs_n0 + right_profs_n0)
        asym_profs = 0.25 * (right_profs_n1 - left_profs_n1)

        # Apply the cached forward transform matrices.
        h0 = precalc['transform_matrix_forward_n0'] @ sym_profs
        h1 = precalc['transform_matrix_forward_n1'] @ asym_profs

        k = np.concatenate([precalc['k_hankel_n0'], precalc['k_hankel_n1']])
        tx = np.vstack([h0, h1])
        if squeeze:
            tx = tx[:, 0]
        return tx, k

    # ------------------------------------------------------------------ #
    @staticmethod
    def iht(y, transformer, interp_method="linear"):
        """Inverse QDHT: reconstruct the real-space profile on the original x
        grid. Only interp_method='linear' is supported (the MATLAB 'spline'
        branch referenced an undefined variable and was dead/legacy code).
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        p = transformer.precalc
        x_hankel_n0 = p['x_hankel_n0'].ravel()
        k_hankel_n0 = p['k_hankel_n0'].ravel()
        k_hankel_n1 = p['k_hankel_n1'].ravel()

        tx_sym = y[:len(k_hankel_n0), :]
        tx_asym = y[len(k_hankel_n1):, :]

        n0 = len(x_hankel_n0)
        real_sym = np.zeros((n0 * 2, y.shape[1]))
        real_sym[n0:, :] = p['transform_matrix_reverse_n0'] @ tx_sym
        real_sym[:n0, :] = np.flip(real_sym[n0:, :], axis=0)

        real_asym = np.zeros((real_sym.shape[0] + 1, real_sym.shape[1]))
        real_asym[n0 + 1:, :] = p['transform_matrix_reverse_n1'] @ tx_asym
        real_asym[:n0, :] = -np.flip(real_asym[n0 + 1:, :], axis=0)

        x = transformer.x
        if interp_method != "linear":
            raise NotImplementedError("Only interp_method='linear' is supported.")

        grid0 = np.concatenate([-np.flip(x_hankel_n0), x_hankel_n0])
        grid1 = np.concatenate([-np.flip(p['x_hankel_n1'].ravel()),
                                [0.0], p['x_hankel_n1'].ravel()])

        def interp_cols(grid, vals, xq):
            out = np.empty((len(xq), vals.shape[1]))
            for c in range(vals.shape[1]):
                out[:, c] = np.interp(xq, grid, vals[:, c])
            return out

        itx = interp_cols(grid0, real_sym, np.abs(x))
        pos = x >= 0
        neg = x < 0
        itx[pos, :] += 2 * interp_cols(grid1, real_asym, np.abs(x[pos]))
        itx[neg, :] -= 2 * interp_cols(grid1, real_asym, np.abs(x[neg]))
        return itx, x

    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_precalc(x):
        """Precompute QDHT sampling grids and forward/reverse transform
        matrices for orders 0 and 1."""
        x = np.asarray(x, dtype=float).ravel()
        zero_pad_length = int(np.ceil(len(x) / 2.0))
        N = zero_pad_length

        # Bessel-function zeros: row0 -> zeros of J0, row1 -> zeros of J1.
        j0 = jn_zeros(0, N)          # length N
        j1 = jn_zeros(1, N)

        # Split about center; the smaller half sets the usable radius.
        left_xData = x[x <= 0]
        right_xData = x[x >= 0]
        if len(left_xData) < len(right_xData):
            max_x_padded = -np.min(left_xData) * N / len(left_xData)
            max_x = -np.min(left_xData)
        else:
            max_x_padded = np.max(right_xData) * N / len(right_xData)
            max_x = np.max(right_xData)

        # Radial sample points = Bessel zeros scaled to the padded radius.
        x_hankel_n0 = j0 * max_x_padded / j0[-1]
        x_hankel_n1 = j1 * max_x_padded / j1[-1]

        max_k_hankel_n0 = j0[-1] / max_x_padded
        max_k_hankel_n1 = j1[-1] / max_x_padded
        k_hankel_n0 = (j0 / max_x_padded).reshape(-1, 1)
        k_hankel_n1 = (j1 / max_x_padded).reshape(-1, 1)

        # Symmetric transform kernels (Guizar-Sicairos matrix C_mn).
        Jn0, Jm0 = np.meshgrid(j0, j0)
        Jn1, Jm1 = np.meshgrid(j1, j1)
        tm_n0 = (2.0 / j0[-1]) * jv(0, Jn0 * Jm0 / j0[-1]) \
            / (np.abs(jv(1, Jn0)) * np.abs(jv(1, Jm0)))
        tm_n1 = (2.0 / j1[-1]) * jv(1, Jn1 * Jm1 / j1[-1]) \
            / (np.abs(jv(2, Jn1)) * np.abs(jv(2, Jm1)))

        # Scaling vectors converting the kernel into physical-unit transforms.
        m1_n0 = np.abs(jv(1, j0)) / max_x_padded
        m2_n0 = np.abs(jv(1, j0)) * 2 * np.pi / max_k_hankel_n0
        m1_n1 = np.abs(jv(2, j1)) / max_x_padded
        m2_n1 = np.abs(jv(2, j1)) * 2 * np.pi / max_k_hankel_n1

        fwd_n0 = np.diag(m2_n0) @ tm_n0 @ np.diag(1.0 / m1_n0)
        rev_n0 = np.diag(m1_n0) @ tm_n0 @ np.diag(1.0 / m2_n0)
        fwd_n1 = np.diag(m2_n1) @ tm_n1 @ np.diag(1.0 / m1_n1)
        rev_n1 = np.diag(m1_n1) @ tm_n1 @ np.diag(1.0 / m2_n1)

        return dict(max_x=max_x, x_hankel_n0=x_hankel_n0, x_hankel_n1=x_hankel_n1,
                    k_hankel_n0=k_hankel_n0, k_hankel_n1=k_hankel_n1,
                    transform_matrix_forward_n0=fwd_n0,
                    transform_matrix_reverse_n0=rev_n0,
                    transform_matrix_forward_n1=fwd_n1,
                    transform_matrix_reverse_n1=rev_n1)