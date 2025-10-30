import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.optimize import newton
from scipy.interpolate import RegularGridInterpolator
import math

def legendre_gauss_lobatto(N):
    x = np.zeros(N)
    w = np.zeros(N)

    x[0] = -1.0
    x[-1] = 1.0

    # Legendre polynomial of degree N-1
    Pn = Legendre.basis(N - 1)
    dPn = Pn.deriv()

    # Chebyshev–Lobatto
    k = np.arange(1, N - 1)
    x_init = -np.cos(np.pi * k / (N - 1))

    # Solve P_{N-1}' = 0
    x[1:-1] = [newton(dPn, xi) for xi in x_init]

    # Compute weights
    for i in range(1, N - 1):
        xi = x[i]
        Pval = Pn(xi)
        w[i] = 2 / (N * (N - 1) * (Pval ** 2))

    # Boundary weights
    w[0] = w[-1] = 2 / (N * (N - 1))

    return x, w


def interpolate_to_uniform(U, N):

    x = np.linspace(-1, 1, N)
    [X, Y] = np.meshgrid(x, x)
    x_lgl, _ = legendre_gauss_lobatto(N)

    interp_u = RegularGridInterpolator((x_lgl, x_lgl), U)
    query_points = np.column_stack([Y.ravel(), X.ravel()])
    U_interp = interp_u(query_points).reshape(X.shape)
    return U_interp


def control_ticks(U, num_digits=1, num_ticks=3):

    def floor_to_decimal(x, decimals):
        factor = 10 ** decimals
        return math.floor(x * factor) / factor
    
    decimals = -int(np.floor(np.log10(abs(U.max()))))
    floor_max = floor_to_decimal(U.max(), decimals + (num_digits-1))
    floor_min = floor_to_decimal(U.min(), decimals + (num_digits-1))
    ticks = np.linspace(floor_min, floor_max, num_ticks)
    return ticks