"""
1D Poisson solver for self-gravity using FFT (periodic boundary conditions).

Solves:  d^2 Phi / dx^2 = 4 pi G rho

and returns the gravitational acceleration g = -dPhi/dx.

For periodic domains, the mean density produces a uniform background that
does not source any gravitational force (the k=0 mode is set to zero).
This is the standard Jeans swindle, appropriate for studying gravitational
instability in a periodic box.
"""

import numpy as np


def solve_gravity(rho, dx, G=1.0):
    """
    Solve the 1D Poisson equation for self-gravity with periodic boundaries.

    Uses FFT to solve d^2 Phi/dx^2 = 4*pi*G*rho in Fourier space, then
    computes g = -dPhi/dx via spectral differentiation.

    The k=0 (mean density) mode is set to zero, implementing the Jeans swindle
    so that the uniform background does not produce a net force.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Mass density on N uniformly spaced cells.
    dx : float
        Cell width.
    G : float, optional
        Gravitational constant. Default is 1.0 (code units).

    Returns
    -------
    g : ndarray, shape (N,)
        Gravitational acceleration at cell centres, g = -dPhi/dx.
    """
    N = len(rho)
    L = N * dx

    # FFT of the density field
    rho_hat = np.fft.rfft(rho)

    # Wavenumbers for the real FFT
    # k_n = 2*pi*n / L  for n = 0, 1, ..., N//2
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

    # Solve Poisson equation in Fourier space:
    #   -k^2 * Phi_hat = 4*pi*G * rho_hat
    #   Phi_hat = -4*pi*G * rho_hat / k^2
    #
    # Gravitational acceleration in Fourier space:
    #   g_hat = -i*k * Phi_hat = i*k * 4*pi*G * rho_hat / k^2
    #         = 4*pi*G * i * rho_hat / k
    #
    # But we must be careful: use the discrete Laplacian for better accuracy.
    # The discrete second derivative in Fourier space is:
    #   -(2/dx^2) * (1 - cos(k*dx))
    # This avoids aliasing artifacts at high k.

    # Discrete Laplacian eigenvalues
    laplacian_k = -(2.0 / dx**2) * (1.0 - np.cos(k * dx))

    # Solve for Phi_hat: laplacian_k * Phi_hat = 4*pi*G * rho_hat
    # => Phi_hat = 4*pi*G * rho_hat / laplacian_k
    Phi_hat = np.zeros_like(rho_hat)
    # Skip k=0 mode (Jeans swindle: uniform background produces no force)
    nonzero = np.abs(laplacian_k) > 1e-30
    Phi_hat[nonzero] = 4.0 * np.pi * G * rho_hat[nonzero] / laplacian_k[nonzero]
    Phi_hat[~nonzero] = 0.0

    # Recover Phi in real space
    Phi = np.fft.irfft(Phi_hat, n=N)

    # Compute g = -dPhi/dx using central finite differences (periodic)
    g = np.zeros(N)
    g[1:-1] = -(Phi[2:] - Phi[:-2]) / (2.0 * dx)
    # Periodic boundaries for the gradient
    g[0] = -(Phi[1] - Phi[-1]) / (2.0 * dx)
    g[-1] = -(Phi[0] - Phi[-2]) / (2.0 * dx)

    return g


def solve_potential(rho, dx, G=1.0):
    """
    Solve the 1D Poisson equation and return the gravitational potential.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Mass density.
    dx : float
        Cell width.
    G : float, optional
        Gravitational constant. Default is 1.0.

    Returns
    -------
    Phi : ndarray, shape (N,)
        Gravitational potential (k=0 mode set to zero).
    """
    N = len(rho)
    rho_hat = np.fft.rfft(rho)
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

    laplacian_k = -(2.0 / dx**2) * (1.0 - np.cos(k * dx))

    Phi_hat = np.zeros_like(rho_hat)
    nonzero = np.abs(laplacian_k) > 1e-30
    Phi_hat[nonzero] = 4.0 * np.pi * G * rho_hat[nonzero] / laplacian_k[nonzero]

    return np.fft.irfft(Phi_hat, n=N)
