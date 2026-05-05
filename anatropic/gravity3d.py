"""
3D Poisson solver for self-gravity using FFT (periodic boundary conditions).

Solves:  nabla^2 Phi = 4 pi G rho

and returns the gravitational acceleration (gx, gy, gz) = -grad(Phi).

For periodic domains, the mean density produces a uniform background that
does not source any gravitational force (the k=0 mode is set to zero).
This is the standard Jeans swindle, appropriate for studying gravitational
instability in a periodic box.

The numerical kernel is written against the backend abstraction in
``anatropic._backend``: it runs on NumPy (float64, CPU) by default and on
MLX (float32, Apple Silicon GPU) when ``ANATROPIC_BACKEND=mlx``.
"""

import numpy as np

from . import _backend as _be
from ._backend import xp, xpfft


def solve_gravity_3d(rho3d, dx, dy, dz, G=1.0):
    """
    Solve the 3D Poisson equation for self-gravity with periodic boundaries.

    Uses FFT to solve nabla^2 Phi = 4*pi*G*rho in Fourier space, then
    computes (gx, gy, gz) = -grad(Phi) via central finite differences.

    The k=0 (mean density) mode is set to zero, implementing the Jeans swindle
    so that the uniform background does not produce a net force.

    Parameters
    ----------
    rho3d : ndarray, shape (Nz, Ny, Nx)
        Mass density on a uniform 3D grid.
    dx : float
        Cell width in the x-direction (axis 2).
    dy : float
        Cell width in the y-direction (axis 1).
    dz : float
        Cell width in the z-direction (axis 0).
    G : float, optional
        Gravitational constant. Default is 1.0 (code units).

    Returns
    -------
    gx, gy, gz : ndarray, shape (Nz, Ny, Nx)
        Gravitational acceleration components at cell centres.
        gx = -dPhi/dx, gy = -dPhi/dy, gz = -dPhi/dz.
    """
    Phi = solve_potential_3d(rho3d, dx, dy, dz, G=G)

    # Build gradient of Phi via periodic finite differences using xp.roll.
    # This avoids per-axis slice writes (which are awkward to express
    # uniformly across NumPy and MLX).
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_2dz = 1.0 / (2.0 * dz)

    # axis ordering: rho3d is (Nz, Ny, Nx)  -> z=0, y=1, x=2
    Phi_xp = xp.roll(Phi, -1, axis=2)
    Phi_xm = xp.roll(Phi,  1, axis=2)
    Phi_yp = xp.roll(Phi, -1, axis=1)
    Phi_ym = xp.roll(Phi,  1, axis=1)
    Phi_zp = xp.roll(Phi, -1, axis=0)
    Phi_zm = xp.roll(Phi,  1, axis=0)

    gx = -(Phi_xp - Phi_xm) * inv_2dx
    gy = -(Phi_yp - Phi_ym) * inv_2dy
    gz = -(Phi_zp - Phi_zm) * inv_2dz

    return gx, gy, gz


def solve_potential_3d(rho3d, dx, dy, dz, G=1.0):
    """
    Solve the 3D Poisson equation and return the gravitational potential.

    Parameters
    ----------
    rho3d : ndarray, shape (Nz, Ny, Nx)
        Mass density on a uniform 3D grid.
    dx : float
        Cell width in x (axis 2).
    dy : float
        Cell width in y (axis 1).
    dz : float
        Cell width in z (axis 0).
    G : float, optional
        Gravitational constant. Default is 1.0.

    Returns
    -------
    Phi : ndarray, shape (Nz, Ny, Nx)
        Gravitational potential (k=0 mode set to zero).
    """
    Nz, Ny, Nx = rho3d.shape

    # 3D FFT of density
    rho_hat = xpfft.fftn(rho3d)

    # Wavenumber arrays (backend-agnostic fftfreq)
    kx = 2.0 * np.pi * _be.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * _be.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * _be.fftfreq(Nz, d=dz)

    # Build (Nz, Ny, Nx) wavenumber arrays via broadcasting.
    # Reshape each 1D fftfreq vector to broadcast along its own axis only.
    KZ = kz.reshape(Nz, 1, 1)
    KY = ky.reshape(1, Ny, 1)
    KX = kx.reshape(1, 1, Nx)

    # Discrete Laplacian eigenvalues:
    #   -(2/dx^2)(1-cos(kx*dx)) - (2/dy^2)(1-cos(ky*dy)) - (2/dz^2)(1-cos(kz*dz))
    laplacian = (
        -(2.0 / dx**2) * (1.0 - xp.cos(KX * dx))
        - (2.0 / dy**2) * (1.0 - xp.cos(KY * dy))
        - (2.0 / dz**2) * (1.0 - xp.cos(KZ * dz))
    )

    # Solve Phi_hat = 4 pi G rho_hat / laplacian (skip k=0: Jeans swindle).
    # We use xp.where to handle the k=0 mode (laplacian == 0) without boolean
    # indexing, which MLX does not yet support. The "safe" denominator is set
    # to 1.0 wherever laplacian is too small, then the result is masked back
    # to zero.
    safe_lap = xp.where(xp.abs(laplacian) > 1e-30, laplacian, 1.0)
    Phi_hat_unmasked = (4.0 * np.pi * G) * rho_hat / safe_lap
    zero_complex = _be.zeros(rho_hat.shape, dtype=rho_hat.dtype)
    Phi_hat = xp.where(xp.abs(laplacian) > 1e-30, Phi_hat_unmasked, zero_complex)

    # Recover potential in real space
    Phi = _be.to_real(xpfft.ifftn(Phi_hat))

    return Phi
