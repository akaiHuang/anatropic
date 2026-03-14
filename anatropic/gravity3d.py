"""
3D Poisson solver for self-gravity using FFT (periodic boundary conditions).

Solves:  nabla^2 Phi = 4 pi G rho

and returns the gravitational acceleration (gx, gy, gz) = -grad(Phi).

For periodic domains, the mean density produces a uniform background that
does not source any gravitational force (the k=0 mode is set to zero).
This is the standard Jeans swindle, appropriate for studying gravitational
instability in a periodic box.
"""

import numpy as np


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

    Nz, Ny, Nx = rho3d.shape

    # Gravitational acceleration via central differences (periodic)
    gx = np.zeros_like(Phi)
    gy = np.zeros_like(Phi)
    gz = np.zeros_like(Phi)

    # g_x = -dPhi/dx  (axis 2)
    gx[:, :, 1:-1] = -(Phi[:, :, 2:] - Phi[:, :, :-2]) / (2.0 * dx)
    gx[:, :, 0]    = -(Phi[:, :, 1]  - Phi[:, :, -1])   / (2.0 * dx)
    gx[:, :, -1]   = -(Phi[:, :, 0]  - Phi[:, :, -2])   / (2.0 * dx)

    # g_y = -dPhi/dy  (axis 1)
    gy[:, 1:-1, :] = -(Phi[:, 2:, :] - Phi[:, :-2, :]) / (2.0 * dy)
    gy[:, 0, :]    = -(Phi[:, 1, :]  - Phi[:, -1, :])   / (2.0 * dy)
    gy[:, -1, :]   = -(Phi[:, 0, :]  - Phi[:, -2, :])   / (2.0 * dy)

    # g_z = -dPhi/dz  (axis 0)
    gz[1:-1, :, :] = -(Phi[2:, :, :] - Phi[:-2, :, :]) / (2.0 * dz)
    gz[0, :, :]    = -(Phi[1, :, :]  - Phi[-1, :, :])   / (2.0 * dz)
    gz[-1, :, :]   = -(Phi[0, :, :]  - Phi[-2, :, :])   / (2.0 * dz)

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
    rho_hat = np.fft.fftn(rho3d)

    # Wavenumber arrays
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Reorder to match (Nz, Ny, Nx) array layout:
    # meshgrid with indexing='ij' gives (Nx, Ny, Nz), so transpose
    KX = KX.transpose(2, 1, 0)  # (Nz, Ny, Nx)
    KY = KY.transpose(2, 1, 0)
    KZ = KZ.transpose(2, 1, 0)

    # Discrete Laplacian eigenvalues:
    #   -(2/dx^2)(1-cos(kx*dx)) - (2/dy^2)(1-cos(ky*dy)) - (2/dz^2)(1-cos(kz*dz))
    laplacian = -(2.0 / dx**2) * (1.0 - np.cos(KX * dx)) \
                -(2.0 / dy**2) * (1.0 - np.cos(KY * dy)) \
                -(2.0 / dz**2) * (1.0 - np.cos(KZ * dz))

    # Solve Phi_hat = 4 pi G rho_hat / laplacian (skip k=0: Jeans swindle)
    Phi_hat = np.zeros_like(rho_hat)
    nonzero = np.abs(laplacian) > 1e-30
    Phi_hat[nonzero] = 4.0 * np.pi * G * rho_hat[nonzero] / laplacian[nonzero]

    # Recover potential in real space
    Phi = np.real(np.fft.ifftn(Phi_hat))

    return Phi
