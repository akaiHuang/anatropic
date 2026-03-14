#!/usr/bin/env python3
"""
Quick verification of the 3D FFT Poisson gravity solver.

Test: sinusoidal density perturbation along x-axis.
  rho(x, y, z) = rho0 * (1 + A * sin(2*pi*x / L))

Analytical solution for the potential (periodic, Jeans swindle removes mean):
  Phi_analytical = -4*pi*G*rho0*A / k^2 * sin(k*x)
where k = 2*pi/L.

The discrete Laplacian gives a slightly different k_eff^2, so we compare
against the discrete-exact solution:
  k_eff^2 = (2/dx^2)(1 - cos(k*dx))
  Phi_discrete = 4*pi*G*rho0*A / k_eff^2 * sin(k*x)
  (positive sign because laplacian eigenvalue is negative)
"""

import sys
import os
import numpy as np

# Ensure anatropic is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from anatropic.gravity3d import solve_potential_3d, solve_gravity_3d


def test_sinusoidal_potential():
    """Test Phi against analytical solution for a sinusoidal density."""
    # Parameters
    Nx, Ny, Nz = 64, 32, 16
    Lx, Ly, Lz = 10.0, 5.0, 3.0
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    G = 1.0
    rho0 = 1.0
    A = 0.1

    # Wavenumber along x
    k = 2.0 * np.pi / Lx

    # 3D coordinate grid (density varies only along x)
    x = np.arange(Nx) * dx
    # rho3d has shape (Nz, Ny, Nx)
    X = x[np.newaxis, np.newaxis, :]  # (1, 1, Nx)

    # Density: varies only along x
    rho3d = rho0 * (1.0 + A * np.sin(k * X)) * np.ones((Nz, Ny, Nx))

    # Solve numerically
    Phi_num = solve_potential_3d(rho3d, dx, dy, dz, G=G)

    # Discrete-exact analytical solution
    # The discrete Laplacian eigenvalue for mode k along x is:
    #   lambda_x = -(2/dx^2)(1 - cos(k*dx))   [negative]
    # Since density is uniform in y, z, those modes are k_y=0, k_z=0
    # so lambda_y = lambda_z = 0.
    # Poisson: lambda * Phi_hat = 4*pi*G*rho_hat
    # => Phi_hat = 4*pi*G*rho_hat / lambda_x
    # Since lambda_x < 0, dividing by it flips the sign:
    #   Phi = -4*pi*G*rho0*A / k_eff^2 * sin(k*x)
    k_eff_sq = (2.0 / dx**2) * (1.0 - np.cos(k * dx))
    Phi_analytical = -(4.0 * np.pi * G * rho0 * A / k_eff_sq) * np.sin(k * X)
    Phi_analytical = Phi_analytical * np.ones((Nz, Ny, Nx))

    # Compare
    err = np.max(np.abs(Phi_num - Phi_analytical))
    rel_err = err / np.max(np.abs(Phi_analytical))

    print(f"  Potential test (sinusoidal along x):")
    print(f"    Grid: {Nx}x{Ny}x{Nz}, L = ({Lx}, {Ly}, {Lz})")
    print(f"    Max |Phi_num - Phi_exact|      = {err:.2e}")
    print(f"    Relative error                  = {rel_err:.2e}")

    return rel_err


def test_sinusoidal_acceleration():
    """Test gx against analytical derivative for a sinusoidal density."""
    # Parameters
    Nx, Ny, Nz = 64, 32, 16
    Lx, Ly, Lz = 10.0, 5.0, 3.0
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    G = 1.0
    rho0 = 1.0
    A = 0.1

    k = 2.0 * np.pi / Lx
    x = np.arange(Nx) * dx
    X = x[np.newaxis, np.newaxis, :]

    rho3d = rho0 * (1.0 + A * np.sin(k * X)) * np.ones((Nz, Ny, Nx))

    gx, gy, gz = solve_gravity_3d(rho3d, dx, dy, dz, G=G)

    # Analytical gx = -dPhi/dx
    # Phi = -C * sin(k*x)  where C = 4*pi*G*rho0*A / k_eff^2
    # dPhi/dx via central difference: [Phi(x+dx) - Phi(x-dx)] / (2*dx)
    # For Phi = -C*sin(k*x): dPhi/dx_cd = -C*[sin(k(x+dx)) - sin(k(x-dx))]/(2*dx)
    #                                    = -C*cos(k*x)*sin(k*dx)/dx
    # gx = -dPhi/dx = C*cos(k*x)*sin(k*dx)/dx
    k_eff_sq = (2.0 / dx**2) * (1.0 - np.cos(k * dx))
    C = 4.0 * np.pi * G * rho0 * A / k_eff_sq
    gx_analytical = C * np.cos(k * X) * np.sin(k * dx) / dx
    gx_analytical = gx_analytical * np.ones((Nz, Ny, Nx))

    err_gx = np.max(np.abs(gx - gx_analytical))
    rel_err_gx = err_gx / np.max(np.abs(gx_analytical))

    # gy and gz should be zero (density uniform in y, z)
    err_gy = np.max(np.abs(gy))
    err_gz = np.max(np.abs(gz))

    print(f"\n  Acceleration test (sinusoidal along x):")
    print(f"    Max |gx_num - gx_exact|         = {err_gx:.2e}")
    print(f"    Relative error in gx             = {rel_err_gx:.2e}")
    print(f"    Max |gy| (should be ~0)          = {err_gy:.2e}")
    print(f"    Max |gz| (should be ~0)          = {err_gz:.2e}")

    return rel_err_gx, err_gy, err_gz


def main():
    print("=" * 60)
    print("  3D Gravity Solver Verification")
    print("=" * 60)
    print()

    tol_potential = 1e-12
    tol_accel = 1e-12
    tol_transverse = 1e-12

    all_pass = True

    # Test 1: Potential
    rel_err_phi = test_sinusoidal_potential()
    if rel_err_phi < tol_potential:
        print(f"    => PASS (rel_err = {rel_err_phi:.2e} < {tol_potential:.0e})")
    else:
        print(f"    => FAIL (rel_err = {rel_err_phi:.2e} >= {tol_potential:.0e})")
        all_pass = False

    # Test 2: Acceleration
    rel_err_gx, err_gy, err_gz = test_sinusoidal_acceleration()
    if rel_err_gx < tol_accel:
        print(f"    => gx:  PASS (rel_err = {rel_err_gx:.2e} < {tol_accel:.0e})")
    else:
        print(f"    => gx:  FAIL (rel_err = {rel_err_gx:.2e} >= {tol_accel:.0e})")
        all_pass = False

    if err_gy < tol_transverse:
        print(f"    => gy:  PASS (|gy| = {err_gy:.2e} < {tol_transverse:.0e})")
    else:
        print(f"    => gy:  FAIL (|gy| = {err_gy:.2e} >= {tol_transverse:.0e})")
        all_pass = False

    if err_gz < tol_transverse:
        print(f"    => gz:  PASS (|gz| = {err_gz:.2e} < {tol_transverse:.0e})")
    else:
        print(f"    => gz:  FAIL (|gz| = {err_gz:.2e} >= {tol_transverse:.0e})")
        all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
