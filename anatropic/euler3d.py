"""
3D compressible Euler equations solver using dimensional splitting with batched HLLE.

Extends the 1D HLLE solver (euler.py) to three dimensions via Strang splitting.
All pencil-based sweeps are fully batched: sweep_x processes (Nz*Ny) pencils of
length Nx simultaneously, avoiding the O(Nz*Ny) Python loop that would make a
naive extension of the 2D code catastrophically slow.

The conservative variables per 1D pencil are U = [rho, rho*v_par, E] where
E = rho*e_int + 0.5*rho*(v_par^2 + v_perp1^2 + v_perp2^2).
Transverse velocities are passively advected with a donor-cell (upwind) scheme.

Author: Anatropic project
"""

import numpy as np

from .euler import DENSITY_FLOOR, PRESSURE_FLOOR


# ============================================================================
# State3D
# ============================================================================

class State3D:
    """
    Primitive-variable state on a 3D grid.

    Stores separate arrays (all shape (Nz, Ny, Nx)):
        rho  -- mass density
        vx   -- x-velocity
        vy   -- y-velocity
        vz   -- z-velocity
        eint -- specific internal energy

    Parameters
    ----------
    Nz, Ny, Nx : int
        Grid dimensions.
    rho0 : float
        Initial uniform density.
    eos : object
        Equation of state with pressure(), sound_speed(), internal_energy().
    """

    def __init__(self, Nz, Ny, Nx, rho0, eos):
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.eos = eos

        self.rho = np.full((Nz, Ny, Nx), rho0)
        self.vx = np.zeros((Nz, Ny, Nx))
        self.vy = np.zeros((Nz, Ny, Nx))
        self.vz = np.zeros((Nz, Ny, Nx))

        P0 = eos.pressure(self.rho, np.zeros_like(self.rho))
        self.eint = eos.internal_energy(self.rho, P0)


# ============================================================================
# Batched primitive extraction
# ============================================================================

def _primitive_from_conservative_batch(U, eos):
    """
    Extract primitive variables from batched conservative state.

    Parameters
    ----------
    U : ndarray, shape (3, batch, N)
        Conservative variables [rho, rho*v, E].
    eos : object
        Equation of state.

    Returns
    -------
    rho, v, P, cs, eint : ndarrays, each shape (batch, N)
    """
    rho = np.maximum(U[0], DENSITY_FLOOR)
    v = U[1] / rho
    eint = U[2] / rho - 0.5 * v ** 2
    eint = np.maximum(eint, 0.0)

    P = eos.pressure(rho, eint)
    P = np.maximum(P, PRESSURE_FLOOR)
    cs = eos.sound_speed(rho, eint)
    cs = np.maximum(cs, 0.0)

    return rho, v, P, cs, eint


# ============================================================================
# Batched flux and HLLE solver
# ============================================================================

def _flux_batch(rho, v, P, E):
    """
    Compute the Euler flux vector F(U) for batched inputs.

    F = [rho*v, rho*v^2 + P, (E + P)*v]

    Parameters
    ----------
    rho, v, P, E : ndarrays, shape (batch, N)

    Returns
    -------
    F : ndarray, shape (3, batch, N)
    """
    F = np.empty((3,) + rho.shape)
    F[0] = rho * v
    F[1] = rho * v ** 2 + P
    F[2] = (E + P) * v
    return F


def _hlle_flux_batch(U_L, U_R, eos):
    """
    Batched HLLE approximate Riemann solver.

    Parameters
    ----------
    U_L : ndarray, shape (3, batch, N_interfaces)
        Left states.
    U_R : ndarray, shape (3, batch, N_interfaces)
        Right states.
    eos : object
        Equation of state.

    Returns
    -------
    F_hlle : ndarray, shape (3, batch, N_interfaces)
        HLLE numerical flux at each interface.
    """
    rho_L, v_L, P_L, cs_L, _ = _primitive_from_conservative_batch(U_L, eos)
    rho_R, v_R, P_R, cs_R, _ = _primitive_from_conservative_batch(U_R, eos)

    E_L = U_L[2]
    E_R = U_R[2]

    # Wave speed estimates (Davis)
    S_L = np.minimum(v_L - cs_L, v_R - cs_R)
    S_R = np.maximum(v_L + cs_L, v_R + cs_R)

    F_L = _flux_batch(rho_L, v_L, P_L, E_L)
    F_R = _flux_batch(rho_R, v_R, P_R, E_R)

    # HLLE flux
    denom = S_R - S_L
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    F_hlle = np.empty_like(F_L)
    for i in range(3):
        F_mid = (S_R * F_L[i] - S_L * F_R[i]
                 + S_L * S_R * (U_R[i] - U_L[i])) / denom
        F_mid = np.where(S_L >= 0, F_L[i], F_mid)
        F_mid = np.where(S_R <= 0, F_R[i], F_mid)
        F_hlle[i] = F_mid

    return F_hlle


# ============================================================================
# Batched ghost-cell padding (periodic, along last axis)
# ============================================================================

def _add_ghost_cells_batch(U, nghosts=2):
    """
    Add ghost cells along the last axis for periodic BC (batched).

    Parameters
    ----------
    U : ndarray, shape (3, batch, N)
    nghosts : int

    Returns
    -------
    U_ext : ndarray, shape (3, batch, N + 2*nghosts)
    """
    # U[..., -nghosts:]  and U[..., :nghosts] give periodic wraps
    left = U[:, :, -nghosts:]
    right = U[:, :, :nghosts]
    return np.concatenate([left, U, right], axis=2)


def _add_ghost_cells_1d_batch(arr, nghosts=2):
    """
    Add ghost cells along the last axis for a (batch, N) array (periodic).

    Parameters
    ----------
    arr : ndarray, shape (batch, N)
    nghosts : int

    Returns
    -------
    arr_ext : ndarray, shape (batch, N + 2*nghosts)
    """
    left = arr[:, -nghosts:]
    right = arr[:, :nghosts]
    return np.concatenate([left, arr, right], axis=1)


# ============================================================================
# Helper: build / decompose batched 1D conservative vectors
# ============================================================================

def _build_conservative_batch(rho, vpar, vperp1, vperp2, eint):
    """
    Build batched 1D conservative vector.

    Parameters
    ----------
    rho, vpar, vperp1, vperp2, eint : ndarrays, shape (batch, N)

    Returns
    -------
    U : ndarray, shape (3, batch, N)
        [rho, rho*vpar, E_total] where E includes all three KE components.
    """
    U = np.empty((3,) + rho.shape)
    U[0] = rho
    U[1] = rho * vpar
    U[2] = rho * (eint + 0.5 * (vpar ** 2 + vperp1 ** 2 + vperp2 ** 2))
    return U


def _decompose_conservative_batch(U, vperp1, vperp2, eos):
    """
    Extract primitives from batched 1D conservative vector, knowing transverse
    velocities.

    Parameters
    ----------
    U : ndarray, shape (3, batch, N)
    vperp1, vperp2 : ndarrays, shape (batch, N)
        Known transverse velocities (unchanged by this sweep).
    eos : object

    Returns
    -------
    rho, vpar, eint : ndarrays, shape (batch, N)
    """
    rho = np.maximum(U[0], DENSITY_FLOOR)
    vpar = U[1] / rho
    eint = U[2] / rho - 0.5 * (vpar ** 2 + vperp1 ** 2 + vperp2 ** 2)
    eint = np.maximum(eint, 0.0)
    return rho, vpar, eint


# ============================================================================
# Donor-cell passive advection of a transverse velocity (batched)
# ============================================================================

def _upwind_advect_batch(vperp, vpar, rho_old, rho_new, dt, dx):
    """
    Donor-cell (upwind) passive advection of a transverse velocity component.

    Parameters
    ----------
    vperp : ndarray, shape (batch, N)
        Transverse velocity to advect.
    vpar : ndarray, shape (batch, N)
        Parallel (sweep-direction) velocity used for upwinding.
    rho_old : ndarray, shape (batch, N)
        Density before this sweep (for density-ratio correction).
    rho_new : ndarray, shape (batch, N)
        Density after this sweep.
    dt : float
    dx : float

    Returns
    -------
    vperp_updated : ndarray, shape (batch, N)
    """
    # Periodic shifts along the last axis
    vperp_ip = np.roll(vperp, -1, axis=-1)   # vperp[i+1]
    vperp_im = np.roll(vperp, 1, axis=-1)    # vperp[i-1]
    vpar_im = np.roll(vpar, 1, axis=-1)      # vpar[i-1]

    flux_R = np.where(vpar >= 0, vpar * vperp, vpar * vperp_ip)
    flux_L = np.where(vpar_im >= 0, vpar_im * vperp_im, vpar_im * vperp)

    rho_new_safe = np.maximum(rho_new, DENSITY_FLOOR)
    vperp_updated = vperp - (dt / dx) * (flux_R - flux_L) * (rho_old / rho_new_safe)
    return vperp_updated


# ============================================================================
# Directional sweeps (batched)
# ============================================================================

def sweep_x(state, dx, dt, eos):
    """
    X-direction sweep: evolve all (Nz*Ny) x-pencils simultaneously.

    Transverse velocities vy, vz are passively advected with donor-cell.

    Parameters
    ----------
    state : State3D
    dx : float
    dt : float
    eos : object

    Returns
    -------
    state : State3D (modified in-place)
    """
    Nz, Ny, Nx = state.Nz, state.Ny, state.Nx
    batch = Nz * Ny
    nghosts = 2

    # Reshape to (batch, Nx)
    rho = state.rho.reshape(batch, Nx)
    vpar = state.vx.reshape(batch, Nx)
    vperp1 = state.vy.reshape(batch, Nx)
    vperp2 = state.vz.reshape(batch, Nx)
    eint = state.eint.reshape(batch, Nx)

    # Build conservative vector (3, batch, Nx)
    U = _build_conservative_batch(rho, vpar, vperp1, vperp2, eint)

    # Add ghost cells -> (3, batch, Nx + 2*nghosts)
    U_ext = _add_ghost_cells_batch(U, nghosts)

    # HLLE fluxes at Nx+1 interfaces
    U_L = U_ext[:, :, nghosts - 1: nghosts + Nx]
    U_R = U_ext[:, :, nghosts: nghosts + Nx + 1]
    F_interface = _hlle_flux_batch(U_L, U_R, eos)  # (3, batch, Nx+1)

    # Conservative update
    U_updated = U.copy()
    for k in range(3):
        U_updated[k] = U[k] - (dt / dx) * (F_interface[k, :, 1:] - F_interface[k, :, :-1])

    # Floor density
    U_updated[0] = np.maximum(U_updated[0], DENSITY_FLOOR)

    # Extract primitives (transverse velocities unchanged by the Euler solve)
    rho_new, vpar_new, eint_new = _decompose_conservative_batch(
        U_updated, vperp1, vperp2, eos
    )

    # Passive advection of vy and vz
    vy_new = _upwind_advect_batch(vperp1, vpar, rho, rho_new, dt, dx)
    vz_new = _upwind_advect_batch(vperp2, vpar, rho, rho_new, dt, dx)

    # Write back (reshape to 3D)
    state.rho = rho_new.reshape(Nz, Ny, Nx)
    state.vx = vpar_new.reshape(Nz, Ny, Nx)
    state.vy = vy_new.reshape(Nz, Ny, Nx)
    state.vz = vz_new.reshape(Nz, Ny, Nx)
    state.eint = eint_new.reshape(Nz, Ny, Nx)
    return state


def sweep_y(state, dy, dt, eos):
    """
    Y-direction sweep: evolve all (Nz*Nx) y-pencils simultaneously.

    The y-axis is axis=1 of the (Nz, Ny, Nx) arrays. We transpose to make
    the sweep axis last, batch, compute, then transpose back.

    Transverse velocities vx, vz are passively advected with donor-cell.

    Parameters
    ----------
    state : State3D
    dy : float
    dt : float
    eos : object

    Returns
    -------
    state : State3D (modified in-place)
    """
    Nz, Ny, Nx = state.Nz, state.Ny, state.Nx
    batch = Nz * Nx
    nghosts = 2

    # Transpose (Nz, Ny, Nx) -> (Nz, Nx, Ny), then reshape to (batch, Ny)
    rho = state.rho.transpose(0, 2, 1).reshape(batch, Ny)
    vpar = state.vy.transpose(0, 2, 1).reshape(batch, Ny)   # vy is parallel
    vperp1 = state.vx.transpose(0, 2, 1).reshape(batch, Ny)  # vx is transverse
    vperp2 = state.vz.transpose(0, 2, 1).reshape(batch, Ny)  # vz is transverse
    eint = state.eint.transpose(0, 2, 1).reshape(batch, Ny)

    U = _build_conservative_batch(rho, vpar, vperp1, vperp2, eint)
    U_ext = _add_ghost_cells_batch(U, nghosts)

    U_L = U_ext[:, :, nghosts - 1: nghosts + Ny]
    U_R = U_ext[:, :, nghosts: nghosts + Ny + 1]
    F_interface = _hlle_flux_batch(U_L, U_R, eos)

    U_updated = U.copy()
    for k in range(3):
        U_updated[k] = U[k] - (dt / dy) * (F_interface[k, :, 1:] - F_interface[k, :, :-1])

    U_updated[0] = np.maximum(U_updated[0], DENSITY_FLOOR)

    rho_new, vpar_new, eint_new = _decompose_conservative_batch(
        U_updated, vperp1, vperp2, eos
    )

    vx_new = _upwind_advect_batch(vperp1, vpar, rho, rho_new, dt, dy)
    vz_new = _upwind_advect_batch(vperp2, vpar, rho, rho_new, dt, dy)

    # Reshape back to (batch, Ny) -> (Nz, Nx, Ny) -> transpose to (Nz, Ny, Nx)
    state.rho = rho_new.reshape(Nz, Nx, Ny).transpose(0, 2, 1)
    state.vy = vpar_new.reshape(Nz, Nx, Ny).transpose(0, 2, 1)
    state.vx = vx_new.reshape(Nz, Nx, Ny).transpose(0, 2, 1)
    state.vz = vz_new.reshape(Nz, Nx, Ny).transpose(0, 2, 1)
    state.eint = eint_new.reshape(Nz, Nx, Ny).transpose(0, 2, 1)
    return state


def sweep_z(state, dz, dt, eos):
    """
    Z-direction sweep: evolve all (Ny*Nx) z-pencils simultaneously.

    The z-axis is axis=0 of the (Nz, Ny, Nx) arrays. We transpose to make
    the sweep axis last, batch, compute, then transpose back.

    Transverse velocities vx, vy are passively advected with donor-cell.

    Parameters
    ----------
    state : State3D
    dz : float
    dt : float
    eos : object

    Returns
    -------
    state : State3D (modified in-place)
    """
    Nz, Ny, Nx = state.Nz, state.Ny, state.Nx
    batch = Ny * Nx
    nghosts = 2

    # Transpose (Nz, Ny, Nx) -> (Ny, Nx, Nz), then reshape to (batch, Nz)
    rho = state.rho.transpose(1, 2, 0).reshape(batch, Nz)
    vpar = state.vz.transpose(1, 2, 0).reshape(batch, Nz)   # vz is parallel
    vperp1 = state.vx.transpose(1, 2, 0).reshape(batch, Nz)  # vx is transverse
    vperp2 = state.vy.transpose(1, 2, 0).reshape(batch, Nz)  # vy is transverse
    eint = state.eint.transpose(1, 2, 0).reshape(batch, Nz)

    U = _build_conservative_batch(rho, vpar, vperp1, vperp2, eint)
    U_ext = _add_ghost_cells_batch(U, nghosts)

    U_L = U_ext[:, :, nghosts - 1: nghosts + Nz]
    U_R = U_ext[:, :, nghosts: nghosts + Nz + 1]
    F_interface = _hlle_flux_batch(U_L, U_R, eos)

    U_updated = U.copy()
    for k in range(3):
        U_updated[k] = U[k] - (dt / dz) * (F_interface[k, :, 1:] - F_interface[k, :, :-1])

    U_updated[0] = np.maximum(U_updated[0], DENSITY_FLOOR)

    rho_new, vpar_new, eint_new = _decompose_conservative_batch(
        U_updated, vperp1, vperp2, eos
    )

    vx_new = _upwind_advect_batch(vperp1, vpar, rho, rho_new, dt, dz)
    vy_new = _upwind_advect_batch(vperp2, vpar, rho, rho_new, dt, dz)

    # Reshape back: (batch, Nz) -> (Ny, Nx, Nz) -> transpose to (Nz, Ny, Nx)
    state.rho = rho_new.reshape(Ny, Nx, Nz).transpose(2, 0, 1)
    state.vz = vpar_new.reshape(Ny, Nx, Nz).transpose(2, 0, 1)
    state.vx = vx_new.reshape(Ny, Nx, Nz).transpose(2, 0, 1)
    state.vy = vy_new.reshape(Ny, Nx, Nz).transpose(2, 0, 1)
    state.eint = eint_new.reshape(Ny, Nx, Nz).transpose(2, 0, 1)
    return state


# ============================================================================
# Gravity source term
# ============================================================================

def add_gravity_source_3d(state, gx, gy, gz, dt):
    """
    Add gravitational source terms to velocities (operator splitting).

    For isothermal / barotropic EOS where eint is tracked separately,
    only the velocities are updated: dv/dt += g.

    For a general EOS the energy update dE/dt += rho*(v . g) is also needed;
    since the State3D stores eint directly (not total energy), and the
    kinetic-energy change is implicit in the velocity update, we update
    velocities only (consistent with the 2D implementation).

    Parameters
    ----------
    state : State3D
    gx, gy, gz : ndarrays, shape (Nz, Ny, Nx)
        Gravitational acceleration components.
    dt : float

    Returns
    -------
    state : State3D (modified in-place)
    """
    state.vx += dt * gx
    state.vy += dt * gy
    state.vz += dt * gz
    return state


# ============================================================================
# CFL timestep
# ============================================================================

def compute_dt_3d(state, dx, dy, dz, eos, cfl=0.3, G=1.0):
    """
    CFL timestep for 3D, including gravitational free-fall constraint.

    dt = cfl * min(dx/(|vx|+cs), dy/(|vy|+cs), dz/(|vz|+cs))

    Also enforces dt < cfl / sqrt(4 pi G rho_max) for self-gravitating flows.

    Parameters
    ----------
    state : State3D
    dx, dy, dz : float
        Cell widths.
    eos : object
        Equation of state (must have .cs attribute for isothermal/tau EOS).
    cfl : float
        CFL number. Default 0.3.
    G : float
        Gravitational constant. Default 1.0.

    Returns
    -------
    dt : float
        Maximum stable timestep.
    """
    cs = eos.cs

    max_speed_x = np.max(np.abs(state.vx)) + cs
    max_speed_y = np.max(np.abs(state.vy)) + cs
    max_speed_z = np.max(np.abs(state.vz)) + cs

    max_speed_x = max(max_speed_x, 1e-30)
    max_speed_y = max(max_speed_y, 1e-30)
    max_speed_z = max(max_speed_z, 1e-30)

    dt_cfl = cfl * min(dx / max_speed_x, dy / max_speed_y, dz / max_speed_z)

    # Gravitational timescale
    rho_max = np.max(state.rho)
    if rho_max > 0 and G > 0:
        dt_grav = cfl / np.sqrt(4.0 * np.pi * G * rho_max)
    else:
        dt_grav = dt_cfl

    return min(dt_cfl, dt_grav)
