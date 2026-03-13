"""
1D compressible Euler equations solver using Godunov method with HLLE Riemann solver.

The conservative variables are U = [rho, rho*v, E] where E = rho*e + 0.5*rho*v^2.

The HLLE solver is chosen for its robustness, particularly in the low sound-speed
(dust) limit where c_s -> 0. A pressure floor prevents negative pressures from
arising in near-vacuum regions.
"""

import numpy as np

# Absolute pressure floor to prevent negative pressures
PRESSURE_FLOOR = 1e-20

# Absolute density floor to prevent negative densities
DENSITY_FLOOR = 1e-30


def _primitive_from_conservative(U, eos):
    """
    Extract primitive variables from conservative state vector.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative variables [rho, rho*v, E].
    eos : object
        Equation of state with pressure(rho, eint) and sound_speed(rho, eint).

    Returns
    -------
    rho : ndarray, shape (N,)
        Mass density.
    v : ndarray, shape (N,)
        Velocity.
    P : ndarray, shape (N,)
        Pressure.
    cs : ndarray, shape (N,)
        Sound speed.
    eint : ndarray, shape (N,)
        Specific internal energy.
    """
    rho = np.maximum(U[0], DENSITY_FLOOR)
    v = U[1] / rho
    eint = U[2] / rho - 0.5 * v**2
    eint = np.maximum(eint, 0.0)

    P = eos.pressure(rho, eint)
    P = np.maximum(P, PRESSURE_FLOOR)
    cs = eos.sound_speed(rho, eint)
    cs = np.maximum(cs, 0.0)

    return rho, v, P, cs, eint


def _flux(rho, v, P, E):
    """
    Compute the Euler flux vector F(U).

    F = [rho*v, rho*v^2 + P, (E + P)*v]

    Parameters
    ----------
    rho : ndarray
        Density.
    v : ndarray
        Velocity.
    P : ndarray
        Pressure.
    E : ndarray
        Total energy density.

    Returns
    -------
    F : ndarray, shape (3, N)
        Flux vector.
    """
    F = np.zeros((3, len(rho)))
    F[0] = rho * v
    F[1] = rho * v**2 + P
    F[2] = (E + P) * v
    return F


def _hlle_flux(U_L, U_R, eos):
    """
    HLLE approximate Riemann solver for the 1D Euler equations.

    Computes the numerical flux at an interface given left and right states.
    Handles the dust limit (c_s -> 0) gracefully by allowing wave speeds
    to degenerate.

    Parameters
    ----------
    U_L : ndarray, shape (3, N_interfaces)
        Conservative variables on the left side of each interface.
    U_R : ndarray, shape (3, N_interfaces)
        Conservative variables on the right side of each interface.
    eos : object
        Equation of state.

    Returns
    -------
    F_hlle : ndarray, shape (3, N_interfaces)
        HLLE numerical flux at each interface.
    """
    rho_L, v_L, P_L, cs_L, _ = _primitive_from_conservative(U_L, eos)
    rho_R, v_R, P_R, cs_R, _ = _primitive_from_conservative(U_R, eos)

    E_L = U_L[2]
    E_R = U_R[2]

    # Wave speed estimates (Davis estimates)
    S_L = np.minimum(v_L - cs_L, v_R - cs_R)
    S_R = np.maximum(v_L + cs_L, v_R + cs_R)

    # In the dust limit (cs -> 0), S_L and S_R collapse to the flow velocities.
    # The HLLE solver remains well-defined in this limit.

    F_L = _flux(rho_L, v_L, P_L, E_L)
    F_R = _flux(rho_R, v_R, P_R, E_R)

    # HLLE flux
    F_hlle = np.zeros_like(F_L)

    # Regions: S_L > 0 (supersonic to the right), S_R < 0 (supersonic to the left),
    # otherwise use the HLLE average.
    for i in range(3):
        # Default: HLLE intermediate state
        denom = S_R - S_L
        # Prevent division by zero when S_L == S_R (static dust)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

        F_hlle[i] = (S_R * F_L[i] - S_L * F_R[i] + S_L * S_R * (U_R[i] - U_L[i])) / denom

        # Supersonic cases
        F_hlle[i] = np.where(S_L >= 0, F_L[i], F_hlle[i])
        F_hlle[i] = np.where(S_R <= 0, F_R[i], F_hlle[i])

    return F_hlle


def _add_ghost_cells(U, nghosts=2):
    """
    Add ghost cells for periodic boundary conditions.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative variables on the physical domain.
    nghosts : int
        Number of ghost cells on each side.

    Returns
    -------
    U_ext : ndarray, shape (3, N + 2*nghosts)
        Extended state with ghost cells filled by periodic wrapping.
    """
    N = U.shape[1]
    U_ext = np.zeros((3, N + 2 * nghosts))
    # Physical cells
    U_ext[:, nghosts:nghosts + N] = U
    # Left ghosts (wrap from right end)
    U_ext[:, :nghosts] = U[:, -nghosts:]
    # Right ghosts (wrap from left end)
    U_ext[:, nghosts + N:] = U[:, :nghosts]
    return U_ext


def compute_dt(U, dx, eos, cfl=0.5):
    """
    Compute the CFL-limited timestep.

    dt = cfl * dx / max(|v| + c_s)

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative variables.
    dx : float
        Cell width.
    eos : object
        Equation of state.
    cfl : float, optional
        CFL number, must be in (0, 1). Default is 0.5.

    Returns
    -------
    dt : float
        Maximum stable timestep.
    """
    rho, v, P, cs, eint = _primitive_from_conservative(U, eos)
    max_speed = np.max(np.abs(v) + cs)

    if max_speed < 1e-30:
        # All speeds are essentially zero; return a large but finite dt
        return cfl * dx / 1e-30

    return cfl * dx / max_speed


def evolve(U, dx, dt, eos, gravity_source=None):
    """
    Advance the conservative variables by one timestep using Godunov's method.

    Uses the HLLE Riemann solver with periodic boundary conditions (ghost cells).
    Optionally includes gravitational source terms via operator splitting.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative variables [rho, rho*v, E] on N cells.
    dx : float
        Cell width (uniform grid).
    dt : float
        Timestep.
    eos : object
        Equation of state providing pressure() and sound_speed().
    gravity_source : ndarray or None, shape (N,)
        Gravitational acceleration g = -dPhi/dx at cell centres.
        If None, no gravitational source terms are added.

    Returns
    -------
    U_new : ndarray, shape (3, N)
        Updated conservative variables after one timestep.
    """
    N = U.shape[1]
    nghosts = 2

    # Step 1: Add ghost cells (periodic BC)
    U_ext = _add_ghost_cells(U, nghosts)

    # Step 2: Compute HLLE fluxes at all N+1 interfaces
    # Interface i sits between cell i-1 and cell i (in physical indexing).
    # In the extended array, physical cell 0 is at index nghosts.
    # Interface i (physical) is between ext index (nghosts + i - 1) and (nghosts + i).
    # We need N+1 interfaces: i = 0, 1, ..., N.
    U_L = U_ext[:, nghosts - 1: nghosts + N]       # cells 0..N-1 in extended
    U_R = U_ext[:, nghosts: nghosts + N + 1]        # cells 1..N in extended
    F_interface = _hlle_flux(U_L, U_R, eos)

    # Step 3: Conservative update: U^{n+1} = U^n - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
    U_new = U.copy()
    for k in range(3):
        U_new[k] = U[k] - (dt / dx) * (F_interface[k, 1:] - F_interface[k, :-1])

    # Step 4: Add gravitational source terms (operator splitting)
    if gravity_source is not None:
        g = gravity_source
        rho = np.maximum(U_new[0], DENSITY_FLOOR)
        v = U_new[1] / rho

        # Momentum source: d(rho*v)/dt += rho * g
        U_new[1] += dt * rho * g
        # Energy source: dE/dt += rho * v * g
        U_new[2] += dt * rho * v * g

    # Step 5: Apply floors
    U_new[0] = np.maximum(U_new[0], DENSITY_FLOOR)

    # Ensure internal energy is non-negative
    rho_new = U_new[0]
    v_new = U_new[1] / rho_new
    eint_new = U_new[2] / rho_new - 0.5 * v_new**2
    eint_new = np.maximum(eint_new, 0.0)
    U_new[2] = rho_new * (eint_new + 0.5 * v_new**2)

    return U_new
