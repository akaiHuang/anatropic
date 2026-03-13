"""
Main simulation driver for the Anatropic project.

Ties together the Euler solver, Poisson gravity solver, and equation of state
into a single high-level interface for running 1D self-gravitating fluid
simulations.
"""

import numpy as np

from . import euler
from . import gravity


class Simulation:
    """
    1D self-gravitating fluid simulation.

    Manages the grid, conservative variables, equation of state, gravity solver,
    timestepping, and output history. Supports arbitrary EOS and optional
    self-gravity.

    Typical usage::

        from anatropic.eos import IsothermalEOS
        from anatropic.simulation import Simulation

        sim = Simulation()
        sim.setup(N=256, L=1.0, rho0=1.0, eos=IsothermalEOS(cs=1.0))
        sim.add_perturbation(amplitude=0.01, mode=1)
        history = sim.run(t_end=1.0, output_interval=0.1)

    Attributes
    ----------
    N : int
        Number of grid cells.
    L : float
        Box size.
    dx : float
        Cell width (= L / N).
    x : ndarray, shape (N,)
        Cell-centre coordinates.
    U : ndarray, shape (3, N)
        Conservative variables [rho, rho*v, E].
    eos : object
        Equation of state.
    use_gravity : bool
        Whether self-gravity is enabled.
    G : float
        Gravitational constant (code units).
    t : float
        Current simulation time.
    history : list of dict
        Snapshots stored during run().
    """

    def __init__(self):
        self.N = 0
        self.L = 0.0
        self.dx = 0.0
        self.x = None
        self.U = None
        self.eos = None
        self.use_gravity = False
        self.G = 1.0
        self.t = 0.0
        self.history = []

    def setup(self, N, L, rho0, eos, use_gravity=True, G=1.0):
        """
        Initialise the simulation grid and uniform initial conditions.

        Creates a uniform-density, zero-velocity, hydrostatic initial state.
        The internal energy is set consistently with the EOS: for ideal gas,
        e_int = P / ((gamma-1)*rho); for isothermal / tau EOS, e_int = c_s^2.

        Parameters
        ----------
        N : int
            Number of grid cells.
        L : float
            Box size (periodic domain [0, L]).
        rho0 : float
            Uniform background density.
        eos : object
            Equation of state (must implement pressure, sound_speed,
            internal_energy).
        use_gravity : bool, optional
            Enable self-gravity. Default is True.
        G : float, optional
            Gravitational constant. Default is 1.0 (code units).

        Returns
        -------
        self : Simulation
            Returns self for method chaining.
        """
        self.N = N
        self.L = L
        self.dx = L / N
        self.x = (np.arange(N) + 0.5) * self.dx  # cell centres
        self.eos = eos
        self.use_gravity = use_gravity
        self.G = G
        self.t = 0.0
        self.history = []

        # Uniform initial state: rho = rho0, v = 0
        rho = np.full(N, rho0)
        v = np.zeros(N)

        # Set internal energy consistent with the EOS.
        # For barotropic EOS (isothermal, tau), pressure depends only on rho;
        # we use internal_energy(rho, P) to get a consistent e_int.
        P0 = eos.pressure(rho, np.zeros(N))
        eint = eos.internal_energy(rho, P0)

        # Conservative variables: [rho, rho*v, E]
        # E = rho * (e_int + 0.5 * v^2)
        self.U = np.zeros((3, N))
        self.U[0] = rho
        self.U[1] = rho * v
        self.U[2] = rho * (eint + 0.5 * v**2)

        return self

    def add_perturbation(self, amplitude, mode=1):
        """
        Add a sinusoidal density perturbation to the initial conditions.

        The perturbation is:
            delta_rho = amplitude * rho0 * sin(2*pi*mode*x / L)

        Velocity is left unchanged. Internal energy is recalculated from
        the perturbed density to maintain consistency with the EOS.

        Parameters
        ----------
        amplitude : float
            Fractional amplitude of the density perturbation (e.g. 0.01 = 1%).
        mode : int, optional
            Fourier mode number (number of wavelengths in the box).
            Default is 1.

        Returns
        -------
        self : Simulation
            Returns self for method chaining.
        """
        if self.U is None:
            raise RuntimeError("Call setup() before add_perturbation().")

        rho0 = self.U[0].copy()
        delta_rho = amplitude * rho0 * np.sin(
            2.0 * np.pi * mode * self.x / self.L
        )
        rho_new = rho0 + delta_rho

        # Ensure density stays positive
        rho_new = np.maximum(rho_new, 1e-30)

        # Keep velocity unchanged
        v = self.U[1] / np.maximum(self.U[0], 1e-30)

        # Recompute internal energy from the perturbed density
        P_new = self.eos.pressure(rho_new, np.zeros(self.N))
        eint_new = self.eos.internal_energy(rho_new, P_new)

        self.U[0] = rho_new
        self.U[1] = rho_new * v
        self.U[2] = rho_new * (eint_new + 0.5 * v**2)

        return self

    def get_state(self):
        """
        Return the current primitive state of the simulation.

        Returns
        -------
        state : dict
            Dictionary with keys:
            - 'x'    : cell centres, shape (N,)
            - 'rho'  : density, shape (N,)
            - 'v'    : velocity, shape (N,)
            - 'P'    : pressure, shape (N,)
            - 'eint' : specific internal energy, shape (N,)
            - 't'    : current time (float)
        """
        if self.U is None:
            raise RuntimeError("Call setup() before get_state().")

        rho = np.maximum(self.U[0], 1e-30)
        v = self.U[1] / rho
        eint = self.U[2] / rho - 0.5 * v**2
        eint = np.maximum(eint, 0.0)
        P = self.eos.pressure(rho, eint)

        return {
            'x': self.x.copy(),
            'rho': rho.copy(),
            'v': v.copy(),
            'P': P.copy(),
            'eint': eint.copy(),
            't': self.t,
        }

    def _save_snapshot(self):
        """Store the current state in the history list."""
        self.history.append(self.get_state())

    def run(self, t_end, output_interval=None, cfl=0.5, max_steps=10_000_000):
        """
        Run the simulation from the current time to t_end.

        Parameters
        ----------
        t_end : float
            Final simulation time.
        output_interval : float or None, optional
            Time interval between saved snapshots. If None, only the initial
            and final states are saved.
        cfl : float, optional
            CFL number for timestepping. Default is 0.5.
        max_steps : int, optional
            Maximum number of timesteps (safety limit). Default is 10^7.

        Returns
        -------
        history : list of dict
            List of snapshot dictionaries (same format as get_state()).
        """
        if self.U is None:
            raise RuntimeError("Call setup() before run().")

        # Save initial state
        self._save_snapshot()

        if output_interval is not None and output_interval > 0:
            t_next_output = self.t + output_interval
        else:
            t_next_output = t_end

        step = 0
        while self.t < t_end and step < max_steps:
            # Compute CFL timestep
            dt = euler.compute_dt(self.U, self.dx, self.eos, cfl=cfl)

            # Don't overshoot t_end or next output time
            dt = min(dt, t_end - self.t)
            if output_interval is not None and output_interval > 0:
                dt = min(dt, t_next_output - self.t)

            if dt <= 0:
                break

            # Compute gravitational acceleration if needed
            grav_accel = None
            if self.use_gravity:
                rho = np.maximum(self.U[0], 1e-30)
                grav_accel = gravity.solve_gravity(rho, self.dx, G=self.G)

            # Evolve one timestep
            self.U = euler.evolve(
                self.U, self.dx, dt, self.eos, gravity_source=grav_accel
            )

            self.t += dt
            step += 1

            # Save snapshot at output intervals
            if output_interval is not None and output_interval > 0:
                if self.t >= t_next_output - 1e-14 * max(t_end, 1.0):
                    self._save_snapshot()
                    t_next_output += output_interval

        # Save final state if not already saved
        if (len(self.history) == 0
                or abs(self.history[-1]['t'] - self.t) > 1e-14 * max(t_end, 1.0)):
            self._save_snapshot()

        return self.history

    def get_density_history(self):
        """
        Extract density arrays from all saved snapshots.

        Returns
        -------
        times : ndarray, shape (N_snapshots,)
            Time of each snapshot.
        densities : ndarray, shape (N_snapshots, N)
            Density field at each snapshot time.
        """
        if not self.history:
            raise RuntimeError(
                "No snapshots available. Run the simulation first."
            )

        times = np.array([s['t'] for s in self.history])
        densities = np.array([s['rho'] for s in self.history])
        return times, densities

    def compute_jeans_time(self):
        """
        Estimate the Jeans (free-fall) time for the current mean density.

        t_J = 1 / sqrt(4 * pi * G * rho0)

        This is the characteristic timescale for gravitational collapse
        in a self-gravitating medium.

        Returns
        -------
        t_jeans : float
            Jeans time in code units.
        """
        rho0 = np.mean(self.U[0])
        return 1.0 / np.sqrt(4.0 * np.pi * self.G * rho0)

    def compute_jeans_length(self):
        """
        Estimate the Jeans length for the current mean density and sound speed.

        lambda_J = c_s * sqrt(pi / (G * rho0))

        Perturbations with wavelength > lambda_J are gravitationally unstable.

        Returns
        -------
        lambda_jeans : float
            Jeans length in code units.
        """
        rho0 = np.mean(self.U[0])
        eint = np.mean(
            self.U[2] / self.U[0]
            - 0.5 * (self.U[1] / self.U[0]) ** 2
        )
        eint = max(eint, 0.0)
        cs = np.mean(
            self.eos.sound_speed(np.array([rho0]), np.array([eint]))
        )
        return cs * np.sqrt(np.pi / (self.G * rho0))
