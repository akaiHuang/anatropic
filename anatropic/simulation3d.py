"""
3D simulation driver for the Anatropic project.

Ties together the 3D Euler solver (euler3d), 3D Poisson gravity solver
(gravity3d), and equation of state into a single high-level interface
for running 3D self-gravitating fluid simulations.

Uses Strang splitting for gravity:
    half-gravity -> X-Y-Z sweeps -> half-gravity
with sweep order alternating each timestep (XYZ/ZYX) for symmetry.
"""

import time as walltime
import numpy as np

from .euler3d import State3D, sweep_x, sweep_y, sweep_z
from .euler3d import compute_dt_3d, add_gravity_source_3d
from .gravity3d import solve_gravity_3d


class Simulation3D:
    """
    3D self-gravitating fluid simulation.

    Manages the grid, primitive-variable state (State3D), equation of state,
    gravity solver, timestepping, and snapshot output. Supports arbitrary
    barotropic EOS and optional self-gravity with Strang operator splitting.

    Typical usage::

        from anatropic.eos import IsothermalEOS
        from anatropic.simulation3d import Simulation3D

        sim = Simulation3D()
        sim.setup(Nx=64, Ny=64, Nz=64, Lx=1.0, Ly=1.0, Lz=1.0,
                  rho0=1.0, eos=IsothermalEOS(cs=0.05))
        sim.add_random_perturbation(amplitude=1e-4, seed=42)
        sim.run(t_end=0.84, cfl=0.3)

    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    Lx, Ly, Lz : float
        Box sizes.
    dx, dy, dz : float
        Cell widths.
    state : State3D
        Primitive-variable state on the 3D grid.
    eos : object
        Equation of state.
    use_gravity : bool
        Whether self-gravity is enabled.
    G : float
        Gravitational constant (code units).
    t : float
        Current simulation time.
    step : int
        Current step number.
    snapshots : list of dict
        Saved snapshots during run().
    """

    def __init__(self):
        self.Nx = self.Ny = self.Nz = 0
        self.Lx = self.Ly = self.Lz = 0.0
        self.dx = self.dy = self.dz = 0.0
        self.state = None
        self.eos = None
        self.use_gravity = False
        self.G = 1.0
        self.t = 0.0
        self.step = 0
        self.snapshots = []

    def setup(self, Nx, Ny, Nz, Lx, Ly, Lz, rho0, eos,
              use_gravity=True, G=1.0):
        """
        Initialize uniform grid and State3D.

        Creates a uniform-density, zero-velocity, hydrostatic initial state.
        Internal energy is set consistently with the EOS.

        Parameters
        ----------
        Nx, Ny, Nz : int
            Number of cells in each direction.
        Lx, Ly, Lz : float
            Box sizes (periodic domain).
        rho0 : float
            Uniform background density.
        eos : object
            Equation of state (must implement pressure, sound_speed,
            internal_energy, and have a .cs attribute for barotropic EOS).
        use_gravity : bool, optional
            Enable self-gravity. Default True.
        G : float, optional
            Gravitational constant. Default 1.0.

        Returns
        -------
        self : Simulation3D
            Returns self for method chaining.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dz = Lz / Nz
        self.eos = eos
        self.use_gravity = use_gravity
        self.G = G
        self.t = 0.0
        self.step = 0
        self.snapshots = []

        # Create uniform initial state
        self.state = State3D(Nz, Ny, Nx, rho0, eos)

        return self

    def add_perturbation_mode(self, amplitude, mode_x=0, mode_y=0, mode_z=0):
        """
        Add a sinusoidal density perturbation.

        delta_rho = amplitude * rho0 * sin(2*pi*mode_x*x/Lx)
                                      * sin(2*pi*mode_y*y/Ly)
                                      * sin(2*pi*mode_z*z/Lz)

        For modes that are zero, the corresponding factor is 1 (no modulation
        in that direction).

        Parameters
        ----------
        amplitude : float
            Fractional amplitude (e.g. 0.01 = 1%).
        mode_x, mode_y, mode_z : int, optional
            Mode numbers in each direction. Default 0 (no modulation).

        Returns
        -------
        self : Simulation3D
            Returns self for method chaining.
        """
        if self.state is None:
            raise RuntimeError("Call setup() before add_perturbation_mode().")

        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        x = (np.arange(Nx) + 0.5) * self.dx
        y = (np.arange(Ny) + 0.5) * self.dy
        z = (np.arange(Nz) + 0.5) * self.dz

        # Build 3D perturbation via outer products
        if mode_x > 0:
            fx = np.sin(2.0 * np.pi * mode_x * x / self.Lx)
        else:
            fx = np.ones(Nx)

        if mode_y > 0:
            fy = np.sin(2.0 * np.pi * mode_y * y / self.Ly)
        else:
            fy = np.ones(Ny)

        if mode_z > 0:
            fz = np.sin(2.0 * np.pi * mode_z * z / self.Lz)
        else:
            fz = np.ones(Nz)

        # (Nz, Ny, Nx) perturbation
        pert = amplitude * np.einsum('i,j,k->ijk', fz, fy, fx)
        self.state.rho *= (1.0 + pert)
        self.state.rho = np.maximum(self.state.rho, 1e-30)

        # Recompute eint for consistency
        P = self.eos.pressure(self.state.rho, np.zeros_like(self.state.rho))
        self.state.eint = self.eos.internal_energy(self.state.rho, P)

        return self

    def add_random_perturbation(self, amplitude, seed=None):
        """
        Add random Gaussian density perturbation.

        delta_rho = amplitude * rho0 * randn(Nz, Ny, Nx)

        Parameters
        ----------
        amplitude : float
            RMS fractional amplitude (e.g. 1e-4).
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        self : Simulation3D
            Returns self for method chaining.
        """
        if self.state is None:
            raise RuntimeError("Call setup() before add_random_perturbation().")

        if seed is not None:
            np.random.seed(seed)

        rho0 = self.state.rho.copy()
        delta = amplitude * rho0 * np.random.randn(self.Nz, self.Ny, self.Nx)
        self.state.rho = np.maximum(rho0 + delta, 1e-30)

        # Recompute eint for consistency
        P = self.eos.pressure(self.state.rho, np.zeros_like(self.state.rho))
        self.state.eint = self.eos.internal_energy(self.state.rho, P)

        return self

    def _gravity_kick(self, dt):
        """Apply gravity source term for duration dt."""
        if not self.use_gravity or dt == 0:
            return
        gx, gy, gz = solve_gravity_3d(
            self.state.rho, self.dx, self.dy, self.dz, G=self.G
        )
        add_gravity_source_3d(self.state, gx, gy, gz, dt)

    def run(self, t_end, cfl=0.3, max_steps=1000000, print_every=100,
            snapshot_interval=None, max_walltime_s=3600):
        """
        Run with Strang splitting: half-gravity, X-Y-Z sweeps, half-gravity.

        Sweep order alternates each timestep (XYZ vs ZYX) for symmetry and
        reduced directional bias.

        Parameters
        ----------
        t_end : float
            Final simulation time.
        cfl : float, optional
            CFL number. Default 0.3.
        max_steps : int, optional
            Maximum number of timesteps. Default 1000000.
        print_every : int, optional
            Print progress every this many steps. Default 100.
        snapshot_interval : float or None, optional
            If set, save snapshots at this time interval.
        max_walltime_s : float, optional
            Maximum wall-clock time in seconds. Default 3600 (1 hour).

        Returns
        -------
        self : Simulation3D
            Returns self for post-analysis.
        """
        if self.state is None:
            raise RuntimeError("Call setup() before run().")

        t_wall_start = walltime.time()

        if snapshot_interval is not None and snapshot_interval > 0:
            t_next_snap = self.t + snapshot_interval
        else:
            t_next_snap = None

        # Save initial snapshot
        if snapshot_interval is not None:
            self._save_snapshot()

        while self.t < t_end and self.step < max_steps:
            # CFL timestep
            dt = compute_dt_3d(
                self.state, self.dx, self.dy, self.dz,
                self.eos, cfl=cfl, G=self.G
            )

            # Don't overshoot
            dt = min(dt, t_end - self.t)
            if snapshot_interval is not None and t_next_snap is not None:
                dt = min(dt, t_next_snap - self.t)

            if dt <= 0:
                break

            # === Strang splitting ===

            # 1. Half-step gravity kick
            self._gravity_kick(0.5 * dt)

            # 2. Dimensional sweeps (alternate order)
            if self.step % 2 == 0:
                # XYZ order
                sweep_x(self.state, self.dx, dt, self.eos)
                sweep_y(self.state, self.dy, dt, self.eos)
                sweep_z(self.state, self.dz, dt, self.eos)
            else:
                # ZYX order
                sweep_z(self.state, self.dz, dt, self.eos)
                sweep_y(self.state, self.dy, dt, self.eos)
                sweep_x(self.state, self.dx, dt, self.eos)

            # 3. Half-step gravity kick (with updated density)
            self._gravity_kick(0.5 * dt)

            self.t += dt
            self.step += 1

            # Snapshot
            if t_next_snap is not None and self.t >= t_next_snap - 1e-14:
                self._save_snapshot()
                t_next_snap += snapshot_interval

            # Progress report
            if self.step % print_every == 0:
                elapsed = walltime.time() - t_wall_start
                rho_max = np.max(self.state.rho)
                rho_min = np.min(self.state.rho)
                t_ff = self.compute_jeans_time()
                print(
                    f"  step {self.step:>7d} | "
                    f"t = {self.t:.4e} ({self.t/t_ff:.2f} t_ff) | "
                    f"dt = {dt:.2e} | "
                    f"rho [{rho_min:.4f}, {rho_max:.4f}] | "
                    f"wall {elapsed:.1f}s"
                )

            # Wall-clock safety
            if walltime.time() - t_wall_start > max_walltime_s:
                print(f"  ** Wall-clock timeout at step {self.step}, "
                      f"t = {self.t:.4e} **")
                break

        # Final snapshot
        if snapshot_interval is not None:
            if (len(self.snapshots) == 0
                    or abs(self.snapshots[-1]['t'] - self.t) > 1e-14):
                self._save_snapshot()

        elapsed = walltime.time() - t_wall_start
        t_ff = self.compute_jeans_time()
        print(f"\n  Simulation complete: {self.step} steps, {elapsed:.1f}s")
        print(f"  Final time: t = {self.t:.4e} ({self.t/t_ff:.2f} t_ff)")
        rho = self.state.rho
        print(f"  Density range: [{np.min(rho):.4e}, {np.max(rho):.4e}]")
        print(f"  Density contrast: {np.max(rho)/max(np.min(rho), 1e-30):.2e}")

        return self

    def _save_snapshot(self):
        """Store current state as a snapshot dict."""
        self.snapshots.append({
            't': self.t,
            'step': self.step,
            'rho': self.state.rho.copy(),
            'vx': self.state.vx.copy(),
            'vy': self.state.vy.copy(),
            'vz': self.state.vz.copy(),
            'eint': self.state.eint.copy(),
        })

    def save_snapshot(self, filename):
        """
        Save current state to a compressed npz file.

        Parameters
        ----------
        filename : str
            Output file path (should end in .npz).
        """
        np.savez_compressed(
            filename,
            t=self.t,
            step=self.step,
            rho=self.state.rho,
            vx=self.state.vx,
            vy=self.state.vy,
            vz=self.state.vz,
            eint=self.state.eint,
            Nx=self.Nx, Ny=self.Ny, Nz=self.Nz,
            Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
            G=self.G,
        )

    def get_power_spectrum_3d(self):
        """
        Compute spherically-averaged 3D power spectrum P(k).

        The density contrast delta = (rho - <rho>) / <rho> is Fourier
        transformed, and |delta_hat|^2 is averaged in spherical shells of
        constant |k|.

        Returns
        -------
        k_bins : ndarray
            Bin-center wavenumbers.
        P_k : ndarray
            Spherically-averaged power per bin.
        """
        rho = self.state.rho
        Nz, Ny, Nx = rho.shape

        # Density contrast
        rho_mean = np.mean(rho)
        delta = (rho - rho_mean) / rho_mean

        # 3D FFT (normalized)
        delta_hat = np.fft.fftn(delta) / (Nx * Ny * Nz)
        power_3d = np.abs(delta_hat) ** 2

        # Wavenumber grid
        kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=self.dy)
        kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=self.dz)

        # Build 3D |k| array
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        # meshgrid with 'ij' gives (Nx, Ny, Nz) -> transpose to (Nz, Ny, Nx)
        KX = KX.transpose(2, 1, 0)
        KY = KY.transpose(2, 1, 0)
        KZ = KZ.transpose(2, 1, 0)
        K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        # Bin by |k|
        dk = 2.0 * np.pi / min(self.Lx, self.Ly, self.Lz)
        k_nyquist = np.pi / max(self.dx, self.dy, self.dz)
        k_edges = np.arange(0.5 * dk, k_nyquist + dk, dk)
        k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
        P_k = np.zeros(len(k_bins))

        for i in range(len(k_bins)):
            mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
            if np.any(mask):
                P_k[i] = np.mean(power_3d[mask])

        # Remove empty bins
        valid = P_k > 0
        return k_bins[valid], P_k[valid]

    def compute_jeans_time(self):
        """
        Estimate the Jeans (free-fall) time for the current mean density.

        t_ff = 1 / sqrt(4 * pi * G * rho0)

        Returns
        -------
        t_ff : float
        """
        rho0 = np.mean(self.state.rho)
        return 1.0 / np.sqrt(4.0 * np.pi * self.G * rho0)

    def compute_jeans_length(self):
        """
        Estimate the Jeans length for the current mean density and sound speed.

        lambda_J = c_s * sqrt(pi / (G * rho0))

        Returns
        -------
        lambda_J : float
        """
        rho0 = np.mean(self.state.rho)
        cs = self.eos.cs
        return cs * np.sqrt(np.pi / (self.G * rho0))

    def get_midplane_slice(self, axis='z'):
        """
        Return the density on the midplane orthogonal to the given axis.

        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'. Default 'z'.

        Returns
        -------
        rho_slice : ndarray, shape (N1, N2)
            2D density slice.
        """
        rho = self.state.rho  # (Nz, Ny, Nx)
        if axis == 'z':
            return rho[self.Nz // 2, :, :].copy()
        elif axis == 'y':
            return rho[:, self.Ny // 2, :].copy()
        elif axis == 'x':
            return rho[:, :, self.Nx // 2].copy()
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
