"""
Equation of state module for the Anatropic project.

Provides three EOS implementations:
- IdealGasEOS:   standard ideal gas with adiabatic index gamma
- IsothermalEOS: constant sound speed (isothermal approximation)
- TauEOS:        scale-dependent EOS from the tau framework, c_s^2 = (mu_0/k_eff)^2

Each EOS class implements three methods with a uniform interface:
- pressure(rho, eint)    : compute pressure from density and specific internal energy
- sound_speed(rho, eint) : compute adiabatic sound speed
- internal_energy(rho, P): compute specific internal energy from density and pressure
"""

import numpy as np


class IdealGasEOS:
    """
    Ideal gas equation of state: P = (gamma - 1) * rho * e_int.

    Parameters
    ----------
    gamma : float
        Adiabatic index (ratio of specific heats). Default is 5/3.
    """

    def __init__(self, gamma=5.0 / 3.0):
        if gamma <= 1.0:
            raise ValueError("Adiabatic index gamma must be > 1.")
        self.gamma = gamma

    def pressure(self, rho, eint):
        """
        Compute pressure.

        Parameters
        ----------
        rho : ndarray
            Mass density.
        eint : ndarray
            Specific internal energy (energy per unit mass).

        Returns
        -------
        P : ndarray
            Pressure.
        """
        return (self.gamma - 1.0) * rho * eint

    def sound_speed(self, rho, eint):
        """
        Compute adiabatic sound speed: c_s = sqrt(gamma * P / rho).

        Parameters
        ----------
        rho : ndarray
            Mass density.
        eint : ndarray
            Specific internal energy.

        Returns
        -------
        cs : ndarray
            Sound speed.
        """
        P = self.pressure(rho, eint)
        P = np.maximum(P, 0.0)
        rho_safe = np.maximum(rho, 1e-30)
        return np.sqrt(self.gamma * P / rho_safe)

    def internal_energy(self, rho, P):
        """
        Compute specific internal energy from pressure.

        Parameters
        ----------
        rho : ndarray
            Mass density.
        P : ndarray
            Pressure.

        Returns
        -------
        eint : ndarray
            Specific internal energy.
        """
        rho_safe = np.maximum(rho, 1e-30)
        return P / ((self.gamma - 1.0) * rho_safe)

    def __repr__(self):
        return f"IdealGasEOS(gamma={self.gamma})"


class IsothermalEOS:
    """
    Isothermal equation of state: P = c_s^2 * rho.

    The internal energy is tracked for compatibility with the Euler solver
    (which carries total energy E = rho*e + 0.5*rho*v^2) but the pressure
    depends only on density. The internal energy is formally e_int = c_s^2,
    a constant.

    Parameters
    ----------
    cs : float
        Constant sound speed.
    """

    def __init__(self, cs):
        if cs < 0:
            raise ValueError("Sound speed must be non-negative.")
        self.cs = cs
        self.cs2 = cs * cs

    def pressure(self, rho, eint):
        """
        Compute pressure: P = c_s^2 * rho.

        The internal energy argument is accepted for interface compatibility
        but ignored (the EOS is barotropic).

        Parameters
        ----------
        rho : ndarray
            Mass density.
        eint : ndarray
            Specific internal energy (ignored).

        Returns
        -------
        P : ndarray
            Pressure.
        """
        return self.cs2 * rho

    def sound_speed(self, rho, eint):
        """
        Return the constant sound speed.

        Parameters
        ----------
        rho : ndarray
            Mass density (ignored).
        eint : ndarray
            Specific internal energy (ignored).

        Returns
        -------
        cs : ndarray
            Sound speed (constant, broadcast to match rho shape).
        """
        return np.full_like(rho, self.cs, dtype=float)

    def internal_energy(self, rho, P):
        """
        Compute specific internal energy consistent with isothermal EOS.

        For an isothermal gas, e_int = c_s^2 (constant).

        Parameters
        ----------
        rho : ndarray
            Mass density.
        P : ndarray
            Pressure (ignored).

        Returns
        -------
        eint : ndarray
            Specific internal energy (= c_s^2 everywhere).
        """
        return np.full_like(rho, self.cs2, dtype=float)

    def __repr__(self):
        return f"IsothermalEOS(cs={self.cs})"


class TauEOS:
    """
    Scale-dependent equation of state from the tau framework.

    In the tau framework, the effective sound speed is scale-dependent:

        c_s^2(k) = (mu_0 / k)^2

    where mu_0 has dimensions of inverse length (the Khronon coupling constant,
    with mu_0 = H_0/c at cosmological scales).

    In real space with a single effective wavenumber k_eff, this reduces to a
    barotropic EOS:

        P = c_s^2 * rho = (mu_0 / k_eff)^2 * rho

    This is a simplification of the full spectral treatment. For a box of size L,
    the natural choice for a single-mode perturbation of mode number n is
    k_eff = 2*pi*n / L. The simplest choice for a general simulation is
    k_eff = 2*pi / L (fundamental mode).

    Key physics: In the limit k_eff -> infinity (small scales), c_s -> 0 and the
    EOS approaches pressureless dust. At large scales (small k), the effective
    pressure support grows, mimicking the phenomenology of dark matter at galactic
    and larger scales.

    Parameters
    ----------
    mu0 : float
        The tau-framework coupling constant (dimensions: 1/length).
    k_eff : float
        Effective wavenumber for the real-space approximation.
    """

    def __init__(self, mu0, k_eff):
        if mu0 < 0:
            raise ValueError("mu0 must be non-negative.")
        if k_eff <= 0:
            raise ValueError("k_eff must be positive.")
        self.mu0 = mu0
        self.k_eff = k_eff
        self.cs2 = (mu0 / k_eff) ** 2
        self.cs = abs(mu0 / k_eff)

    def pressure(self, rho, eint):
        """
        Compute pressure: P = (mu_0/k_eff)^2 * rho.

        The internal energy argument is accepted for interface compatibility
        but ignored (the EOS is barotropic).

        Parameters
        ----------
        rho : ndarray
            Mass density.
        eint : ndarray
            Specific internal energy (ignored).

        Returns
        -------
        P : ndarray
            Pressure.
        """
        return self.cs2 * rho

    def sound_speed(self, rho, eint):
        """
        Return the effective sound speed c_s = mu_0 / k_eff.

        Parameters
        ----------
        rho : ndarray
            Mass density (ignored).
        eint : ndarray
            Specific internal energy (ignored).

        Returns
        -------
        cs : ndarray
            Sound speed (constant for a given k_eff).
        """
        return np.full_like(rho, self.cs, dtype=float)

    def internal_energy(self, rho, P):
        """
        Compute specific internal energy: e_int = c_s^2 (constant).

        Parameters
        ----------
        rho : ndarray
            Mass density.
        P : ndarray
            Pressure (ignored).

        Returns
        -------
        eint : ndarray
            Specific internal energy.
        """
        return np.full_like(rho, self.cs2, dtype=float)

    @classmethod
    def from_box(cls, mu0, L_box, mode=1):
        """
        Construct a TauEOS from box size and mode number.

        Sets k_eff = 2*pi*mode / L_box.

        Parameters
        ----------
        mu0 : float
            Coupling constant (1/length).
        L_box : float
            Box size.
        mode : int, optional
            Mode number. Default is 1 (fundamental mode).

        Returns
        -------
        eos : TauEOS
            Configured equation of state.
        """
        k_eff = 2.0 * np.pi * mode / L_box
        return cls(mu0, k_eff)

    def __repr__(self):
        return (
            f"TauEOS(mu0={self.mu0}, k_eff={self.k_eff}, "
            f"cs={self.cs:.6e})"
        )
