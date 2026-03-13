"""
Anatropic: 1D Euler + self-gravity solver with tau-framework EOS support.

Modules
-------
euler      : 1D compressible Euler equations with HLLE Riemann solver
gravity    : FFT-based 1D Poisson solver for self-gravity
eos        : Equation of state implementations (ideal gas, isothermal, tau-EOS)
simulation : High-level simulation driver
"""

from . import euler
from . import gravity
from . import eos
from . import simulation

__version__ = "0.1.0"
__all__ = ["euler", "gravity", "eos", "simulation"]
