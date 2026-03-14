"""
Anatropic: 1D/3D Euler + self-gravity solver with tau-framework EOS support.

Modules
-------
euler        : 1D compressible Euler equations with HLLE Riemann solver
euler3d      : 3D compressible Euler equations with batched HLLE + dim splitting
gravity      : FFT-based 1D Poisson solver for self-gravity
gravity3d    : FFT-based 3D Poisson solver for self-gravity
eos          : Equation of state implementations (ideal gas, isothermal, tau-EOS)
simulation   : High-level 1D simulation driver
simulation3d : High-level 3D simulation driver
"""

from . import euler
from . import euler3d
from . import gravity
from . import gravity3d
from . import eos
from . import simulation
from . import simulation3d

__version__ = "0.1.0"
__all__ = [
    "euler", "euler3d", "gravity", "gravity3d",
    "eos", "simulation", "simulation3d",
]
