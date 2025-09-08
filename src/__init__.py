"""
src package for PINN Heat + Kinetics project.

Modules:
- fd_reference   : Finite-difference reference solver
- surrogate      : Differentiable surrogate for f_eq(T)
- pinn_heat_kinetics : Main PINN training script
"""

from . import fd_reference
from . import surrogate
from . import pinn_heat_kinetics

__all__ = ["fd_reference", "surrogate", "pinn_heat_kinetics"]
