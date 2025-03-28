# used for setup.py
name = "ceviche"

__version__ = "0.1.3"

from .fdfd import fdfd_ez, fdfd_hz, fdfd_mf_ez, my_fdfd_ez

__all__ = [
    "constants",
    "derivatives",
    "fdfd_ez",
    "fdfd_hz",
    "fdfd_mf_ez",
    "fdtd",
    "my_fdfd_ez",
    "jacobian",
    "modes",
    "optimizer",
    "primitives",
    "solvers",
    "sources",
    "utils",
    "viz",
]
