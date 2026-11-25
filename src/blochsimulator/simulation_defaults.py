"""Default numerical settings shared by the Python API and the GUI."""

#: Default time step in seconds while RF is active. Every sequence entry
#: point (static, dynamic, point-source PSF and spin probes) and the GUI use
#: it unless a caller passes another value. ``None`` must be passed
#: explicitly to compile on the native sequence raster only; long RF samples
#: then become single intervals and carry a large splitting error.
DEFAULT_SIMULATION_TIMESTEP_S = 20e-6

__all__ = ["DEFAULT_SIMULATION_TIMESTEP_S"]
