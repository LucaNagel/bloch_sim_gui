"""Strict-build bindings for the ordinary C Bloch solver.

The implementation is intentionally shared with ``bloch_wrapper.pyx`` while
the extension name and the included C translation unit remain distinct.  This
prevents setuptools from reusing an object compiled with the legacy Fast Math
flags.
"""

include "bloch_wrapper.pyx"

NUMERICAL_CONTRACT = "strict-float64"
