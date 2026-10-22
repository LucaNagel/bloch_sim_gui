"""Interactive Bloch evolution for the Free Mode live mode.

Pulses are ideal hard pulses: instantaneous rotations, so no time passes
and neither gradients nor off-resonance act during them. Between pulses
the spins precess under the switched-on gradient and their off-resonance,
and optionally relax towards Mz = 1.

Signs follow the simulation kernel: a pulse of phase 0° tips +z towards
+y, a pulse of phase 90° towards +x, and a positive local frequency turns
the transverse magnetization clockwise seen from +z.
"""

import numpy as np


def rotation_matrix(rotation) -> np.ndarray:
    """Right-handed rotation by ``|rotation|`` rad about ``rotation``."""
    rotation = np.asarray(rotation, dtype=float)
    angle = float(np.linalg.norm(rotation))
    if angle == 0.0:
        return np.eye(3)
    kx, ky, kz = rotation / angle
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def pulse_rotation(flip_deg: float, phase_deg: float) -> np.ndarray:
    """Rotation vector of a hard pulse, as the kernel turns it."""
    flip = np.radians(flip_deg)
    phase = np.radians(phase_deg)
    return flip * np.array([-np.cos(phase), np.sin(phase), 0.0])


class LiveSpinSystem:
    """Spins at fixed positions and off-resonances, driven step by step.

    Parameters
    ----------
    positions_m : array-like
        Position of every spin group, (N, 3) in metres.
    offsets_hz : array-like
        Off-resonance of the isochromats at each position, (F,) in Hz.
    gamma_hz_per_t : float
        Gyromagnetic ratio of the nucleus.
    t1_s, t2_s : float
        Relaxation times; ``np.inf`` disables a relaxation.
    """

    def __init__(
        self, positions_m, offsets_hz, gamma_hz_per_t, t1_s=np.inf, t2_s=np.inf
    ):
        self.positions_m = np.asarray(positions_m, dtype=float).reshape(-1, 3)
        self.offsets_hz = np.atleast_1d(np.asarray(offsets_hz, dtype=float))
        self.gamma_hz_per_t = float(gamma_hz_per_t)
        self.t1_s = float(t1_s)
        self.t2_s = float(t2_s)
        self.reset()

    def reset(self):
        """Return every spin to equilibrium (Mz = 1) at time zero."""
        shape = (len(self.positions_m), len(self.offsets_hz), 3)
        self.magnetization = np.zeros(shape)
        self.magnetization[..., 2] = 1.0
        self.time_s = 0.0

    def apply_pulse(self, flip_deg: float, phase_deg: float = 0.0):
        """Rotate all spins by an ideal hard pulse."""
        rotation = rotation_matrix(pulse_rotation(flip_deg, phase_deg))
        self.magnetization = self.magnetization @ rotation.T

    def local_frequencies_hz(self, gradient_t_per_m=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Precession frequency of every isochromat, (N, F) in Hz."""
        gradient = np.asarray(gradient_t_per_m, dtype=float)
        position_hz = self.gamma_hz_per_t * (self.positions_m @ gradient)
        return position_hz[:, None] + self.offsets_hz[None, :]

    def evolve(self, dt_s: float, gradient_t_per_m=(0.0, 0.0, 0.0), relax=True):
        """Precess (and relax) for ``dt_s`` under a constant gradient."""
        dt_s = float(dt_s)
        if dt_s <= 0.0:
            return
        m = self.magnetization
        phase = 2.0 * np.pi * self.local_frequencies_hz(gradient_t_per_m) * dt_s
        transverse = (m[..., 0] + 1j * m[..., 1]) * np.exp(-1j * phase)
        mz = m[..., 2]
        if relax:
            transverse = transverse * np.exp(-dt_s / self.t2_s)
            mz = 1.0 + (mz - 1.0) * np.exp(-dt_s / self.t1_s)
        m[..., 0] = transverse.real
        m[..., 1] = transverse.imag
        m[..., 2] = mz
        self.time_s += dt_s
