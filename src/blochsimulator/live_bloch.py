"""Interactive Bloch evolution for the Free Mode live mode.

Pulses are rectangular (hard) pulses of finite length: during a pulse the
spins turn about the effective field of B1, their off-resonance and the
switched-on gradient, so time passes and the signal evolves. Between pulses
the spins precess under the gradient and their off-resonance. Optionally
they relax towards Mz = 1. :meth:`LiveSpinSystem.apply_pulse` keeps the
instantaneous ideal rotation.

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

    def drive(
        self,
        dt_s: float,
        rabi_hz: float,
        phase_deg: float = 0.0,
        gradient_t_per_m=(0.0, 0.0, 0.0),
        relax=True,
    ):
        """Turn the spins for ``dt_s`` under a constant RF field.

        ``rabi_hz`` is the nutation rate gamma*B1 of the pulse; each
        isochromat turns about its effective field, which also contains its
        off-resonance and the gradient.
        """
        dt_s = float(dt_s)
        if dt_s <= 0.0:
            return
        omega_1 = 2.0 * np.pi * float(rabi_hz)
        phase = np.radians(phase_deg)
        local = self.local_frequencies_hz(gradient_t_per_m)
        rotation = np.empty(local.shape + (3,))
        rotation[..., 0] = -omega_1 * np.cos(phase) * dt_s
        rotation[..., 1] = omega_1 * np.sin(phase) * dt_s
        # A positive local frequency turns clockwise seen from +z.
        rotation[..., 2] = -2.0 * np.pi * local * dt_s
        angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
        axis = np.divide(
            rotation, angle, out=np.zeros_like(rotation), where=angle > 0.0
        )
        m = self.magnetization
        cos, sin = np.cos(angle), np.sin(angle)
        along = np.sum(axis * m, axis=-1, keepdims=True)
        self.magnetization = (
            m * cos + np.cross(axis, m) * sin + axis * along * (1.0 - cos)
        )
        if relax:
            self._relax(dt_s)
        self.time_s += dt_s

    def _relax(self, dt_s: float):
        m = self.magnetization
        e2 = np.exp(-dt_s / self.t2_s)
        m[..., 0] *= e2
        m[..., 1] *= e2
        m[..., 2] = 1.0 + (m[..., 2] - 1.0) * np.exp(-dt_s / self.t1_s)

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
        m[..., 0] = transverse.real
        m[..., 1] = transverse.imag
        if relax:
            self._relax(dt_s)
        self.time_s += dt_s
