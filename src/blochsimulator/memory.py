"""RAM budgeting helpers for simulation allocations."""

from dataclasses import dataclass
from typing import Optional, Tuple


GIB = 1024**3
DEFAULT_MINIMUM_RESERVE_BYTES = 2 * GIB
DEFAULT_RESERVE_FRACTION = 0.10
EMERGENCY_RESERVE_BYTES = 512 * 1024**2
FALLBACK_MEMORY_BUDGET_BYTES = 512 * 1024**2
MEMORY_ERROR_PREFIX = "Memory limit exceeded:"


class SimulationMemoryError(MemoryError):
    """Raised before a simulation would exceed its safe RAM budget."""


@dataclass(frozen=True)
class MemoryPolicy:
    """User-selectable policy for simulation RAM usage."""

    mode: str = "automatic"
    reserve_bytes: int = DEFAULT_MINIMUM_RESERVE_BYTES
    limit_bytes: int = 8 * GIB

    def __post_init__(self):
        if self.mode not in {"automatic", "custom_reserve", "fixed_limit"}:
            raise ValueError(f"Unknown memory policy: {self.mode}")
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must not be negative")
        if self.limit_bytes <= 0:
            raise ValueError("limit_bytes must be greater than zero")


@dataclass(frozen=True)
class MemoryBudget:
    """Resolved memory budget and the system values it was derived from."""

    limit_bytes: int
    available_bytes: Optional[int]
    total_bytes: Optional[int]
    reserve_bytes: Optional[int]
    mode: str


_default_policy = MemoryPolicy()


def set_default_memory_policy(policy: MemoryPolicy) -> None:
    """Set the process-wide policy used by GUI and sequence compilation."""
    global _default_policy
    _default_policy = policy


def get_default_memory_policy() -> MemoryPolicy:
    """Return the current process-wide simulation memory policy."""
    return _default_policy


def system_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    """Return ``(available, total)`` physical memory when detectable."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.available), int(memory.total)
    except (ImportError, AttributeError, OSError):
        return None, None


def resolve_memory_budget(
    *,
    policy: Optional[MemoryPolicy] = None,
    explicit_limit_bytes: Optional[int] = None,
    available_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
) -> MemoryBudget:
    """Resolve the safe allocation budget for one simulation.

    Automatic mode keeps the larger of 2 GiB and 10% of physical RAM free.
    Custom-reserve mode uses the user's reserve. Fixed-limit mode still keeps
    a small emergency reserve so a stale setting cannot consume all currently
    available memory.
    """
    if explicit_limit_bytes is not None:
        if explicit_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be greater than zero")
        policy = MemoryPolicy(mode="fixed_limit", limit_bytes=explicit_limit_bytes)
    elif policy is None:
        policy = get_default_memory_policy()

    if available_bytes is None and total_bytes is None:
        available_bytes, total_bytes = system_memory_bytes()

    if available_bytes is None:
        return MemoryBudget(
            min(FALLBACK_MEMORY_BUDGET_BYTES, policy.limit_bytes),
            None,
            total_bytes,
            None,
            policy.mode,
        )

    available_bytes = max(0, int(available_bytes))
    if total_bytes is not None:
        total_bytes = max(0, int(total_bytes))

    if policy.mode == "automatic":
        proportional_reserve = (
            int(total_bytes * DEFAULT_RESERVE_FRACTION)
            if total_bytes is not None
            else 0
        )
        reserve_bytes = max(DEFAULT_MINIMUM_RESERVE_BYTES, proportional_reserve)
        limit_bytes = max(0, available_bytes - reserve_bytes)
    elif policy.mode == "custom_reserve":
        reserve_bytes = policy.reserve_bytes
        limit_bytes = max(0, available_bytes - reserve_bytes)
    else:
        reserve_bytes = EMERGENCY_RESERVE_BYTES
        safely_available = max(0, available_bytes - reserve_bytes)
        limit_bytes = min(policy.limit_bytes, safely_available)

    return MemoryBudget(
        limit_bytes,
        available_bytes,
        total_bytes,
        reserve_bytes,
        policy.mode,
    )


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""
    value = int(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def enforce_memory_budget(
    estimated_bytes: int,
    budget: MemoryBudget,
    *,
    description: str,
    suggestions: str,
) -> None:
    """Reject an allocation whose estimated peak exceeds the safe budget."""
    if estimated_bytes <= budget.limit_bytes:
        return

    if budget.available_bytes is None:
        basis = "the conservative fallback limit"
    elif budget.mode == "fixed_limit":
        basis = "the configured limit and the 512 MiB emergency reserve"
    else:
        basis = (
            f"{format_bytes(budget.available_bytes)} currently available RAM "
            f"minus a {format_bytes(budget.reserve_bytes or 0)} reserve"
        )

    raise SimulationMemoryError(
        f"{MEMORY_ERROR_PREFIX} The selected simulation needs approximately "
        f"{format_bytes(estimated_bytes)} RAM, but the current safe budget is "
        f"{format_bytes(budget.limit_bytes)}. Budget basis: {basis}. "
        f"Selection: {description}. To continue, {suggestions.lower()}."
    )


def enforce_sequence_memory(
    point_count: int, *, estimated_bytes_per_point: int = 80
) -> None:
    """Reject oversized sequence arrays before NumPy allocates them.

    A compiled sequence normally owns complex B1, three float gradients and a
    float time axis (48 bytes per point). The higher estimate accounts for
    carrier application, resampling and concatenation temporaries.
    """
    if point_count < 0:
        raise ValueError("point_count must not be negative")
    estimated_bytes = int(point_count) * int(estimated_bytes_per_point)
    enforce_memory_budget(
        estimated_bytes,
        resolve_memory_budget(),
        description=f"The compiled sequence would contain {point_count:,} time points",
        suggestions=(
            "Increase the time step, shorten TR or the additional tail, or reduce "
            "the number of sequence repetitions"
        ),
    )
