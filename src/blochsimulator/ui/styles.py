"""Small cross-platform Qt style helpers shared by desktop workspaces."""

import sys


INTERFACE_STYLE_AUTOMATIC = "automatic"
INTERFACE_STYLE_SYSTEM = "system"
INTERFACE_STYLE_FUSION = "fusion"

INTERFACE_STYLE_CHOICES = (
    (
        "Automatic (system on macOS, Fusion elsewhere)",
        INTERFACE_STYLE_AUTOMATIC,
    ),
    ("System / native", INTERFACE_STYLE_SYSTEM),
    ("Fusion (consistent across platforms)", INTERFACE_STYLE_FUSION),
)


def normalize_interface_style(value) -> str:
    """Return a supported persistent interface-style identifier."""
    style = str(value or "").strip().lower()
    if style in {
        INTERFACE_STYLE_AUTOMATIC,
        INTERFACE_STYLE_SYSTEM,
        INTERFACE_STYLE_FUSION,
    }:
        return style
    return INTERFACE_STYLE_AUTOMATIC


def resolve_interface_style(value, *, platform=None) -> str:
    """Resolve ``automatic`` to the concrete style used on this platform."""
    style = normalize_interface_style(value)
    if style != INTERFACE_STYLE_AUTOMATIC:
        return style
    platform = sys.platform if platform is None else str(platform)
    return INTERFACE_STYLE_SYSTEM if platform == "darwin" else INTERFACE_STYLE_FUSION


# macOS' native QGroupBox style does not consistently honor a font declared
# only on QGroupBox::title.  Bold the group itself (which paints the title),
# then explicitly return contained widgets to normal weight.  Nested group
# boxes receive bold again for their own titles.
BOLD_GROUP_TITLES_STYLE = (
    "QGroupBox { font-weight: bold; }"
    "QGroupBox QWidget { font-weight: normal; }"
    "QGroupBox QGroupBox { font-weight: bold; }"
)


# Optional colour accents ("Colour accents" in Settings > Interface).
RUN_BUTTON_ACCENT_STYLE = (
    "QPushButton:enabled { background-color: #43a047; color: white;"
    " border: 1px solid #2e7d32; border-radius: 5px; padding: 3px 12px;"
    " font-weight: bold; }"
    "QPushButton:enabled:hover { background-color: #4caf50; }"
    "QPushButton:enabled:pressed { background-color: #2e7d32; }"
    "QPushButton:disabled { border: 1px solid palette(mid); border-radius: 5px;"
    " padding: 3px 12px; }"
)

# Hue in degrees at 0 % and 100 % progress: orange through yellow to green.
_PROGRESS_START_HUE = 30
_PROGRESS_END_HUE = 125
# Restyling every value change is wasteful; the hue moves in 5 % steps.
_PROGRESS_STEPS = 20


def progress_accent_style(fraction) -> str:
    """Progress-bar stylesheet whose chunk colour follows ``fraction``.

    ``None`` (an indeterminate bar) gives a neutral blue.
    """
    if fraction is None:
        hue = 210
    else:
        fraction = min(1.0, max(0.0, float(fraction)))
        hue = round(
            _PROGRESS_START_HUE + (_PROGRESS_END_HUE - _PROGRESS_START_HUE) * fraction
        )
    return (
        "QProgressBar { border: 1px solid palette(mid); border-radius: 4px;"
        " background-color: palette(base); color: #111111; text-align: center;"
        " min-height: 16px; }"
        f"QProgressBar::chunk {{ background-color: hsl({hue}, 60%, 58%);"
        " border-radius: 3px; }"
    )


class ProgressAccent:
    """Tint a QProgressBar from orange to green as it fills.

    Disabled, the bar keeps the platform's native look.
    """

    def __init__(self, bar, enabled=True):
        self.bar = bar
        self.enabled = bool(enabled)
        self._step = object()
        bar.valueChanged.connect(self.refresh)
        self.refresh()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self._step = object()
        self.refresh()

    def refresh(self, *_):
        if not self.enabled:
            if self._step is not None:
                self.bar.setStyleSheet("")
                self._step = None
            return
        span = self.bar.maximum() - self.bar.minimum()
        step = (
            None
            if span <= 0
            else round(_PROGRESS_STEPS * (self.bar.value() - self.bar.minimum()) / span)
        )
        if step == self._step:
            return
        self._step = step
        self.bar.setStyleSheet(
            progress_accent_style(None if step is None else step / _PROGRESS_STEPS)
        )
