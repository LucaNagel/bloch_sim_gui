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
