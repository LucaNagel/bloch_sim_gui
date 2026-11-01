"""Application appearance: light and dark themes on top of Qt's Fusion style.

A theme is a table of colour tokens. :func:`apply_appearance` turns the
selected appearance into the Qt style, an application palette and one
application-wide style sheet. Widgets mark their role with dynamic
properties (``primary`` buttons, ``role`` labels, ``segmented`` button bars)
instead of carrying their own colours, so a theme change restyles them all.
Labels whose text colour carries meaning use :func:`set_tone`.
"""

import atexit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PyQt5.QtGui import QColor, QPalette

from .styles import (
    INTERFACE_STYLE_DARK,
    INTERFACE_STYLE_LIGHT,
    INTERFACE_STYLE_SYSTEM,
    resolve_interface_style,
)

LIGHT = {
    "window": "#f3f5f8",
    "card": "#ffffff",
    "field": "#ffffff",
    "alt": "#f7f9fb",
    "border": "#dbe1e8",
    "border_strong": "#c3ccd6",
    "text": "#16202b",
    "muted": "#5b6877",
    "accent": "#0b7a8f",
    "accent_hover": "#0d8aa1",
    "accent_soft": "#e2f2f5",
    "on_accent": "#ffffff",
    "hover": "#eef2f6",
    "disabled": "#a3adb8",
    "ok": "#2f7d3a",
    "warning": "#b45309",
    "error": "#b91c1c",
    "info": "#2563eb",
}

DARK = {
    "window": "#0f151b",
    "card": "#161e26",
    "field": "#1c2630",
    "alt": "#1a232c",
    "border": "#2a3541",
    "border_strong": "#3a4856",
    "text": "#e3e9ef",
    "muted": "#98a6b4",
    "accent": "#4cc6dc",
    "accent_hover": "#6fd3e5",
    "accent_soft": "#14303a",
    "on_accent": "#06151a",
    "hover": "#202b36",
    "disabled": "#5d6b78",
    "ok": "#6cc77a",
    "warning": "#f0a35c",
    "error": "#f87171",
    "info": "#7cb4ff",
}

# The native style keeps the platform's own colours; only the accents of
# the group bar and of the primary action are fixed.
NATIVE_ACCENT = "#0a64d8"

TONES = ("muted", "ok", "warning", "error", "info", "accent")

# Accent presets offered under Settings -> Interface. The empty default keeps
# the theme's own teal; any other entry replaces the accent of both themes.
ACCENT_DEFAULT = ""
ACCENT_CHOICES = (
    ("Teal (default)", ACCENT_DEFAULT),
    ("Blue", "#1f5fbf"),
    ("Indigo", "#4f46e5"),
    ("Purple", "#7c3aed"),
    ("Magenta", "#b5267a"),
    ("Red", "#c0392b"),
    ("Orange", "#c2570c"),
    ("Green", "#2e7d32"),
    ("Graphite", "#3f4a56"),
)

_TONE_PROPERTY = "blochSimulatorTone"
_TONE_EXTRA_PROPERTY = "blochSimulatorToneExtra"

_state = {
    "tokens": LIGHT,
    "appearance": INTERFACE_STYLE_LIGHT,
    "dark": False,
    "accent": ACCENT_DEFAULT,
}
_system_palette = {}
_icon_directory = {}


def current_tokens() -> dict:
    """Colour tokens of the applied appearance (light before any is applied)."""
    return _state["tokens"]


def current_appearance() -> str:
    """The concrete appearance applied last: ``light``, ``dark`` or ``system``."""
    return _state["appearance"]


def color(name: str) -> str:
    """One colour of the applied appearance, e.g. ``color("warning")``."""
    return _state["tokens"][name]


def is_dark() -> bool:
    return bool(_state["dark"])


def current_accent() -> str:
    """The custom accent applied last, or ``""`` for the theme's own."""
    return _state["accent"]


def normalize_accent(value) -> str:
    """A ``#rrggbb`` accent colour, or ``""`` for the theme default."""
    text = str(value or "").strip()
    if not text:
        return ACCENT_DEFAULT
    colour = QColor(text)
    if not colour.isValid():
        return ACCENT_DEFAULT
    return colour.name()


def _mix(first: str, second: str, amount: float) -> str:
    """``first`` moved ``amount`` (0..1) of the way towards ``second``."""
    a, b = QColor(first), QColor(second)
    channels = (
        round(a.red() + (b.red() - a.red()) * amount),
        round(a.green() + (b.green() - a.green()) * amount),
        round(a.blue() + (b.blue() - a.blue()) * amount),
    )
    return QColor(*channels).name()


def _relative_luminance(value: str) -> float:
    colour = QColor(value)

    def linear(channel):
        channel /= 255.0
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linear(colour.red())
        + 0.7152 * linear(colour.green())
        + 0.0722 * linear(colour.blue())
    )


def with_accent(tokens, accent) -> dict:
    """``tokens`` with the accent family derived from one chosen colour.

    The dark theme lifts the colour towards white so that it keeps its
    contrast on the dark cards; the soft tint mixes it into the card colour.
    """
    accent = normalize_accent(accent)
    if not accent:
        return tokens
    dark = _relative_luminance(tokens["card"]) < 0.2
    base = _mix(accent, "#ffffff", 0.35) if dark else accent
    result = dict(tokens)
    result["accent"] = base
    result["accent_hover"] = _mix(base, "#ffffff", 0.15)
    result["accent_soft"] = _mix(tokens["card"], base, 0.22 if dark else 0.12)
    result["on_accent"] = (
        "#ffffff" if _relative_luminance(base) < 0.35 else tokens["window"]
    )
    return result


def tone_style(tone: str, extra: str = "") -> str:
    """Style sheet giving a label the text colour of ``tone``."""
    return f"color: {color(tone)};{(' ' + extra) if extra else ''}"


def set_tone(widget, tone, extra=""):
    """Colour ``widget``'s text by meaning; ``None`` returns to the default.

    The tone is remembered, so a later theme change recolours the widget.
    ``extra`` holds further declarations such as ``font-weight: bold;``.
    """
    widget.setProperty(_TONE_PROPERTY, tone or "")
    widget.setProperty(_TONE_EXTRA_PROPERTY, extra or "")
    if tone:
        widget.setStyleSheet(tone_style(tone, extra))
    else:
        widget.setStyleSheet(extra or "")


def tone_of(widget):
    """The tone last given to ``widget`` by :func:`set_tone`, or ``None``."""
    return widget.property(_TONE_PROPERTY) or None


def _refresh_tones(app):
    for widget in app.allWidgets():
        tone = widget.property(_TONE_PROPERTY)
        if tone:
            widget.setStyleSheet(
                tone_style(str(tone), str(widget.property(_TONE_EXTRA_PROPERTY) or ""))
            )


def remember_system_palette(app):
    """Keep the palette Qt chose before a theme replaces it."""
    if "palette" not in _system_palette:
        _system_palette["palette"] = QPalette(app.palette())
    return _system_palette["palette"]


def system_prefers_dark(palette=None, *, platform=None) -> bool:
    """Whether the operating system currently uses a dark appearance."""
    platform = sys.platform if platform is None else str(platform)
    if platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.stdout.strip().lower() == "dark"
        except (OSError, subprocess.SubprocessError):
            pass
    elif platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return int(value) == 0
        except (ImportError, OSError):
            pass
    if palette is None:
        palette = _system_palette.get("palette")
    if palette is None:
        return False
    return palette.color(QPalette.Window).lightness() < 128


def palette_for(tokens) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.Window: "window",
        QPalette.WindowText: "text",
        QPalette.Base: "field",
        QPalette.AlternateBase: "alt",
        QPalette.Text: "text",
        QPalette.Button: "card",
        QPalette.ButtonText: "text",
        QPalette.BrightText: "on_accent",
        QPalette.Highlight: "accent",
        QPalette.HighlightedText: "on_accent",
        QPalette.ToolTipBase: "card",
        QPalette.ToolTipText: "text",
        QPalette.Link: "accent",
        QPalette.LinkVisited: "accent",
        QPalette.Light: "card",
        QPalette.Midlight: "border",
        QPalette.Mid: "border_strong",
        QPalette.Dark: "muted",
        QPalette.PlaceholderText: "muted",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(tokens[token]))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(tokens["disabled"]))
    palette.setColor(QPalette.Disabled, QPalette.Base, QColor(tokens["alt"]))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(tokens["alt"]))
    return palette


_CHEVRONS = {
    "down": "M2.5 4.5 6 8l3.5-3.5",
    "up": "M2.5 7.5 6 4l3.5 3.5",
    "left": "M7.5 2.5 4 6l3.5 3.5",
    "right": "M4.5 2.5 8 6l-3.5 3.5",
}


def _icons(tokens) -> Path:
    """Directory of chevron icons drawn in the theme's muted text colour."""
    stroke = tokens["muted"]
    directory = _icon_directory.get(stroke)
    if directory is not None and directory.is_dir():
        return directory
    root = _icon_directory.get("root")
    if root is None or not root.is_dir():
        root = Path(tempfile.mkdtemp(prefix="blochsimulator-theme-"))
        _icon_directory["root"] = root
        atexit.register(shutil.rmtree, root, True)
    directory = root / stroke.lstrip("#")
    directory.mkdir(exist_ok=True)
    for name, path in _CHEVRONS.items():
        (directory / f"{name}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"'
            f' viewBox="0 0 12 12"><path d="{path}" fill="none" stroke="{stroke}"'
            ' stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>",
            encoding="utf-8",
        )
    _icon_directory[stroke] = directory
    return directory


def _segmented_rules(accent, on_accent, border, card, muted, hover, text) -> str:
    """Group-bar rules shared by every appearance."""
    bar = 'QWidget[segmented="true"] > QToolButton'
    large = 'QWidget[segmented="true"][size="large"] > QToolButton'
    return f"""
{bar} {{ border: 1px solid {border}; background: {card}; color: {muted};
    padding: 4px 16px; margin-left: -1px; border-radius: 0; font-weight: 600;
    min-height: 0; }}
{large} {{ padding: 6px 22px; }}
{bar}[segment="first"] {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
{bar}[segment="last"] {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
{bar}:hover:!checked {{ color: {text}; background: {hover}; }}
{bar}:checked {{ background: {accent}; color: {on_accent}; border-color: {accent}; }}
{bar}:disabled {{ color: {border}; }}
"""


def stylesheet_for(tokens) -> str:
    """Application-wide style sheet of a Fusion-based theme."""
    c = tokens
    icons = _icons(tokens).as_posix()
    return f"""
QToolTip {{ color: {c['text']}; background: {c['card']}; border: 1px solid {c['border']};
    padding: 4px; }}
QGroupBox {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px;
    margin-top: 16px; padding: 12px 10px 8px 10px; }}
QGroupBox[title=""] {{ margin-top: 0px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px;
    padding: 0 4px; color: {c['text']}; }}
QGroupBox QGroupBox {{ background: {c['alt']}; }}
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 8px;
    background: {c['card']}; top: -1px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {c['muted']}; border: none;
    border-bottom: 2px solid transparent; padding: 7px 12px 6px 12px; margin-right: 2px; }}
QTabBar::tab:selected {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}
QTabBar::tab:hover:!selected {{ color: {c['text']}; background: {c['hover']};
    border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:disabled {{ color: {c['disabled']}; }}
QTabBar QToolButton {{ border: none; background: {c['card']}; padding: 0; min-height: 0;
    border-radius: 0; }}
QTabBar QToolButton::left-arrow {{ image: url({icons}/left.svg); width: 12px; height: 12px; }}
QTabBar QToolButton::right-arrow {{ image: url({icons}/right.svg); width: 12px; height: 12px; }}
QPushButton, QToolButton {{ background: {c['card']}; color: {c['text']};
    border: 1px solid {c['border_strong']}; border-radius: 6px; padding: 5px 12px;
    min-height: 18px; }}
QPushButton:hover, QToolButton:hover {{ background: {c['hover']}; }}
QPushButton:pressed, QPushButton:checked, QToolButton:pressed, QToolButton:checked {{
    background: {c['accent_soft']}; border-color: {c['accent']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c['disabled']};
    border-color: {c['border']}; background: {c['alt']}; }}
QPushButton[primary="true"] {{ background: {c['accent']}; color: {c['on_accent']};
    border-color: {c['accent']}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {c['accent_hover']}; }}
QPushButton[primary="true"]:disabled {{ background: {c['alt']}; color: {c['disabled']};
    border-color: {c['border']}; }}
QToolButton[autoRaise="true"], QToolButton[flat="true"] {{ border: 1px solid transparent;
    background: transparent; }}
QToolButton[autoRaise="true"]:hover, QToolButton[flat="true"]:hover {{
    background: {c['hover']}; }}
QToolButton[flat="true"] {{ color: {c['muted']}; padding: 4px 8px; }}
QToolButton[flat="true"]:hover {{ color: {c['text']}; }}
QToolButton[popupMode="2"] {{ padding-right: 22px; }}
QToolButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
QPushButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
QComboBox, QAbstractSpinBox, QLineEdit {{ background: {c['field']}; color: {c['text']};
    border: 1px solid {c['border_strong']}; border-radius: 6px; padding: 4px 6px;
    min-height: 18px; selection-background-color: {c['accent']};
    selection-color: {c['on_accent']}; }}
QComboBox:focus, QAbstractSpinBox:focus, QLineEdit:focus {{ border: 1px solid {c['accent']}; }}
QComboBox:disabled, QAbstractSpinBox:disabled, QLineEdit:disabled {{ color: {c['disabled']};
    background: {c['alt']}; border-color: {c['border']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({icons}/down.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {c['card']}; border: 1px solid {c['border']};
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ border: none; width: 18px;
    background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-origin: border;
    subcontrol-position: bottom right; }}
QAbstractSpinBox::up-arrow {{ image: url({icons}/up.svg); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({icons}/down.svg); width: 10px; height: 10px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: none; }}
QTableView, QTreeView, QListView, QTextEdit, QPlainTextEdit, QTextBrowser {{
    background: {c['field']}; border: 1px solid {c['border']}; border-radius: 6px;
    gridline-color: {c['border']}; alternate-background-color: {c['alt']};
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QHeaderView::section {{ background: {c['alt']}; color: {c['muted']}; border: none;
    border-bottom: 1px solid {c['border']}; padding: 4px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {c['alt']}; border: none; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {c['border_strong']}; border-radius: 3px;
    min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {c['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QProgressBar {{ border: 1px solid {c['border_strong']}; border-radius: 6px;
    background: {c['field']}; color: {c['text']}; text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 5px; }}
QMenu {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['border']};
    padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QMenu::item:disabled {{ color: {c['disabled']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 6px; }}
QStatusBar {{ color: {c['muted']}; }}
QLabel[role="caption"] {{ color: {c['muted']}; }}
QLabel[role="heading"] {{ color: {c['text']}; }}
QFrame[role="rule"] {{ background: {c['border']}; max-height: 1px; min-height: 1px;
    border: none; }}
""" + _segmented_rules(
        c["accent"],
        c["on_accent"],
        c["border_strong"],
        c["card"],
        c["muted"],
        c["hover"],
        c["text"],
    )


def native_stylesheet(accent=None) -> str:
    """Role rules for the platform style; everything else stays native."""
    accent = normalize_accent(accent) or NATIVE_ACCENT
    return (
        'QFrame[role="rule"] { background: palette(mid); max-height: 1px;'
        " min-height: 1px; border: none; }"
        'QToolButton[flat="true"] { border: none; padding: 3px 6px; }'
        'QPushButton[dimWhenDisabled="true"]:disabled { color: rgba(70, 70, 70, 145);'
        " background-color: rgba(210, 210, 210, 150); }"
        + _segmented_rules(
            accent,
            "#ffffff",
            "palette(mid)",
            "palette(button)",
            "palette(button-text)",
            "palette(light)",
            "palette(button-text)",
        )
    )


def apply_appearance(
    app, appearance, *, system_style_name=None, platform=None, accent=None
):
    """Apply an appearance identifier from Settings to ``app``.

    ``accent`` is a ``#rrggbb`` colour replacing the theme's teal accent;
    ``None`` or ``""`` keeps it. Returns the concrete appearance: ``light``,
    ``dark`` or ``system``.
    """
    if app is None:
        return None
    system_palette = remember_system_palette(app)
    concrete = resolve_interface_style(
        appearance,
        system_dark=lambda: system_prefers_dark(system_palette, platform=platform),
    )
    if concrete == INTERFACE_STYLE_SYSTEM:
        if system_style_name:
            app.setStyle(str(system_style_name))
        app.setPalette(system_palette)
        dark = system_prefers_dark(system_palette, platform=platform)
        _state["tokens"] = with_accent(DARK if dark else LIGHT, accent)
        _state["dark"] = dark
        app.setStyleSheet(native_stylesheet(accent))
    else:
        dark = concrete == INTERFACE_STYLE_DARK
        tokens = with_accent(DARK if dark else LIGHT, accent)
        _state["tokens"] = tokens
        _state["dark"] = dark
        app.setStyle("Fusion")
        app.setPalette(palette_for(tokens))
        app.setStyleSheet(stylesheet_for(tokens))
    _state["accent"] = normalize_accent(accent)
    _state["appearance"] = concrete
    _refresh_tones(app)
    return concrete


__all__ = [
    "ACCENT_CHOICES",
    "ACCENT_DEFAULT",
    "DARK",
    "INTERFACE_STYLE_DARK",
    "INTERFACE_STYLE_LIGHT",
    "LIGHT",
    "TONES",
    "apply_appearance",
    "color",
    "current_accent",
    "current_appearance",
    "current_tokens",
    "is_dark",
    "native_stylesheet",
    "normalize_accent",
    "palette_for",
    "remember_system_palette",
    "set_tone",
    "stylesheet_for",
    "system_prefers_dark",
    "tone_style",
    "with_accent",
]
