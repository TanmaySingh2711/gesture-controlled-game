"""Colour themes: the classic look, a high-contrast mode and a colour-blind-safe palette.

Every colour the game and the gesture panel draw with comes from a `Theme`, so an
accessibility mode is a palette swap rather than special cases scattered through the drawing
code. Press ``C`` in game to cycle themes.

Colour is never the *only* carrier of meaning, whichever theme is active:

* direction labels always pair colour with text and an arrow,
* frightened ghosts also change shape (a wavy mouth), and eaten ghosts are eyes only,
* power pellets are larger than ordinary pellets and pulse.

The colour-blind theme uses the Okabe-Ito palette, chosen because its eight colours stay
distinguishable under the common forms of colour-vision deficiency. `tests/test_theme.py`
checks every theme's text against its backgrounds for a WCAG 2.1 contrast ratio of at least
4.5:1, so a palette edit cannot quietly make the interface unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    # maze
    background: Color
    wall_fill: Color
    wall_edge: Color
    door: Color
    pellet: Color
    power: Color
    # characters
    player: Color
    ghosts: Mapping[str, Color]
    frightened: Color
    frightened_flash: Color
    frightened_face: Color
    eye: Color
    pupil: Color
    # text and interface
    text: Color
    dim_text: Color
    accent: Color
    panel_background: Color
    panel_edge: Color
    heading: Color
    ok: Color
    warn: Color
    error: Color
    directions: Mapping[str, Color]
    wall_width: int = 2


def _frozen(mapping: dict[str, Color]) -> Mapping[str, Color]:
    return MappingProxyType(mapping)


CLASSIC: Final = Theme(
    name="classic",
    label="Classic",
    background=(8, 8, 20),
    wall_fill=(16, 22, 66),
    wall_edge=(84, 118, 255),
    door=(230, 180, 200),
    pellet=(255, 224, 176),
    power=(255, 236, 200),
    player=(255, 228, 60),
    ghosts=_frozen(
        {
            "chaser": (255, 82, 82),
            "ambusher": (255, 170, 215),
            "flanker": (105, 220, 235),
            "drifter": (255, 178, 82),
        }
    ),
    frightened=(60, 80, 225),
    frightened_flash=(235, 235, 245),
    frightened_face=(255, 206, 206),
    eye=(235, 235, 245),
    pupil=(25, 25, 60),
    text=(240, 240, 245),
    dim_text=(168, 168, 184),
    accent=(255, 228, 60),
    panel_background=(14, 14, 30),
    panel_edge=(52, 52, 84),
    heading=(150, 170, 255),
    ok=(120, 220, 140),
    warn=(255, 186, 96),
    error=(255, 124, 124),
    directions=_frozen(
        {
            "left": (96, 204, 255),
            "right": (128, 226, 128),
            "up": (255, 196, 96),
            "down": (206, 152, 255),
        }
    ),
)

HIGH_CONTRAST: Final = Theme(
    name="high-contrast",
    label="High contrast",
    background=(0, 0, 0),
    wall_fill=(0, 0, 0),
    wall_edge=(255, 255, 255),
    door=(255, 255, 255),
    pellet=(255, 255, 255),
    power=(255, 255, 0),
    player=(255, 255, 0),
    ghosts=_frozen(
        {
            "chaser": (255, 64, 64),
            "ambusher": (255, 120, 255),
            "flanker": (0, 255, 255),
            "drifter": (255, 170, 0),
        }
    ),
    frightened=(40, 110, 255),
    frightened_flash=(255, 255, 255),
    frightened_face=(255, 255, 255),
    eye=(255, 255, 255),
    pupil=(0, 0, 0),
    text=(255, 255, 255),
    dim_text=(230, 230, 230),
    accent=(255, 255, 0),
    panel_background=(0, 0, 0),
    panel_edge=(255, 255, 255),
    heading=(0, 255, 255),
    ok=(0, 255, 0),
    warn=(255, 255, 0),
    error=(255, 90, 90),
    directions=_frozen(
        {"left": (0, 255, 255), "right": (0, 255, 0), "up": (255, 255, 0), "down": (255, 120, 255)}
    ),
    wall_width=3,
)

# Okabe-Ito: orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple.
COLORBLIND: Final = Theme(
    name="colorblind",
    label="Colour-blind safe",
    background=(8, 8, 20),
    wall_fill=(14, 30, 52),
    wall_edge=(86, 180, 233),
    door=(204, 121, 167),
    pellet=(240, 228, 66),
    power=(240, 228, 66),
    player=(240, 228, 66),
    ghosts=_frozen(
        {
            "chaser": (213, 94, 0),
            "ambusher": (204, 121, 167),
            "flanker": (86, 180, 233),
            "drifter": (230, 159, 0),
        }
    ),
    # Not Okabe-Ito blue: under simulated protanopia that sat only 51 sRGB units from the
    # reddish-purple Ambusher. This violet keeps a gap of at least 127 from every ghost under
    # protanopia, deuteranopia and tritanopia, and 3.7:1 contrast against the maze.
    frightened=(120, 75, 225),
    frightened_flash=(235, 235, 245),
    frightened_face=(240, 228, 66),
    eye=(235, 235, 245),
    pupil=(25, 25, 60),
    text=(240, 240, 245),
    dim_text=(176, 176, 190),
    accent=(240, 228, 66),
    panel_background=(14, 14, 30),
    panel_edge=(60, 60, 92),
    heading=(86, 180, 233),
    ok=(0, 190, 140),
    warn=(230, 159, 0),
    error=(240, 120, 60),
    directions=_frozen(
        {
            "left": (86, 180, 233),
            "right": (0, 190, 140),
            "up": (240, 228, 66),
            "down": (204, 121, 167),
        }
    ),
)

THEMES: Final[dict[str, Theme]] = {
    theme.name: theme for theme in (CLASSIC, HIGH_CONTRAST, COLORBLIND)
}


def next_theme(current: Theme) -> Theme:
    """The theme after `current`, wrapping round - what the ``C`` key cycles through."""
    names = list(THEMES)
    return THEMES[names[(names.index(current.name) + 1) % len(names)]]


def relative_luminance(color: Color) -> float:
    """WCAG 2.1 relative luminance of an sRGB colour."""

    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(first: Color, second: Color) -> float:
    """WCAG 2.1 contrast ratio between two colours, from 1.0 to 21.0."""
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
