"""Every colour theme must stay readable, and must not hide game state behind colour alone."""

from __future__ import annotations

import itertools

import pytest

from game.theme import CLASSIC, THEMES, Color, Theme, contrast_ratio, next_theme

# WCAG 2.1 AA for normal-size text.
MIN_TEXT_CONTRAST = 4.5

# Machado et al. (2009) full-severity colour-vision-deficiency matrices, in linear RGB.
CVD_MATRICES = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

# Minimum sRGB distance between a dangerous ghost and a frightened one after simulation.
# Roughly the gap between two clearly different swatches; below ~40 colours start to read as
# "the same, slightly off".
MIN_STATE_DISTANCE = 60.0


def _to_linear(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_srgb(value: float) -> float:
    value = min(1.0, max(0.0, value))
    c = value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
    return c * 255.0


def simulate(color: Color, deficiency: str) -> tuple[float, float, float]:
    linear = [_to_linear(v) for v in color]
    matrix = CVD_MATRICES[deficiency]
    mixed = [sum(row[i] * linear[i] for i in range(3)) for row in matrix]
    return (_to_srgb(mixed[0]), _to_srgb(mixed[1]), _to_srgb(mixed[2]))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)


themes = pytest.mark.parametrize("theme", list(THEMES.values()), ids=list(THEMES))


@themes
def test_body_text_meets_wcag_aa(theme: Theme) -> None:
    for fg_name, bg_name in itertools.product(
        ("text", "dim_text", "accent"), ("background", "panel_background")
    ):
        ratio = contrast_ratio(getattr(theme, fg_name), getattr(theme, bg_name))
        assert ratio >= MIN_TEXT_CONTRAST, f"{fg_name} on {bg_name} is only {ratio:.2f}:1"


@themes
def test_status_and_direction_colours_are_readable_on_the_panel(theme: Theme) -> None:
    for name in ("heading", "ok", "warn", "error"):
        ratio = contrast_ratio(getattr(theme, name), theme.panel_background)
        assert ratio >= MIN_TEXT_CONTRAST, f"{name} is only {ratio:.2f}:1"
    for direction, colour in theme.directions.items():
        ratio = contrast_ratio(colour, theme.panel_background)
        assert ratio >= MIN_TEXT_CONTRAST, f"{direction} is only {ratio:.2f}:1"


@themes
def test_theme_covers_every_ghost_and_direction(theme: Theme) -> None:
    assert set(theme.ghosts) == {"chaser", "ambusher", "flanker", "drifter"}
    assert set(theme.directions) == {"left", "right", "up", "down"}
    assert len({*theme.ghosts.values()}) == 4, "two ghosts share a colour"
    assert theme.frightened not in theme.ghosts.values()


@pytest.mark.parametrize("deficiency", list(CVD_MATRICES))
def test_colourblind_theme_keeps_frightened_ghosts_distinct(deficiency: str) -> None:
    theme = THEMES["colorblind"]
    frightened = simulate(theme.frightened, deficiency)
    for role, colour in theme.ghosts.items():
        gap = distance(simulate(colour, deficiency), frightened)
        assert gap >= MIN_STATE_DISTANCE, (
            f"under {deficiency} the {role} ghost is {gap:.0f} from a frightened one"
        )


def test_themes_cycle_through_all_and_wrap() -> None:
    seen = [CLASSIC]
    for _ in range(len(THEMES)):
        seen.append(next_theme(seen[-1]))
    assert [t.name for t in seen[:-1]] == list(THEMES)
    assert seen[-1] is CLASSIC


def test_contrast_ratio_matches_the_wcag_reference_values() -> None:
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)
    assert contrast_ratio((255, 255, 255), (255, 255, 255)) == pytest.approx(1.0)
    # #777777 on white is the textbook "just fails AA" example at ~4.48:1.
    assert contrast_ratio((0x77, 0x77, 0x77), (255, 255, 255)) == pytest.approx(4.48, abs=0.01)


def test_theme_palettes_are_immutable() -> None:
    with pytest.raises(TypeError):
        CLASSIC.ghosts["chaser"] = (0, 0, 0)  # type: ignore[index]
