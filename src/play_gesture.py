"""CNN Gesture Controlled Pac-Man - the finished application.

    python src/play_gesture.py                   gesture control, keyboard still available
    python src/play_gesture.py --no-camera       keyboard only; loads no CNN, no webcam
    python src/play_gesture.py --benchmark 900   timed run, prints an integration report

    FIST          LEFT        THUMBS UP     UP
    OPEN PALM     RIGHT       THUMBS DOWN   DOWN

    arrows / WASD  move    P  pause    H  help    C  colour theme    M  sound    ESC  quit

One window, two areas: the maze on the left at its native 28x31 tiles, and a gesture panel
beside it. The panel never overlaps the maze - the window is simply wider by the panel.

Both inputs feed the same seam, `game.request_direction`, so neither can move Pac-Man directly
and the most recent request wins. The webcam preview is drawn in the main thread from a frame
the worker copied, so no OpenCV window and no pygame call ever crosses a thread boundary, and
the drawing code never touches the camera or the CNN.

Every colour comes from the game's active `Theme`, so the high-contrast and colour-blind-safe
themes apply to the panel and the start screen as well as to the maze.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from typing import TYPE_CHECKING, Final

import pygame

from game.engine import FPS, GAME_OVER, WINDOW_HEIGHT, WINDOW_WIDTH, Game
from game.theme import CLASSIC, Color, Theme
from src.game_integration import GestureController, Snapshot

if TYPE_CHECKING:
    from collections.abc import Callable

TITLE: Final = "CNN GESTURE CONTROLLED PAC-MAN"
FONT: Final = "consolas,dejavusansmono,monospace"

PANEL_WIDTH: Final = 264
PREVIEW_SIZE: Final = 200
PANEL_MARGIN: Final = 18
PANEL_LEFT: Final = WINDOW_WIDTH + PANEL_MARGIN
PANEL_RIGHT: Final = WINDOW_WIDTH + PANEL_WIDTH - PANEL_MARGIN
PANEL_CENTRE: Final = WINDOW_WIDTH + PANEL_WIDTH // 2

FRAME_HISTORY: Final = FPS * 60 * 30  # thirty minutes of frame times, then the oldest go
RECENT_FRAMES: Final = 90  # the on-screen FPS figure averages this many frames

# Plain ASCII labels on purpose. The default pygame/system fonts on this machine render hand
# emoji as missing-glyph boxes, which looks broken in a demonstration.
GESTURE_ROWS: Final = (
    ("FIST", "LEFT"),
    ("OPEN PALM", "RIGHT"),
    ("THUMBS UP", "UP"),
    ("THUMBS DOWN", "DOWN"),
)

KEYBOARD_ROWS: Final = (
    "Arrows/WASD move   R restart",
    "P pause   H help   ESC quit",
    "C theme   M sound",
)

GESTURE_HELP: Final = (
    ("Fist / Open palm", "Gesture: left / right"),
    ("Thumbs up / down", "Gesture: up / down"),
)

# Stands in for the worker's state when there is no worker at all (--no-camera).
_EMPTY: Final = Snapshot(status="off")

STARTING_STATUSES: Final = ("starting", "loading model", "opening camera")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# display logic - pure functions, so the UI checks can assert on wording without pixels
# ---------------------------------------------------------------------------------------
_COMMAND_OVERRIDES: Final = {
    "camera error": "CAMERA ERROR",
    "reconnecting": "RECONNECTING",
    "stopped": "GESTURE OFF",
}


def command_text(snapshot: Snapshot, fresh: bool, use_camera: bool) -> str:
    """The headline command, exactly as the panel prints it."""
    if not use_camera:
        return "GESTURE OFF"
    if snapshot.status in _COMMAND_OVERRIDES:
        return _COMMAND_OVERRIDES[snapshot.status]
    if snapshot.status in STARTING_STATUSES:
        return "STARTING"
    if not fresh:
        return "WAITING"
    if snapshot.stable_command is None:
        return "NO COMMAND"
    return snapshot.stable_command.upper()


def command_color(text: str, theme: Theme = CLASSIC) -> Color:
    if text.lower() in theme.directions:
        return theme.directions[text.lower()]
    if text == "CAMERA ERROR":
        return theme.error
    if text in ("WAITING", "STARTING", "RECONNECTING"):
        return theme.warn
    return theme.dim_text


def control_status(snapshot: Snapshot, fresh: bool, use_camera: bool) -> str:
    """The one-line health of gesture control, from the worker's own state."""
    if not use_camera or snapshot.status == "stopped":
        return "GESTURE CONTROL: OFF"
    if snapshot.status == "camera error":
        return "CAMERA ERROR"
    if snapshot.status in STARTING_STATUSES:
        return "GESTURE CONTROL: STARTING"
    if snapshot.status == "reconnecting":
        return "CAMERA RECONNECTING"
    return "GESTURE CONTROL: READY" if fresh else "GESTURE CONTROL: WAITING"


def status_color(text: str, theme: Theme = CLASSIC) -> Color:
    if text == "CAMERA ERROR":
        return theme.error
    if text.endswith("READY"):
        return theme.ok
    if text.endswith("OFF"):
        return theme.dim_text
    return theme.warn


def confidence_text(snapshot: Snapshot, fresh: bool, use_camera: bool) -> str | None:
    """A percentage only when it describes a command that is actually steering the game.

    Printing the raw confidence beside NO COMMAND or WAITING would suggest the game is acting on
    a direction it is deliberately ignoring, which is the confusion the raw / thresholded /
    stable split exists to prevent.
    """
    if not (use_camera and fresh and snapshot.stable_command):
        return None
    return f"Confidence: {snapshot.raw_confidence * 100:.1f}%"


def loading_text(snapshot: Snapshot, use_camera: bool, theme: Theme = CLASSIC) -> tuple[str, Color]:
    """What the start screen says about the recognizer while it is coming up."""
    if not use_camera:
        return "Gesture control off - keyboard only", theme.dim_text
    if snapshot.status == "camera error":
        return "CAMERA ERROR - keyboard controls available", theme.error
    if snapshot.status in STARTING_STATUSES:
        return "Initializing gesture recognition...", theme.warn
    if snapshot.status == "reconnecting":
        return "Reconnecting to the camera...", theme.warn
    if snapshot.status == "stopped":
        return "Gesture control stopped - keyboard still works", theme.dim_text
    return "Camera ready", theme.ok


def probability(text: str) -> float:
    """argparse type for a confidence threshold: a number in (0, 1]."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"threshold must be in (0, 1], got {value}")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


class GesturePacman:
    """The main-thread application: start screen, game, gesture panel, control bridge."""

    def __init__(
        self,
        use_camera: bool = True,
        threshold: float | None = None,
        window: int | None = None,
        agreement: int | None = None,
    ) -> None:
        self.use_camera = use_camera
        self.game = Game(caption=TITLE.title(), panel_width=PANEL_WIDTH)
        self.game.help_lines = [*GESTURE_HELP, *self.game.help_lines]
        self.controller: GestureController | None = (
            GestureController.with_worker(threshold, window, agreement) if use_camera else None
        )

        self.title_font = pygame.font.SysFont(FONT, 27, bold=True)
        self.heading = pygame.font.SysFont(FONT, 15, bold=True)
        self.command_font = pygame.font.SysFont(FONT, 26, bold=True)
        self.font = pygame.font.SysFont(FONT, 14)
        self.small = pygame.font.SysFont(FONT, 12)

        self.phase = "start"
        self.preview_surface: pygame.Surface | None = None
        self.preview_sequence = 0
        self.frame_times: deque[float] = deque(maxlen=FRAME_HISTORY)
        self._recent_frames: deque[float] = deque(maxlen=RECENT_FRAMES)
        self.previous_state = self.game.state
        self._static: dict[str, pygame.Surface] = {}
        self._static_theme: str | None = None

    @property
    def theme(self) -> Theme:
        return self.game.theme

    def _ensure_static(self) -> dict[str, pygame.Surface]:
        """Pre-render every piece of text that never changes, once per theme.

        The panel would otherwise rasterise about twenty strings per frame purely to say the same
        thing again; only the handful that actually change are rendered live.
        """
        theme = self.theme
        if self._static_theme == theme.name:
            return self._static
        cache: dict[str, pygame.Surface] = {}
        cache["panel_title_a"] = self.heading.render("CNN GESTURE CONTROLLED", True, theme.heading)
        cache["panel_title_b"] = self.title_font.render("PAC-MAN", True, theme.accent)
        cache["camera_label"] = self.heading.render("CAMERA / HAND AREA", True, theme.heading)
        cache["camera_hint"] = self.small.render("Keep your hand in this box", True, theme.dim_text)
        cache["current_label"] = self.heading.render("CURRENT GESTURE", True, theme.heading)
        cache["controls_label"] = self.heading.render("GESTURE CONTROLS", True, theme.heading)
        cache["keyboard_label"] = self.heading.render("KEYBOARD", True, theme.heading)
        cache["no_image"] = self.small.render("no camera image", True, theme.dim_text)
        for gesture, direction in GESTURE_ROWS:
            label = f"{gesture:<12}{direction}"
            cache[f"row:{direction}"] = self.font.render(label, True, theme.text)
            cache[f"row!{direction}"] = self.font.render(
                label, True, theme.directions[direction.lower()]
            )
        for line in KEYBOARD_ROWS:
            cache[f"key:{line}"] = self.small.render(line, True, theme.dim_text)
        self._static, self._static_theme = cache, theme.name
        return cache

    # --- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        if self.controller is not None:
            self.controller.start()

    def stop(self) -> bool:
        """Stop the worker and confirm it exited, so no camera thread is orphaned."""
        if self.controller is None:
            return True
        return self.controller.stop()

    def begin_play(self) -> None:
        """Leave the start screen. The recognizer has been running the whole time."""
        self.phase = "playing"
        self.previous_state = self.game.state
        if self.controller is not None:
            self.controller.reset()  # a gesture held while reading must not steer now

    # --- events --------------------------------------------------------------------------
    def handle_start_events(self) -> None:
        actions: dict[int, Callable[[], None]] = {
            pygame.K_SPACE: self.begin_play,
            pygame.K_RETURN: self.begin_play,
            pygame.K_KP_ENTER: self.begin_play,
            pygame.K_c: self.game.cycle_theme,
            pygame.K_m: self.game.toggle_mute,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.game.running = False
            elif event.type == pygame.KEYDOWN and event.key in actions:
                actions[event.key]()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            was_over = self.game.state == GAME_OVER
            self.game.handle_event(event)
            # A restart must not inherit a direction the player was holding beforehand.
            if was_over and self.game.state != GAME_OVER and self.controller is not None:
                self.controller.reset()

    # --- per-frame -----------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """One frame: gesture intent, then the game, exactly as the keyboard path does."""
        if self.controller is not None:
            self.controller.apply_to(self.game)
            self.game.input_status = self.controller.status_line()
        else:
            self.game.input_status = "GESTURE: OFF"

        self.game.update(dt)

        # Drop stale intent whenever play pauses: death, round change, ready screen. The camera
        # and the model keep running; only the control state is cleared.
        if self.game.state != self.previous_state:
            if self.game.state in ("dying", "round_clear", "ready") and self.controller:
                self.controller.reset()
            self.previous_state = self.game.state

    def draw(self) -> None:
        self.game.draw()
        self.draw_panel()

    # --- the start screen ----------------------------------------------------------------
    def draw_start(self) -> None:
        theme = self.theme
        screen = self.game.screen
        screen.fill(theme.background)
        centre = (WINDOW_WIDTH + PANEL_WIDTH) // 2

        def centred(surface: pygame.Surface, y: int) -> None:
            screen.blit(surface, surface.get_rect(midtop=(centre, y)))

        def rule(y: int) -> None:
            pygame.draw.line(screen, theme.panel_edge, (centre - 250, y), (centre + 250, y))

        centred(self.title_font.render(TITLE, True, theme.accent), 58)
        centred(
            self.font.render(
                "Real-time hand gesture control with a MobileNetV2 CNN", True, theme.dim_text
            ),
            100,
        )
        rule(134)

        centred(self.heading.render("GESTURE CONTROLS", True, theme.heading), 156)
        y = 190
        for gesture, direction in GESTURE_ROWS:
            name = self.font.render(gesture, True, theme.text)
            arrow = self.font.render("->", True, theme.dim_text)
            target = self.font.render(direction, True, theme.directions[direction.lower()])
            screen.blit(name, name.get_rect(midright=(centre - 30, y)))
            screen.blit(arrow, arrow.get_rect(center=(centre, y)))
            screen.blit(target, target.get_rect(midleft=(centre + 30, y)))
            y += 28

        centred(self.font.render("Keep your hand inside the camera area.", True, theme.text), 322)
        centred(
            self.small.render("Use only the four gestures shown above.", True, theme.dim_text), 346
        )

        rule(382)
        centred(self.heading.render("KEYBOARD", True, theme.heading), 402)
        centred(self.font.render("Arrow Keys / WASD  -  Move", True, theme.text), 432)
        centred(
            self.font.render("P  -  Pause     H  -  Help     ESC  -  Quit", True, theme.text), 456
        )
        centred(
            self.small.render(
                f"C  -  Colour theme ({theme.label})     M  -  Sound ({self.game.sound_status})",
                True,
                theme.dim_text,
            ),
            482,
        )

        snapshot = self.controller.poll()[0] if self.controller else _EMPTY
        message, colour = loading_text(snapshot, self.use_camera, theme)
        centred(self.font.render(message, True, colour), 536)

        prompt = self.heading.render("Press SPACE or ENTER to Start", True, theme.accent)
        box = prompt.get_rect(midtop=(centre, 592))
        pygame.draw.rect(screen, theme.accent, box.inflate(34, 20), width=2, border_radius=8)
        screen.blit(prompt, box)

    # --- the gesture panel ---------------------------------------------------------------
    def draw_panel(self) -> None:
        """Draw the side panel top-down, with the keyboard block pinned to the bottom.

        Laying it out with a running cursor rather than fixed offsets means the extra lines a
        camera failure adds cannot push anything off the end of the panel.
        """
        theme, screen = self.theme, self.game.screen
        pygame.draw.rect(
            screen, theme.panel_background, pygame.Rect(WINDOW_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        )
        pygame.draw.line(screen, theme.panel_edge, (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))
        if self.controller is not None:
            snapshot, fresh = self.controller.poll()
        else:
            snapshot, fresh = _EMPTY, False

        y = self._panel_identity(14)
        y = self._panel_camera(y, snapshot)
        y, command = self._panel_command(y, snapshot, fresh)
        y = self._panel_mapping(y, command)
        self._panel_health(y, snapshot, fresh)
        self._panel_keyboard(snapshot)

    def _panel_rule(self, y: int) -> None:
        pygame.draw.line(self.game.screen, self.theme.panel_edge, (PANEL_LEFT, y), (PANEL_RIGHT, y))

    def _panel_identity(self, y: int) -> int:
        cache, screen = self._ensure_static(), self.game.screen
        screen.blit(cache["panel_title_a"], (PANEL_LEFT, y))
        y += 19
        screen.blit(cache["panel_title_b"], (PANEL_LEFT, y))
        y += cache["panel_title_b"].get_height() + 10
        self._panel_rule(y)
        return y + 10

    def _panel_camera(self, y: int, snapshot: Snapshot) -> int:
        """What the CNN sees: the 300x300 ROI, scaled into the panel."""
        cache, screen, theme = self._ensure_static(), self.game.screen, self.theme
        screen.blit(cache["camera_label"], (PANEL_LEFT, y))
        y += 22
        box = pygame.Rect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
        box.midtop = (PANEL_CENTRE, y)
        self._update_preview(snapshot)
        if self.preview_surface is not None:
            screen.blit(self.preview_surface, box.topleft)
        else:
            pygame.draw.rect(screen, theme.background, box)
            screen.blit(cache["no_image"], cache["no_image"].get_rect(center=box.center))
        pygame.draw.rect(screen, theme.panel_edge, box, width=1)
        y = box.bottom + 6
        screen.blit(cache["camera_hint"], cache["camera_hint"].get_rect(midtop=(PANEL_CENTRE, y)))
        y += 22
        self._panel_rule(y)
        return y + 10

    def _panel_command(self, y: int, snapshot: Snapshot, fresh: bool) -> tuple[int, str]:
        """The command actually steering the game, with its confidence when it has one."""
        cache, screen, theme = self._ensure_static(), self.game.screen, self.theme
        screen.blit(cache["current_label"], (PANEL_LEFT, y))
        y += 22
        command = command_text(snapshot, fresh, self.use_camera)
        surface = self.command_font.render(command, True, command_color(command, theme))
        screen.blit(surface, (PANEL_LEFT, y))
        y += surface.get_height() + 2
        confidence = confidence_text(snapshot, fresh, self.use_camera)
        if confidence:
            screen.blit(self.font.render(confidence, True, theme.dim_text), (PANEL_LEFT, y))
        y += 22
        self._panel_rule(y)
        return y + 10, command

    def _panel_mapping(self, y: int, command: str) -> int:
        """The four gestures, with the one currently in force highlighted."""
        cache, screen = self._ensure_static(), self.game.screen
        screen.blit(cache["controls_label"], (PANEL_LEFT, y))
        y += 22
        for _, direction in GESTURE_ROWS:
            key = f"row!{direction}" if command == direction else f"row:{direction}"
            screen.blit(cache[key], (PANEL_LEFT, y))
            y += 20
        y += 4
        self._panel_rule(y)
        return y + 10

    def _panel_health(self, y: int, snapshot: Snapshot, fresh: bool) -> None:
        screen, theme = self.game.screen, self.theme
        status = control_status(snapshot, fresh, self.use_camera)
        screen.blit(self.font.render(status, True, status_color(status, theme)), (PANEL_LEFT, y))
        y += 19
        if status == "CAMERA ERROR":
            for line in ("Gesture control unavailable.", "Use Arrow Keys / WASD."):
                screen.blit(self.small.render(line, True, theme.dim_text), (PANEL_LEFT, y))
                y += 15

    def _panel_keyboard(self, snapshot: Snapshot) -> None:
        """Keyboard fallback and diagnostics, pinned to the bottom of the panel."""
        cache, screen = self._ensure_static(), self.game.screen
        bottom = WINDOW_HEIGHT - PANEL_MARGIN
        self._draw_diagnostics(screen, PANEL_LEFT, bottom, snapshot)
        bottom -= 20
        for line in reversed(KEYBOARD_ROWS):
            bottom -= 16
            screen.blit(cache[f"key:{line}"], (PANEL_LEFT, bottom))
        bottom -= 21
        screen.blit(cache["keyboard_label"], (PANEL_LEFT, bottom))
        self._panel_rule(bottom - 10)

    def _draw_diagnostics(
        self, screen: pygame.Surface, left: int, y: int, snapshot: Snapshot
    ) -> None:
        """Small print: the raw prediction and timings, kept clearly subordinate.

        The raw class is never the headline - it is a transient frame result, while the game
        acts only on the smoothed stable command shown above. It is still drawn in a colour that
        meets WCAG contrast, because small print is exactly the text low-vision users struggle
        with most.
        """
        total = sum(self._recent_frames)
        fps = len(self._recent_frames) / total if total else 0.0
        raw = (snapshot.raw_direction or "-")[:5]
        line = f"raw {raw:<6}{snapshot.inference_ms:5.1f}ms {fps:5.1f}fps"
        screen.blit(self.small.render(line, True, self.theme.dim_text), (left, y))

    def _update_preview(self, snapshot: Snapshot) -> None:
        """Convert the worker's ROI copy into a surface, only when a new one arrives."""
        if snapshot.preview is None or snapshot.sequence == self.preview_sequence:
            return
        self.preview_sequence = snapshot.sequence
        image = snapshot.preview  # BGR ndarray from the worker
        rgb = image[:, :, ::-1]  # to RGB without importing cv2 here
        surface = pygame.image.frombuffer(rgb.tobytes(), (image.shape[1], image.shape[0]), "RGB")
        self.preview_surface = pygame.transform.smoothscale(surface, (PREVIEW_SIZE, PREVIEW_SIZE))

    # --- loops ---------------------------------------------------------------------------
    def _record_frame(self, dt: float) -> None:
        self.frame_times.append(dt)
        self._recent_frames.append(dt)

    def run(self) -> None:
        self.start()
        released = False
        try:
            while self.game.running:
                dt = self.game.clock.tick(FPS) / 1000.0
                if self.phase == "start":
                    self.handle_start_events()
                    self.draw_start()
                else:
                    self._record_frame(dt)
                    self.handle_events()
                    self.step(min(dt, 0.05))
                    self.draw()
                pygame.display.flip()
        finally:
            released = self.stop()
            pygame.quit()
            self._report(released)

    def benchmark(self, frames: int) -> None:
        """Timed run with everything live, then an integration report."""
        self.start()
        self.begin_play()  # measure gameplay, not the menu
        print(f"running {frames} game frames with recognition active...")
        released = False
        try:
            for _ in range(frames):
                if not self.game.running:
                    break
                dt = self.game.clock.tick(FPS) / 1000.0
                self._record_frame(dt)
                self.handle_events()
                self.step(min(dt, 0.05))
                self.draw()
                pygame.display.flip()
        finally:
            released = self.stop()
            pygame.quit()
            self._report(released, verbose=True)

    def _report(self, released: bool, verbose: bool = False) -> None:
        if not self.frame_times:
            return
        times = sorted(self.frame_times)
        total = sum(times)
        fps = len(times) / total if total else 0.0
        p95 = times[max(0, int(len(times) * 0.95) - 1)] * 1000.0
        print(f"game: {len(times)} frames | {fps:.1f} FPS | frame p95 {p95:.1f} ms")
        if self.controller is None:
            return
        summary = self.controller.worker.summary() if self.controller.worker else {}
        if summary:
            print(
                f"camera: {int(summary['frames'])} frames | {summary['camera_fps']:.1f} FPS | "
                f"CNN mean {summary['cnn_mean_ms']:.2f} ms | "
                f"median {summary['cnn_median_ms']:.2f} | p95 {summary['cnn_p95_ms']:.2f} | "
                f"reconnects {int(summary['reconnects'])}"
            )
            print(
                f"requests applied: {self.controller.applied_count} | "
                f"snapshots published: {self.controller.state.writes} (one slot, never a queue)"
            )
        latency = self.controller.latency_summary()
        if latency:
            print(
                f"camera frame -> direction request: median {latency['median_ms']:.1f} ms | "
                f"p95 {latency['p95_ms']:.1f} ms ({int(latency['samples'])} samples)"
            )
        print(f"worker stopped cleanly: {released}")
        if verbose and self.controller.snapshot.error:
            print(f"last error: {self.controller.snapshot.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNN gesture controlled Pac-Man.")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="keyboard only; does not load the model or open the webcam",
    )
    parser.add_argument(
        "--benchmark",
        type=positive_int,
        metavar="FRAMES",
        help="run N frames with everything live, then print a report",
    )
    parser.add_argument(
        "--threshold",
        type=probability,
        default=None,
        help="override the frozen 0.90 confidence threshold (diagnostics only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="how much the application logs to the terminal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    app = GesturePacman(use_camera=not args.no_camera, threshold=args.threshold)
    if args.benchmark:
        app.benchmark(args.benchmark)
    else:
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
