"""CNN Gesture Controlled Pac-Man - the finished application.

    python src/play_gesture.py                   gesture control, keyboard still available
    python src/play_gesture.py --no-camera       keyboard only; loads no CNN, no webcam
    python src/play_gesture.py --benchmark 900   timed run, prints an integration report

    FIST          LEFT        THUMBS UP     UP
    OPEN PALM     RIGHT       THUMBS DOWN   DOWN

One window, two areas: the maze on the left at its native 28x31 tiles, and a gesture panel
beside it. The panel never overlaps the maze - the window is simply wider by the panel.

Both inputs feed the same seam, `game.request_direction`, so neither can move Pac-Man
directly and the most recent request wins. The webcam preview is drawn in the main thread
from a frame the worker copied, so no OpenCV window and no pygame call ever crosses a thread
boundary, and the drawing code never touches the camera or the CNN.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "game"))

import pygame

from game import (BACKGROUND, DIM_TEXT, FPS, GAME_OVER, PLAYER_COLOR, TEXT_COLOR,
                  WINDOW_HEIGHT, WINDOW_WIDTH, Game)
from game_integration import GestureController, Snapshot

TITLE = "CNN GESTURE CONTROLLED PAC-MAN"

PANEL_WIDTH = 264
PREVIEW_SIZE = 200

PANEL_BACKGROUND = (14, 14, 30)
PANEL_EDGE = (44, 44, 72)
HEADING_COLOR = (150, 170, 255)
OK_COLOR = (120, 220, 140)
WARN_COLOR = (255, 186, 96)
ERROR_COLOR = (255, 116, 116)

DIRECTION_COLORS = {
    "left": (96, 204, 255), "right": (128, 226, 128),
    "up": (255, 196, 96), "down": (206, 152, 255),
}

# Plain ASCII labels on purpose. The default pygame/system fonts on this machine render
# hand emoji as missing-glyph boxes, which looks broken in a demonstration; the spec allows
# text labels precisely for this case.
GESTURE_ROWS = (("FIST", "LEFT"), ("OPEN PALM", "RIGHT"),
                ("THUMBS UP", "UP"), ("THUMBS DOWN", "DOWN"))

KEYBOARD_ROWS = ("Arrow Keys / WASD   Move", "R   Restart", "ESC   Quit")

# Stands in for the worker's state when there is no worker at all (--no-camera).
_EMPTY = Snapshot(status="off")

STARTING_STATUSES = ("starting", "loading model", "opening camera")


# ---------------------------------------------------------------------------------------
# display logic - pure functions, so the UI checks can assert on wording without pixels
# ---------------------------------------------------------------------------------------
def command_text(snapshot, fresh, use_camera):
    """The headline command, exactly as the panel prints it."""
    if not use_camera:
        return "GESTURE OFF"
    if snapshot.status == "camera error":
        return "CAMERA ERROR"
    if snapshot.status in STARTING_STATUSES:
        return "STARTING"
    if snapshot.status == "stopped":
        return "GESTURE OFF"
    if not fresh:
        return "WAITING"
    if snapshot.stable_command is None:
        return "NO COMMAND"
    return snapshot.stable_command.upper()


def command_color(text):
    if text in ("LEFT", "RIGHT", "UP", "DOWN"):
        return DIRECTION_COLORS[text.lower()]
    if text == "CAMERA ERROR":
        return ERROR_COLOR
    if text in ("WAITING", "STARTING"):
        return WARN_COLOR
    return DIM_TEXT


def control_status(snapshot, fresh, use_camera):
    """The one-line health of gesture control, from the worker's own state."""
    if not use_camera:
        return "GESTURE CONTROL: OFF"
    if snapshot.status == "camera error":
        return "CAMERA ERROR"
    if snapshot.status in STARTING_STATUSES:
        return "GESTURE CONTROL: STARTING"
    if snapshot.status == "stopped":
        return "GESTURE CONTROL: OFF"
    return "GESTURE CONTROL: READY" if fresh else "GESTURE CONTROL: WAITING"


def status_color(text):
    if text == "CAMERA ERROR":
        return ERROR_COLOR
    if text.endswith("READY"):
        return OK_COLOR
    if text.endswith("OFF"):
        return DIM_TEXT
    return WARN_COLOR


def confidence_text(snapshot, fresh, use_camera):
    """A percentage only when it describes a command that is actually steering the game.

    Printing the raw confidence beside NO COMMAND or WAITING would suggest the game is
    acting on a direction it is deliberately ignoring, which is the confusion the raw /
    thresholded / stable split exists to prevent.
    """
    if not (use_camera and fresh and snapshot.stable_command):
        return None
    return f"Confidence: {snapshot.raw_confidence * 100:.1f}%"


def loading_text(snapshot, use_camera):
    """What the start screen says about the recognizer while it is coming up."""
    if not use_camera:
        return "Gesture control off - keyboard only", DIM_TEXT
    if snapshot.status == "camera error":
        return "CAMERA ERROR - keyboard controls available", ERROR_COLOR
    if snapshot.status in STARTING_STATUSES:
        return "Initializing gesture recognition...", WARN_COLOR
    if snapshot.status == "stopped":
        return "Gesture control stopped - keyboard still works", DIM_TEXT
    return "Camera ready", OK_COLOR


class GesturePacman:
    """The main-thread application: start screen, game, gesture panel, control bridge."""

    def __init__(self, use_camera=True, threshold=None, window=None, agreement=None):
        self.use_camera = use_camera
        self.game = Game(caption=TITLE.title(), panel_width=PANEL_WIDTH)
        self.controller = (GestureController.with_worker(threshold, window, agreement)
                           if use_camera else None)

        self.title_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 27,
                                              bold=True)
        self.heading = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15, bold=True)
        self.command_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 26,
                                                bold=True)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,monospace", 12)

        self.phase = "start"
        self.preview_surface = None
        self.preview_sequence = 0
        self.frame_times = []
        self.previous_state = self.game.state
        self._static = {}
        self._build_static()

    def _build_static(self):
        """Pre-render every piece of text that never changes.

        The panel would otherwise rasterise about twenty strings per frame purely to say
        the same thing again; only the handful that actually change are rendered live.
        """
        cache = self._static
        cache["panel_title_a"] = self.heading.render("CNN GESTURE CONTROLLED", True,
                                                     HEADING_COLOR)
        cache["panel_title_b"] = self.title_font.render("PAC-MAN", True, PLAYER_COLOR)
        cache["camera_label"] = self.heading.render("CAMERA / HAND AREA", True, HEADING_COLOR)
        cache["camera_hint"] = self.small.render("Keep your hand in this box", True, DIM_TEXT)
        cache["current_label"] = self.heading.render("CURRENT GESTURE", True, HEADING_COLOR)
        cache["controls_label"] = self.heading.render("GESTURE CONTROLS", True, HEADING_COLOR)
        cache["keyboard_label"] = self.heading.render("KEYBOARD", True, HEADING_COLOR)
        cache["no_image"] = self.small.render("no camera image", True, DIM_TEXT)
        for gesture, direction in GESTURE_ROWS:
            label = f"{gesture:<12}{direction}"
            cache[f"row:{direction}"] = self.font.render(label, True, TEXT_COLOR)
            cache[f"row!{direction}"] = self.font.render(label, True,
                                                         DIRECTION_COLORS[direction.lower()])
        for line in KEYBOARD_ROWS:
            cache[f"key:{line}"] = self.small.render(line, True, DIM_TEXT)

    # --- lifecycle -----------------------------------------------------------------------
    def start(self):
        if self.controller is not None:
            self.controller.start()

    def stop(self):
        """Stop the worker and confirm it exited, so no camera thread is orphaned."""
        if self.controller is None:
            return True
        return self.controller.stop()

    def begin_play(self):
        """Leave the start screen. The recognizer has been running the whole time."""
        self.phase = "playing"
        self.previous_state = self.game.state
        if self.controller is not None:
            self.controller.reset()     # a gesture held while reading must not steer now

    # --- events --------------------------------------------------------------------------
    def handle_start_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game.running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.begin_play()

    def handle_events(self):
        for event in pygame.event.get():
            was_over = self.game.state == GAME_OVER
            self.game.handle_event(event)
            # A restart must not inherit a direction the player was holding beforehand.
            if was_over and self.game.state != GAME_OVER and self.controller is not None:
                self.controller.reset()

    # --- per-frame -----------------------------------------------------------------------
    def step(self, dt):
        """One frame: gesture intent, then the game, exactly as the keyboard path does."""
        if self.controller is not None:
            self.controller.apply_to(self.game)
            self.game.input_status = self.controller.status_line()
        else:
            self.game.input_status = "GESTURE: OFF"

        self.game.update(dt)

        # Drop stale intent whenever play pauses: death, round change, ready screen. The
        # camera and the model keep running; only the control state is cleared.
        if self.game.state != self.previous_state:
            if self.game.state in ("dying", "round_clear", "ready") and self.controller:
                self.controller.reset()
            self.previous_state = self.game.state

    def draw(self):
        self.game.draw()
        self.draw_panel()

    # --- the start screen ----------------------------------------------------------------
    def draw_start(self):
        screen = self.game.screen
        screen.fill(BACKGROUND)
        width = WINDOW_WIDTH + PANEL_WIDTH
        centre = width // 2

        def centred(surface, y):
            screen.blit(surface, surface.get_rect(midtop=(centre, y)))

        centred(self.title_font.render(TITLE, True, PLAYER_COLOR), 64)
        centred(self.font.render("Real-time hand gesture control with a MobileNetV2 CNN",
                                 True, DIM_TEXT), 106)
        pygame.draw.line(screen, PANEL_EDGE, (centre - 250, 142), (centre + 250, 142))

        centred(self.heading.render("GESTURE CONTROLS", True, HEADING_COLOR), 168)
        y = 202
        for gesture, direction in GESTURE_ROWS:
            name = self.font.render(gesture, True, TEXT_COLOR)
            arrow = self.font.render("->", True, DIM_TEXT)
            target = self.font.render(direction, True, DIRECTION_COLORS[direction.lower()])
            screen.blit(name, name.get_rect(midright=(centre - 30, y)))
            screen.blit(arrow, arrow.get_rect(center=(centre, y)))
            screen.blit(target, target.get_rect(midleft=(centre + 30, y)))
            y += 28

        centred(self.font.render("Keep your hand inside the camera area.", True, TEXT_COLOR),
                336)
        centred(self.small.render("Use only the four gestures shown above.", True, DIM_TEXT),
                360)

        pygame.draw.line(screen, PANEL_EDGE, (centre - 250, 400), (centre + 250, 400))
        centred(self.heading.render("KEYBOARD", True, HEADING_COLOR), 422)
        centred(self.font.render("Arrow Keys / WASD  -  Move", True, TEXT_COLOR), 454)
        centred(self.font.render("R  -  Restart          ESC  -  Quit", True, TEXT_COLOR), 478)

        snapshot = self.controller.poll()[0] if self.controller else _EMPTY
        message, colour = loading_text(snapshot, self.use_camera)
        centred(self.font.render(message, True, colour), 540)

        prompt = self.heading.render("Press SPACE or ENTER to Start", True, PLAYER_COLOR)
        box = prompt.get_rect(midtop=(centre, 596))
        pygame.draw.rect(screen, PLAYER_COLOR, box.inflate(34, 20), width=2, border_radius=8)
        screen.blit(prompt, box)

    # --- the gesture panel ---------------------------------------------------------------
    def draw_panel(self):
        """Draw the side panel top-down, with the keyboard block pinned to the bottom.

        Laying it out with a running cursor rather than fixed offsets means the extra lines
        a camera failure adds cannot push anything off the end of the panel.
        """
        screen = self.game.screen
        panel = pygame.Rect(WINDOW_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, PANEL_BACKGROUND, panel)
        pygame.draw.line(screen, PANEL_EDGE, (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))

        if self.controller is not None:
            snapshot, fresh = self.controller.poll()
        else:
            snapshot, fresh = _EMPTY, False

        left = WINDOW_WIDTH + 18
        right = WINDOW_WIDTH + PANEL_WIDTH - 18
        centre = WINDOW_WIDTH + PANEL_WIDTH // 2
        cache = self._static

        def rule(y):
            pygame.draw.line(screen, PANEL_EDGE, (left, y), (right, y))

        # --- identity ---------------------------------------------------------------
        y = 14
        screen.blit(cache["panel_title_a"], (left, y))
        y += 19
        screen.blit(cache["panel_title_b"], (left, y))
        y += cache["panel_title_b"].get_height() + 10
        rule(y)
        y += 10

        # --- what the CNN sees ------------------------------------------------------
        screen.blit(cache["camera_label"], (left, y))
        y += 22
        box = pygame.Rect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
        box.midtop = (centre, y)
        self._update_preview(snapshot)
        if self.preview_surface is not None:
            screen.blit(self.preview_surface, box.topleft)
        else:
            pygame.draw.rect(screen, (26, 26, 44), box)
            screen.blit(cache["no_image"], cache["no_image"].get_rect(center=box.center))
        pygame.draw.rect(screen, PANEL_EDGE, box, width=1)
        y = box.bottom + 6
        screen.blit(cache["camera_hint"], cache["camera_hint"].get_rect(midtop=(centre, y)))
        y += 22
        rule(y)
        y += 10

        # --- the command actually steering the game ---------------------------------
        screen.blit(cache["current_label"], (left, y))
        y += 22
        command = command_text(snapshot, fresh, self.use_camera)
        surface = self.command_font.render(command, True, command_color(command))
        screen.blit(surface, (left, y))
        y += surface.get_height() + 2
        confidence = confidence_text(snapshot, fresh, self.use_camera)
        if confidence:
            screen.blit(self.font.render(confidence, True, DIM_TEXT), (left, y))
        y += 22
        rule(y)
        y += 10

        # --- the mapping, with the live one highlighted ------------------------------
        screen.blit(cache["controls_label"], (left, y))
        y += 22
        for _, direction in GESTURE_ROWS:
            key = f"row!{direction}" if command == direction else f"row:{direction}"
            screen.blit(cache[key], (left, y))
            y += 20
        y += 4
        rule(y)
        y += 10

        # --- health ------------------------------------------------------------------
        status = control_status(snapshot, fresh, self.use_camera)
        screen.blit(self.font.render(status, True, status_color(status)), (left, y))
        y += 19
        if status == "CAMERA ERROR":
            for line in ("Gesture control unavailable.", "Use Arrow Keys / WASD."):
                screen.blit(self.small.render(line, True, DIM_TEXT), (left, y))
                y += 15

        # --- keyboard fallback, pinned to the bottom ---------------------------------
        bottom = WINDOW_HEIGHT - 18
        self._draw_diagnostics(screen, left, bottom, snapshot)
        bottom -= 20
        for line in reversed(KEYBOARD_ROWS):
            bottom -= 16
            screen.blit(cache[f"key:{line}"], (left, bottom))
        bottom -= 21
        screen.blit(cache["keyboard_label"], (left, bottom))
        rule(bottom - 10)

    def _draw_diagnostics(self, screen, left, y, snapshot):
        """Small print: the raw prediction and timings, kept clearly subordinate.

        The raw class is never the headline - it is a transient frame result, while the
        game acts only on the smoothed stable command shown above.
        """
        fps = 0.0
        if self.frame_times:
            recent = self.frame_times[-90:]
            total = sum(recent)
            fps = len(recent) / total if total else 0.0
        raw = (snapshot.raw_direction or "-")[:5]
        line = f"raw {raw:<6}{snapshot.inference_ms:5.1f}ms {fps:5.1f}fps"
        screen.blit(self.small.render(line, True, (86, 86, 110)), (left, y))

    def _update_preview(self, snapshot):
        """Convert the worker's ROI copy into a surface, only when a new one arrives."""
        if snapshot.preview is None or snapshot.sequence == self.preview_sequence:
            return
        self.preview_sequence = snapshot.sequence
        image = snapshot.preview                      # BGR ndarray from the worker
        rgb = image[:, :, ::-1]                       # to RGB without importing cv2 here
        surface = pygame.image.frombuffer(rgb.tobytes(), (image.shape[1], image.shape[0]),
                                          "RGB")
        self.preview_surface = pygame.transform.smoothscale(surface,
                                                            (PREVIEW_SIZE, PREVIEW_SIZE))

    # --- loops ---------------------------------------------------------------------------
    def run(self):
        self.start()
        try:
            while self.game.running:
                dt = self.game.clock.tick(FPS) / 1000.0
                if self.phase == "start":
                    self.handle_start_events()
                    self.draw_start()
                else:
                    self.frame_times.append(dt)
                    self.handle_events()
                    self.step(min(dt, 0.05))
                    self.draw()
                pygame.display.flip()
        finally:
            released = self.stop()
            pygame.quit()
            self._report(released)

    def benchmark(self, frames):
        """Timed run with everything live, then an integration report."""
        self.start()
        self.begin_play()                             # measure gameplay, not the menu
        print(f"running {frames} game frames with recognition active...")
        try:
            for _ in range(frames):
                if not self.game.running:
                    break
                dt = self.game.clock.tick(FPS) / 1000.0
                self.frame_times.append(dt)
                self.handle_events()
                self.step(min(dt, 0.05))
                self.draw()
                pygame.display.flip()
        finally:
            released = self.stop()
            pygame.quit()
            self._report(released, verbose=True)

    def _report(self, released, verbose=False):
        if not self.frame_times:
            return
        times = sorted(self.frame_times)
        total = sum(self.frame_times)
        fps = len(self.frame_times) / total if total else 0.0
        p95 = times[int(len(times) * 0.95) - 1] * 1000.0
        print(f"game: {len(self.frame_times)} frames | {fps:.1f} FPS | "
              f"frame p95 {p95:.1f} ms")
        if self.controller is None:
            return
        summary = self.controller.worker.summary() if self.controller.worker else {}
        if summary:
            print(f"camera: {summary['frames']} frames | {summary['camera_fps']:.1f} FPS | "
                  f"CNN mean {summary['cnn_mean_ms']:.2f} ms | "
                  f"median {summary['cnn_median_ms']:.2f} | p95 {summary['cnn_p95_ms']:.2f}")
            print(f"requests applied: {self.controller.applied_count} | "
                  f"snapshots published: {self.controller.state.writes} "
                  f"(one slot, never a queue)")
        print(f"worker stopped cleanly: {released}")
        if verbose and self.controller.snapshot.error:
            print(f"last error: {self.controller.snapshot.error}")


def main():
    parser = argparse.ArgumentParser(description="CNN gesture controlled Pac-Man.")
    parser.add_argument("--no-camera", action="store_true",
                        help="keyboard only; does not load the model or open the webcam")
    parser.add_argument("--benchmark", type=int, metavar="FRAMES",
                        help="run N frames with everything live, then print a report")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the frozen 0.90 confidence threshold (diagnostics)")
    args = parser.parse_args()

    app = GesturePacman(use_camera=not args.no_camera, threshold=args.threshold)
    if args.benchmark:
        app.benchmark(args.benchmark)
    else:
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
