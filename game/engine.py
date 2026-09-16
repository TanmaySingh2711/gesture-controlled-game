"""The Pac-Man-style game: rules, scoring, rounds, and original Pygame drawing.

Input reaches the player only through `Controls.request_direction` - the keyboard handler here
and the gesture recognizer in `src/` both use that one seam. Nothing in this package imports
torch, OpenCV or the recognizer, so the game is fully playable with the keyboard alone.

Every graphic is drawn from primitives - rectangles, circles and polygons. There are no sprite
sheets, no maze image and no arcade artwork. The static maze is rendered once into a cached
surface (and again only when the colour theme changes), so a frame costs one blit rather than
hundreds of rectangle draws.

Frozen numbers
--------------
    maze                28 x 31 tiles, 20 px per tile
    player speed        6.2 tiles/s
    ghost speed         5.4 tiles/s base, +0.2 per round, capped at 6.0 (always below player)
    Chaser "Elroy"      +0.3 at 20 pellets left, +0.5 at 10, capped at 6.1 (still below player)
    frightened ghosts   3.1 tiles/s        eaten ghosts 11.0 tiles/s
    turn buffer         0.35 s             (controls.REQUEST_GRACE)
    scatter/chase       7 / 20 / 7 / 20 / 5 / 20 / 5 s, then permanent chase
    frightened          6.0 s in round 1, -0.5 s per round, floor 2.0 s
    scoring             pellet 10, power pellet 50, ghost chain 200/400/800/1600
    bonus fruit         after 70 and 170 pellets, 9.5 s, 100 in round 1 rising to 5000
    lives               3, one extra life at 10,000 points, awarded once
    collision           centres within 0.6 of a tile

Keys
----
    arrows / WASD  move          P  pause         H or F1  help
    R              restart       M  mute sound    C        cycle colour theme
    ESC            quit
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import ClassVar, Final, TypedDict

import pygame

from .audio import Audio
from .controls import Controls
from .ghost import CHASE, EATEN, FRIGHTENED, SCATTER, Ghost, make_ghosts
from .maze import FRUIT_TILE, TILE, Maze
from .player import Player
from .profile import ProfileStore, default_profile_path
from .theme import CLASSIC, THEMES, Theme, next_theme

# --- layout ------------------------------------------------------------------------------
HUD_TOP: Final = 46
HUD_BOTTOM: Final = 38
WINDOW_WIDTH: Final = 28 * TILE
WINDOW_HEIGHT: Final = HUD_TOP + 31 * TILE + HUD_BOTTOM
FPS: Final = 60
FONT: Final = "consolas,dejavusansmono,monospace"

# --- rules -------------------------------------------------------------------------------
START_LIVES: Final = 3
EXTRA_LIFE_SCORE: Final = 10_000
PELLET_SCORE: Final = 10
POWER_PELLET_SCORE: Final = 50
GHOST_CHAIN: Final = (200, 400, 800, 1600)

# (mode, seconds). A None duration means "stay here for the rest of the round".
MODE_SCHEDULE: Final[list[tuple[str, float | None]]] = [
    (SCATTER, 7.0),
    (CHASE, 20.0),
    (SCATTER, 7.0),
    (CHASE, 20.0),
    (SCATTER, 5.0),
    (CHASE, 20.0),
    (SCATTER, 5.0),
    (CHASE, None),
]

FRIGHTENED_BASE: Final = 6.0
FRIGHTENED_PER_ROUND: Final = 0.5
FRIGHTENED_FLOOR: Final = 2.0
FRIGHTENED_FLASH_AT: Final = 2.0  # start flashing this long before frightened mode ends

COLLISION_TILES: Final = 0.6  # centres closer than this (in tiles) count as a touch

READY_SECONDS: Final = 1.8
DYING_SECONDS: Final = 1.4
ROUND_CLEAR_SECONDS: Final = 1.6

# Bonus fruit, as in the arcade: appears twice a round, below the ghost house, for a while.
FRUIT_SPAWN_AT: Final = (70, 170)  # pellets eaten this round
FRUIT_SECONDS: Final = 9.5
FRUIT_POINTS: Final = (100, 300, 500, 700, 1000, 2000, 3000, 5000)
FRUIT_COLORS: Final = ((235, 60, 70), (255, 120, 170), (255, 150, 50), (120, 210, 80))

POPUP_SECONDS: Final = 1.0

# --- states ------------------------------------------------------------------------------
READY: Final = "ready"
PLAYING: Final = "playing"
DYING: Final = "dying"
ROUND_CLEAR: Final = "round_clear"
GAME_OVER: Final = "game_over"

# --- palette (the classic theme; kept as module names for code that imports them) ----------
BACKGROUND: Final = CLASSIC.background
WALL_COLOR: Final = CLASSIC.wall_fill
WALL_EDGE: Final = CLASSIC.wall_edge
DOOR_COLOR: Final = CLASSIC.door
PELLET_COLOR: Final = CLASSIC.pellet
POWER_COLOR: Final = CLASSIC.power
PLAYER_COLOR: Final = CLASSIC.player
TEXT_COLOR: Final = CLASSIC.text
DIM_TEXT: Final = CLASSIC.dim_text

WALL_INSET: Final = 4  # px between a wall tile's edge and its outline

DEFAULT_HELP: Final[tuple[tuple[str, str], ...]] = (
    ("Arrow keys / WASD", "Move"),
    ("P", "Pause / resume"),
    ("H or F1", "Show / hide this help"),
    ("M", "Mute / unmute sound"),
    ("C", "Change colour theme"),
    ("R", "Restart after Game Over"),
    ("ESC", "Quit"),
)


class HudText(TypedDict):
    score: str
    round: str
    lives: int
    high_score: str


@dataclass
class Popup:
    """A short-lived score shown where it was earned."""

    text: str
    x: float
    y: float
    remaining: float = POPUP_SECONDS


class Game:
    KEY_DIRECTIONS: ClassVar[dict[int, str]] = {
        pygame.K_LEFT: "left",
        pygame.K_a: "left",
        pygame.K_RIGHT: "right",
        pygame.K_d: "right",
        pygame.K_UP: "up",
        pygame.K_w: "up",
        pygame.K_DOWN: "down",
        pygame.K_s: "down",
    }

    def __init__(
        self,
        caption: str = "Pac-Man - gesture controlled (keyboard)",
        headless: bool = False,
        seed: int = 0,
        panel_width: int = 0,
        *,
        profile: ProfileStore | None = None,
    ) -> None:
        pygame.init()
        self.headless = headless
        # Only a real window on a real display plays sound or writes the profile. Headless
        # games, and the dummy video driver the test suites use, do neither.
        self.interactive = not headless and os.environ.get("SDL_VIDEODRIVER") != "dummy"
        # An optional side panel for the gesture build. The maze always occupies the left
        # WINDOW_WIDTH pixels, so nothing else moves.
        self.panel_width = panel_width
        size = (WINDOW_WIDTH + panel_width, WINDOW_HEIGHT)
        if headless:
            self.screen = pygame.Surface(size)
        else:
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT, 20, bold=True)
        self.big_font = pygame.font.SysFont(FONT, 34, bold=True)
        self.small_font = pygame.font.SysFont(FONT, 15)

        self.profile_store = profile or ProfileStore(
            default_profile_path() if self.interactive else None
        )
        self.profile = self.profile_store.load()
        self.theme: Theme = THEMES.get(self.profile.theme, CLASSIC)
        self.audio = Audio(enabled=self.interactive)
        self.audio.muted = self.profile.muted

        self.seed = seed
        self.maze = Maze()
        self._pellets_total = self.maze.pellets_total
        self.player = Player(self.maze)
        self.ghosts: list[Ghost] = make_ghosts(self.maze, seed=seed)
        self.controls = Controls()
        # A short line the HUD shows, e.g. "GESTURE: LEFT". Set by whatever is driving the
        # game; the game itself knows nothing about where input comes from.
        self.input_status: str | None = None
        self.help_lines: list[tuple[str, str]] = list(DEFAULT_HELP)
        self.running = True
        self.paused = False
        self.show_help = False
        self.popups: list[Popup] = []
        self.fruit_timer = 0.0
        self._fruit_spawned: set[int] = set()
        self._maze_surface = self._build_maze_surface()
        self._help_backdrop: pygame.Surface | None = None

        # Game state, fully (re)initialised by new_game().
        self.score = 0
        self.lives = START_LIVES
        self.round = 1
        self.extra_life_awarded = False
        self.state: str = READY
        self.state_timer = READY_SECONDS
        self.mode: str = SCATTER
        self.mode_index = 0
        self.mode_timer = 0.0
        self.frightened_timer = 0.0
        self.ghost_chain = 0
        self.new_game()

    # --- lifecycle -----------------------------------------------------------------------
    def new_game(self) -> None:
        """A completely fresh game: score, lives, round, pellets, timers, buffers."""
        self.score = 0
        self.lives = START_LIVES
        self.round = 1
        self.extra_life_awarded = False
        self.paused = False
        self.maze.reset_pellets()
        self.controls.clear()
        self._start_round(reset_pellets=False)

    def _start_round(self, reset_pellets: bool = True) -> None:
        if reset_pellets:
            self.maze.reset_pellets()
        for ghost in self.ghosts:
            ghost.set_round(self.round)
        self._fruit_spawned.clear()
        self._reset_positions()
        self.state = READY
        self.state_timer = READY_SECONDS

    def _reset_positions(self) -> None:
        """Put everyone back at spawn without touching score, lives or eaten pellets."""
        self.player.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.controls.clear()
        self.mode_index = 0
        self.mode, first_duration = MODE_SCHEDULE[0]
        self.mode_timer = first_duration if first_duration is not None else math.inf
        self.frightened_timer = 0.0
        self.ghost_chain = 0
        self.fruit_timer = 0.0
        self.popups.clear()
        for ghost in self.ghosts:
            ghost.rejoin_mode(self.mode)
            if not ghost.in_house:
                ghost.state = self.mode

    # --- control seam --------------------------------------------------------------------
    def request_direction(self, direction: str) -> None:
        """The one entry point for input of any kind, keyboard and gesture alike."""
        self.controls.request_direction(direction)

    # --- player options ------------------------------------------------------------------
    def toggle_pause(self) -> None:
        """Freeze the game. A buffered turn is dropped so it cannot fire on resume."""
        if self.state == GAME_OVER:
            return
        self.paused = not self.paused
        self.controls.clear()

    def toggle_help(self) -> None:
        self.show_help = not self.show_help
        self.controls.clear()

    def toggle_mute(self) -> None:
        self.profile.muted = self.audio.toggle_mute()
        self.profile_store.save(self.profile)

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._maze_surface = self._build_maze_surface()
        self.profile.theme = theme.name
        self.profile_store.save(self.profile)

    def cycle_theme(self) -> None:
        self.set_theme(next_theme(self.theme))

    # --- rules ---------------------------------------------------------------------------
    def frightened_duration(self) -> float:
        """6 s in round 1, half a second less each round, never below 2 s."""
        return max(FRIGHTENED_FLOOR, FRIGHTENED_BASE - FRIGHTENED_PER_ROUND * (self.round - 1))

    def fruit_points(self) -> int:
        return FRUIT_POINTS[min(self.round - 1, len(FRUIT_POINTS) - 1)]

    def add_score(self, points: int) -> None:
        self.score += points
        if not self.extra_life_awarded and self.score >= EXTRA_LIFE_SCORE:
            self.extra_life_awarded = True  # once per game, never again
            self.lives += 1
            self.audio.play("life")

    def update(self, dt: float) -> None:
        if self.paused or self.show_help:
            return
        self._tick_popups(dt)
        if self.state == READY:
            self.state_timer -= dt
            if self.state_timer <= 0:
                self.state = PLAYING
            return
        if self.state == DYING:
            self.state_timer -= dt
            if self.state_timer <= 0:
                self._after_death()
            return
        if self.state == ROUND_CLEAR:
            self.state_timer -= dt
            if self.state_timer <= 0:
                self.round += 1
                self._start_round()
            return
        if self.state == GAME_OVER:
            return

        self._update_modes(dt)
        self.player.update(dt, self.controls)
        self._eat_pellet()
        self._update_fruit(dt)

        remaining = self.maze.pellets_remaining
        chaser = next((g for g in self.ghosts if g.role == "chaser"), None)
        for ghost in self.ghosts:
            ghost.set_pellets_remaining(remaining)
            ghost.update(dt, self.player, chaser)

        self._handle_collisions()
        self._check_round_clear()

    def _update_modes(self, dt: float) -> None:
        """Advance frightened mode if it is running, otherwise the scatter/chase schedule."""
        if self.frightened_timer > 0.0:
            self.frightened_timer -= dt
            flashing = self.frightened_timer <= FRIGHTENED_FLASH_AT
            for ghost in self.ghosts:
                if ghost.state == FRIGHTENED:
                    ghost.frightened_flash = flashing and int(self.frightened_timer * 6) % 2 == 0
            if self.frightened_timer <= 0.0:
                self.frightened_timer = 0.0
                self.ghost_chain = 0
                for ghost in self.ghosts:
                    ghost.unfrighten(self.mode)
            return

        _, duration = MODE_SCHEDULE[self.mode_index]
        if duration is None:
            return
        self.mode_timer -= dt
        if self.mode_timer <= 0.0:
            self.mode_index = min(self.mode_index + 1, len(MODE_SCHEDULE) - 1)
            self.mode, next_duration = MODE_SCHEDULE[self.mode_index]
            self.mode_timer = next_duration if next_duration is not None else math.inf
            for ghost in self.ghosts:
                ghost.set_mode(self.mode)  # ghosts reverse on a mode change
                ghost.rejoin_mode(self.mode)

    def _eat_pellet(self) -> None:
        eaten = self.maze.eat(self.player.tile)
        if eaten is None:
            return
        if eaten == "pellet":
            self.add_score(PELLET_SCORE)
            self.audio.waka()
        else:
            self.add_score(POWER_PELLET_SCORE)
            self.audio.play("power")
            self._start_frightened()
        self._maybe_spawn_fruit()

    def _maybe_spawn_fruit(self) -> None:
        eaten_this_round = self._pellets_total - self.maze.pellets_remaining
        for threshold in FRUIT_SPAWN_AT:
            if eaten_this_round >= threshold and threshold not in self._fruit_spawned:
                self._fruit_spawned.add(threshold)
                self.fruit_timer = FRUIT_SECONDS

    def _update_fruit(self, dt: float) -> None:
        if self.fruit_timer <= 0.0:
            return
        if self.player.tile == FRUIT_TILE:
            points = self.fruit_points()
            self.add_score(points)
            self.fruit_timer = 0.0
            self._popup(str(points), *self.maze.tile_center(*FRUIT_TILE))
            self.audio.play("fruit")
            return
        self.fruit_timer = max(0.0, self.fruit_timer - dt)

    def _start_frightened(self) -> None:
        """A new power pellet refreshes the timer and restarts the 200-point chain."""
        self.frightened_timer = self.frightened_duration()
        self.ghost_chain = 0
        for ghost in self.ghosts:
            ghost.frighten()

    def _handle_collisions(self) -> None:
        threshold = COLLISION_TILES * TILE
        for ghost in self.ghosts:
            if self.player.distance_to(ghost) > threshold:
                continue
            if ghost.is_edible:
                points = GHOST_CHAIN[min(self.ghost_chain, len(GHOST_CHAIN) - 1)]
                self.ghost_chain += 1
                self.add_score(points)
                self._popup(str(points), ghost.x, ghost.y)
                self.audio.play("ghost")
                ghost.get_eaten()
            elif ghost.is_dangerous:
                self._lose_life()
                return
            # EATEN and HOUSE ghosts are harmless and are simply passed through.

    def _lose_life(self) -> None:
        self.lives -= 1
        self.state = DYING
        self.state_timer = DYING_SECONDS
        self.audio.play("death")

    def _after_death(self) -> None:
        if self.lives <= 0:
            self.state = GAME_OVER
            self._record_high_score()
            return
        self._reset_positions()  # pellets stay eaten, score stays
        self.state = READY
        self.state_timer = READY_SECONDS

    def _check_round_clear(self) -> None:
        if self.maze.pellets_remaining == 0:
            self.state = ROUND_CLEAR
            self.state_timer = ROUND_CLEAR_SECONDS
            self.audio.play("round")

    def _record_high_score(self) -> None:
        if self.score > self.profile.high_score:
            self.profile.high_score = self.score
            self.profile_store.save(self.profile)

    @property
    def sound_status(self) -> str:
        """What the help screen reports - including when there is no audio device at all."""
        if not self.audio.available:
            return "unavailable"
        return "off" if self.audio.muted else "on"

    @property
    def high_score(self) -> int:
        return max(self.profile.high_score, self.score)

    def _popup(self, text: str, x: float, y: float) -> None:
        self.popups.append(Popup(text, x, y))

    def _tick_popups(self, dt: float) -> None:
        for popup in self.popups:
            popup.remaining -= dt
        self.popups = [popup for popup in self.popups if popup.remaining > 0.0]

    # --- input ---------------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        """Keys only submit requests or toggle options - they never move the player."""
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r and self.state == GAME_OVER:
            self.new_game()
        elif key == pygame.K_p:
            self.toggle_pause()
        elif key in (pygame.K_h, pygame.K_F1):
            self.toggle_help()
        elif key == pygame.K_m:
            self.toggle_mute()
        elif key == pygame.K_c:
            self.cycle_theme()
        elif key in self.KEY_DIRECTIONS:
            self.request_direction(self.KEY_DIRECTIONS[key])

    # --- drawing -------------------------------------------------------------------------
    def draw(self) -> None:
        self.screen.fill(self.theme.background)
        self._draw_maze()
        self._draw_pellets()
        self._draw_fruit()
        self._draw_player()
        if self.state != DYING:  # the ghosts vanish while Pac-Man's death plays out
            for ghost in self.ghosts:
                self._draw_ghost(ghost)
        self._draw_popups()
        self._draw_hud()
        self._draw_banner()
        if self.show_help:
            self._draw_help()

    def _maze_origin(self) -> tuple[int, int]:
        return 0, HUD_TOP

    def _build_maze_surface(self) -> pygame.Surface:
        """Render the static maze once: filled walls with one continuous outline.

        Each wall tile draws an edge only on sides that face open space, inset from the tile
        border. Edges run to the tile boundary where the neighbouring tile is also wall, so
        adjacent tiles join into one line; inner corners get a short L of their own. The
        result reads as solid connected walls instead of a grid of separate blocks.
        """
        theme, maze = self.theme, self.maze
        surface = pygame.Surface((maze.pixel_width, maze.pixel_height))
        surface.fill(theme.background)
        inset, width = WALL_INSET, theme.wall_width

        def wall(row: int, column: int) -> bool:
            return maze.is_wall(row, column)

        def fill(x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
            if w > 0 and h > 0:
                surface.fill(color, pygame.Rect(x, y, w, h))

        for row in range(maze.rows):
            for column in range(maze.columns):
                x0, y0 = column * TILE, row * TILE
                x1, y1 = x0 + TILE, y0 + TILE
                cell = maze.layout[row][column]
                if cell == "-":
                    fill(x0, y0 + TILE // 2 - 2, TILE, 4, theme.door)
                    continue
                if not wall(row, column):
                    continue

                up, down = wall(row - 1, column), wall(row + 1, column)
                left, right = wall(row, column - 1), wall(row, column + 1)

                left_x = x0 + (0 if left else inset)
                top_y = y0 + (0 if up else inset)
                right_x = x1 - (0 if right else inset)
                bottom_y = y1 - (0 if down else inset)
                fill(left_x, top_y, right_x - left_x, bottom_y - top_y, theme.wall_fill)

                edge = theme.wall_edge
                if not up:
                    fill(left_x, y0 + inset, right_x - left_x, width, edge)
                if not down:
                    fill(left_x, y1 - inset - width, right_x - left_x, width, edge)
                if not left:
                    fill(x0 + inset, top_y, width, bottom_y - top_y, edge)
                if not right:
                    fill(x1 - inset - width, top_y, width, bottom_y - top_y, edge)

                # Inner corners: both orthogonal neighbours are wall but the diagonal is open.
                for d_row, d_column in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    if not (
                        wall(row + d_row, column)
                        and wall(row, column + d_column)
                        and not wall(row + d_row, column + d_column)
                    ):
                        continue
                    corner_x = x0 if d_column < 0 else x1 - inset
                    corner_y = y0 if d_row < 0 else y1 - inset
                    fill(corner_x, corner_y, inset, inset, theme.background)
                    line_x = x0 + inset if d_column < 0 else x1 - inset - width
                    line_y = y0 + inset if d_row < 0 else y1 - inset - width
                    if d_row < 0:
                        fill(line_x, y0, width, inset + width, edge)
                    else:
                        fill(line_x, y1 - inset - width, width, inset + width, edge)
                    if d_column < 0:
                        fill(x0, line_y, inset + width, width, edge)
                    else:
                        fill(x1 - inset - width, line_y, inset + width, width, edge)
        return surface

    def _draw_maze(self) -> None:
        self.screen.blit(self._maze_surface, self._maze_origin())

    def _draw_pellets(self) -> None:
        origin_x, origin_y = self._maze_origin()
        for row, column in self.maze.pellets:
            x, y = self.maze.tile_center(row, column)
            pygame.draw.circle(
                self.screen, self.theme.pellet, (int(origin_x + x), int(origin_y + y)), 2
            )
        # Power pellets are larger and pulse, so they differ by size and motion, not colour.
        pulse = 4 + 2 * abs(math.sin(pygame.time.get_ticks() / 220.0))
        for row, column in self.maze.power_pellets:
            x, y = self.maze.tile_center(row, column)
            pygame.draw.circle(
                self.screen, self.theme.power, (int(origin_x + x), int(origin_y + y)), int(pulse)
            )

    def _draw_fruit(self) -> None:
        if self.fruit_timer <= 0.0:
            return
        origin_x, origin_y = self._maze_origin()
        x, y = self.maze.tile_center(*FRUIT_TILE)
        x, y = origin_x + x, origin_y + y
        color = FRUIT_COLORS[(self.round - 1) % len(FRUIT_COLORS)]
        stem = (110, 190, 90)
        pygame.draw.line(self.screen, stem, (x - 3, y), (x + 3, y - 8), 2)
        pygame.draw.line(self.screen, stem, (x + 4, y + 1), (x + 3, y - 8), 2)
        pygame.draw.circle(self.screen, color, (int(x - 4), int(y + 3)), 5)
        pygame.draw.circle(self.screen, color, (int(x + 4), int(y + 4)), 5)

    def _draw_player(self) -> None:
        origin_x, origin_y = self._maze_origin()
        centre = (origin_x + self.player.x, origin_y + self.player.y)
        radius = TILE * 0.45

        if self.state == DYING:
            # The mouth opens all the way round until nothing is left.
            progress = 1.0 - max(0.0, self.state_timer) / DYING_SECONDS
            half_angle = 40.0 + 140.0 * min(1.0, progress * 1.25)
            facing = math.radians(90.0)
        else:
            half_angle = 5.0 + 35.0 * self.player.mouth_openness
            facing = math.radians(self.player.facing_angle)

        if half_angle >= 179.0:
            return
        pygame.draw.circle(
            self.screen, self.theme.player, (int(centre[0]), int(centre[1])), int(radius)
        )
        if half_angle <= 6.0:
            return

        # The mouth is a background-coloured wedge cut out of the circle.
        half = math.radians(half_angle)
        steps = 12
        points = [centre]
        for index in range(steps + 1):
            angle = facing - half + index * (2 * half / steps)
            points.append(
                (
                    centre[0] + math.cos(angle) * radius * 1.15,
                    centre[1] - math.sin(angle) * radius * 1.15,
                )
            )
        pygame.draw.polygon(self.screen, self.theme.background, points)

    def ghost_body_color(self, ghost: Ghost) -> tuple[int, int, int] | None:
        if ghost.state == EATEN:
            return None  # eyes only
        if ghost.state == FRIGHTENED:
            return self.theme.frightened_flash if ghost.frightened_flash else self.theme.frightened
        return self.theme.ghosts[ghost.role]

    def _draw_ghost(self, ghost: Ghost) -> None:
        origin_x, origin_y = self._maze_origin()
        x, y = origin_x + ghost.x, origin_y + ghost.y
        radius = TILE * 0.44
        body = self.ghost_body_color(ghost)
        theme = self.theme

        if body is not None:
            pygame.draw.circle(self.screen, body, (int(x), int(y - radius * 0.15)), int(radius))
            skirt = pygame.Rect(x - radius, y - radius * 0.15, radius * 2, radius * 1.05)
            pygame.draw.rect(self.screen, body, skirt)
            # three little feet, so the silhouette reads as a ghost rather than a blob
            foot = radius * 2 / 3.0
            for index in range(3):
                centre_x = x - radius + foot * (index + 0.5)
                pygame.draw.circle(
                    self.screen, body, (int(centre_x), int(y + radius * 0.85)), int(foot / 2)
                )

        if ghost.state == FRIGHTENED and body is not None:
            # A frightened face differs in *shape* as well as colour: small eyes and a wavy
            # mouth, so the state is readable without relying on telling blue from red.
            for offset in (-radius * 0.35, radius * 0.35):
                pygame.draw.circle(
                    self.screen, theme.frightened_face, (int(x + offset), int(y - radius * 0.2)), 2
                )
            mouth_y = y + radius * 0.35
            points = [
                (x - radius * 0.7 + index * radius * 0.28, mouth_y + (-2 if index % 2 else 2))
                for index in range(6)
            ]
            pygame.draw.lines(self.screen, theme.frightened_face, False, points, 2)
            return

        look_x, look_y = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(
            ghost.direction or "left", (-1, 0)
        )
        for offset in (-radius * 0.38, radius * 0.38):
            eye = (int(x + offset), int(y - radius * 0.25))
            pygame.draw.circle(self.screen, theme.eye, eye, int(radius * 0.28))
            pygame.draw.circle(
                self.screen,
                theme.pupil,
                (int(eye[0] + look_x * 2.4), int(eye[1] + look_y * 2.4)),
                int(radius * 0.15),
            )

    def _draw_popups(self) -> None:
        origin_x, origin_y = self._maze_origin()
        for popup in self.popups:
            text = self.small_font.render(popup.text, True, self.theme.accent)
            rise = (POPUP_SECONDS - popup.remaining) * 14
            self.screen.blit(
                text, text.get_rect(center=(origin_x + popup.x, origin_y + popup.y - 12 - rise))
            )

    def hud_text(self) -> HudText:
        """The HUD's values as plain data, so the UI checks can read them without pixels."""
        return {
            "score": f"SCORE {self.score:>6}",
            "round": f"ROUND {self.round}",
            "lives": max(0, self.lives),
            "high_score": f"HIGH {self.high_score}",
        }

    def _draw_hud(self) -> None:
        theme, values = self.theme, self.hud_text()
        score = self.font.render(values["score"], True, theme.text)
        self.screen.blit(score, (12, 12))
        round_text = self.font.render(values["round"], True, theme.text)
        self.screen.blit(round_text, (WINDOW_WIDTH - round_text.get_width() - 12, 12))
        high = self.small_font.render(values["high_score"], True, theme.dim_text)
        self.screen.blit(high, high.get_rect(midtop=(WINDOW_WIDTH // 2, 16)))

        base_y = HUD_TOP + self.maze.pixel_height + HUD_BOTTOM // 2
        label = self.small_font.render("LIVES", True, theme.dim_text)
        self.screen.blit(label, (12, base_y - 7))
        for index in range(values["lives"]):
            pygame.draw.circle(
                self.screen, theme.player, (66 + index * 26, base_y), int(TILE * 0.38)
            )

        # The right of the bottom row carries exactly one line: the keyboard reminder when the
        # game stands alone, or the live command when a panel is carrying the rest of the
        # interface. Drawing a centred status *and* a right-aligned hint is what made the old
        # debug HUD overlap itself at common window widths.
        if self.panel_width:
            right, colour = self.input_status, theme.accent
        else:
            right, colour = "ARROWS/WASD move  P pause  H help", theme.dim_text
        if right:
            text = self.small_font.render(right, True, colour)
            self.screen.blit(text, (WINDOW_WIDTH - text.get_width() - 12, base_y - 7))

    def banner_lines(self) -> tuple[str, list[str]] | None:
        """What the banner says in the current state: (headline, [detail lines]) or None.

        Separated from drawing so the UI checks can assert on the wording of every state
        without resorting to pixel comparisons. `lives` is already decremented by the time
        DYING begins, so it reads as the remaining count directly.
        """
        if self.paused:
            return "PAUSED", ["Press P to resume"]
        if self.state == READY:
            return "READY!", []
        if self.state == DYING:
            return "LIFE LOST", [f"Lives Remaining: {max(0, self.lives)}"]
        if self.state == ROUND_CLEAR:
            return f"ROUND {self.round} CLEARED!", []
        if self.state == GAME_OVER:
            details = [f"Final Score: {self.score}", f"Round Reached: {self.round}"]
            if self.score and self.score >= self.profile.high_score:
                details.append("NEW HIGH SCORE!")
            return "GAME OVER", [*details, "", "Press R to Restart", "Press ESC to Quit"]
        return None

    def _draw_banner(self) -> None:
        lines = self.banner_lines()
        if lines is None or self.show_help:
            return
        headline, details = lines
        theme = self.theme

        title = self.big_font.render(headline, True, theme.accent)
        rendered = [self.font.render(line, True, theme.text) if line else None for line in details]

        line_height = self.font.get_linesize()
        height = title.get_height() + line_height * len(rendered)
        width = max([title.get_width()] + [r.get_width() for r in rendered if r])

        centre_y = HUD_TOP + self.maze.pixel_height * 0.62
        backdrop = pygame.Rect(0, 0, width + 44, height + 34)
        backdrop.center = (WINDOW_WIDTH // 2, int(centre_y))
        pygame.draw.rect(self.screen, theme.background, backdrop, border_radius=8)
        pygame.draw.rect(self.screen, theme.accent, backdrop, width=2, border_radius=8)

        y = backdrop.top + 17
        self.screen.blit(title, title.get_rect(midtop=(backdrop.centerx, y)))
        y += title.get_height() + 6
        for surface in rendered:
            if surface is not None:
                self.screen.blit(surface, surface.get_rect(midtop=(backdrop.centerx, y)))
            y += line_height

    def _draw_help(self) -> None:
        theme = self.theme
        maze_area = pygame.Rect(0, HUD_TOP, WINDOW_WIDTH, self.maze.pixel_height)
        if self._help_backdrop is None or self._help_backdrop.get_size() != maze_area.size:
            self._help_backdrop = pygame.Surface(maze_area.size, pygame.SRCALPHA)
            self._help_backdrop.fill((0, 0, 0, 215))
        self.screen.blit(self._help_backdrop, maze_area.topleft)

        title = self.big_font.render("HELP", True, theme.accent)
        self.screen.blit(title, title.get_rect(midtop=(maze_area.centerx, maze_area.top + 40)))
        y = maze_area.top + 100
        for keys, action in self.help_lines:
            key_text = self.font.render(keys, True, theme.accent)
            action_text = self.small_font.render(action, True, theme.text)
            self.screen.blit(key_text, key_text.get_rect(topright=(maze_area.centerx - 14, y)))
            self.screen.blit(action_text, (maze_area.centerx + 14, y + 3))
            y += 30
        footer = self.small_font.render(
            f"Theme: {theme.label}   Sound: {self.sound_status}",
            True,
            theme.dim_text,
        )
        self.screen.blit(footer, footer.get_rect(midtop=(maze_area.centerx, y + 20)))
        close = self.small_font.render("Press H to close", True, theme.dim_text)
        self.screen.blit(close, close.get_rect(midtop=(maze_area.centerx, y + 44)))

    # --- main loop -----------------------------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                dt = min(dt, 0.05)  # a stall must not teleport anyone
                for event in pygame.event.get():
                    self.handle_event(event)
                self.update(dt)
                self.draw()
                pygame.display.flip()
        finally:
            pygame.quit()
