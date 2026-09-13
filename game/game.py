"""The Pac-Man-style game: rules, scoring, rounds, and simple original Pygame drawing.

Keyboard input reaches the player only through `Controls.request_direction`, the same seam
P8 will drive from the CNN. Nothing here imports torch, OpenCV or the recognizer, and the
game is fully playable with the keyboard alone.

Every graphic is drawn from primitives at run time - circles, rectangles and a polygon wedge
for the mouth. No sprite sheets, no maze image, no arcade artwork.

Frozen numbers
--------------
    maze                28 x 31 tiles, 20 px per tile
    player speed        6.2 tiles/s
    ghost speed         5.4 tiles/s base, +0.2 per round, capped at 6.0 (always below player)
    frightened ghosts   3.1 tiles/s        eaten ghosts 11.0 tiles/s
    turn buffer         0.35 s             (controls.REQUEST_GRACE)
    scatter/chase       7 / 20 / 7 / 20 / 5 / 20 / 5 s, then permanent chase
    frightened          6.0 s in round 1, -0.5 s per round, floor 2.0 s
    scoring             pellet 10, power pellet 50, ghost chain 200/400/800/1600
    lives               3, one extra life at 10,000 points, awarded once
    collision           centres within 0.6 of a tile
"""

import math
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controls import DIRECTIONS, Controls
from ghost import (CHASE, EATEN, EYE_COLOR, FRIGHTENED, HOUSE, SCATTER, Ghost, make_ghosts)
from maze import GHOST_SPAWN, TILE, Maze
from player import PLAYER_SPEED, Player

# --- layout ------------------------------------------------------------------------------
HUD_TOP = 46
HUD_BOTTOM = 38
WINDOW_WIDTH = 28 * TILE
WINDOW_HEIGHT = HUD_TOP + 31 * TILE + HUD_BOTTOM
FPS = 60

# --- rules -------------------------------------------------------------------------------
START_LIVES = 3
EXTRA_LIFE_SCORE = 10_000
PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_CHAIN = (200, 400, 800, 1600)

# (mode, seconds). A None duration means "stay here for the rest of the round".
MODE_SCHEDULE = [
    (SCATTER, 7.0), (CHASE, 20.0),
    (SCATTER, 7.0), (CHASE, 20.0),
    (SCATTER, 5.0), (CHASE, 20.0),
    (SCATTER, 5.0), (CHASE, None),
]

FRIGHTENED_BASE = 6.0
FRIGHTENED_PER_ROUND = 0.5
FRIGHTENED_FLOOR = 2.0
FRIGHTENED_FLASH_AT = 2.0       # start flashing this long before frightened mode ends

COLLISION_TILES = 0.6           # centres closer than this (in tiles) count as a touch

READY_SECONDS = 1.8
DYING_SECONDS = 1.4
ROUND_CLEAR_SECONDS = 1.6

# --- states ------------------------------------------------------------------------------
READY = "ready"
PLAYING = "playing"
DYING = "dying"
ROUND_CLEAR = "round_clear"
GAME_OVER = "game_over"

# --- palette -----------------------------------------------------------------------------
BACKGROUND = (8, 8, 20)
WALL_COLOR = (40, 66, 200)
WALL_EDGE = (92, 126, 255)
DOOR_COLOR = (230, 180, 200)
PELLET_COLOR = (255, 224, 176)
POWER_COLOR = (255, 236, 200)
PLAYER_COLOR = (255, 228, 60)
TEXT_COLOR = (240, 240, 245)
DIM_TEXT = (150, 150, 165)


class Game:
    def __init__(self, caption="Pac-Man - gesture controlled (keyboard)", headless=False,
                 seed=0, panel_width=0):
        pygame.init()
        self.headless = headless
        # An optional side panel, used by the P8 gesture build for the camera preview. The
        # maze always occupies the left WINDOW_WIDTH pixels, so nothing else moves.
        self.panel_width = panel_width
        size = (WINDOW_WIDTH + panel_width, WINDOW_HEIGHT)
        if headless:
            self.screen = pygame.Surface(size)
        else:
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
        self.big_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 34, bold=True)
        self.small_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)

        self.seed = seed
        self.maze = Maze()
        self.player = Player(self.maze)
        self.ghosts = make_ghosts(self.maze, seed=seed)
        self.controls = Controls()
        # A short line the HUD shows, e.g. "GESTURE: LEFT". Set by whatever is driving the
        # game; the game itself knows nothing about where input comes from.
        self.input_status = None
        self.running = True
        self.new_game()

    # --- lifecycle -----------------------------------------------------------------------
    def new_game(self):
        """A completely fresh game: score, lives, round, pellets, timers, buffers."""
        self.score = 0
        self.lives = START_LIVES
        self.round = 1
        self.extra_life_awarded = False
        self.maze.reset_pellets()
        self.controls.clear()
        self._start_round(reset_pellets=False)

    def _start_round(self, reset_pellets=True):
        if reset_pellets:
            self.maze.reset_pellets()
        for ghost in self.ghosts:
            ghost.set_round(self.round)
        self._reset_positions()
        self.state = READY
        self.state_timer = READY_SECONDS

    def _reset_positions(self):
        """Put everyone back at spawn without touching score, lives or eaten pellets."""
        self.player.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.controls.clear()
        self.mode_index = 0
        self.mode_timer = MODE_SCHEDULE[0][1]
        self.mode = MODE_SCHEDULE[0][0]
        self.frightened_timer = 0.0
        self.ghost_chain = 0
        for ghost in self.ghosts:
            ghost.rejoin_mode(self.mode)
            if not ghost.in_house:
                ghost.state = self.mode

    # --- control seam --------------------------------------------------------------------
    def request_direction(self, direction):
        """The one entry point for input of any kind. P8 will call this from the recognizer."""
        self.controls.request_direction(direction)

    # --- rules ---------------------------------------------------------------------------
    def frightened_duration(self):
        """6 s in round 1, half a second less each round, never below 2 s."""
        return max(FRIGHTENED_FLOOR,
                   FRIGHTENED_BASE - FRIGHTENED_PER_ROUND * (self.round - 1))

    def add_score(self, points):
        self.score += points
        if not self.extra_life_awarded and self.score >= EXTRA_LIFE_SCORE:
            self.extra_life_awarded = True      # once per game, never again
            self.lives += 1

    def update(self, dt):
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

        chaser = next((g for g in self.ghosts if g.role == "chaser"), None)
        for ghost in self.ghosts:
            ghost.update(dt, self.player, chaser)

        self._handle_collisions()
        self._check_round_clear()

    def _update_modes(self, dt):
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

        mode, duration = MODE_SCHEDULE[self.mode_index]
        if duration is None:
            return
        self.mode_timer -= dt
        if self.mode_timer <= 0.0:
            self.mode_index = min(self.mode_index + 1, len(MODE_SCHEDULE) - 1)
            self.mode, self.mode_timer = MODE_SCHEDULE[self.mode_index]
            if self.mode_timer is None:
                self.mode_timer = float("inf")
            for ghost in self.ghosts:
                ghost.set_mode(self.mode)        # ghosts reverse on a mode change
                ghost.rejoin_mode(self.mode)

    def _eat_pellet(self):
        eaten = self.maze.eat(self.player.tile)
        if eaten == "pellet":
            self.add_score(PELLET_SCORE)
        elif eaten == "power":
            self.add_score(POWER_PELLET_SCORE)
            self._start_frightened()

    def _start_frightened(self):
        """A new power pellet refreshes the timer and restarts the 200-point chain."""
        self.frightened_timer = self.frightened_duration()
        self.ghost_chain = 0
        for ghost in self.ghosts:
            ghost.frighten()

    def _handle_collisions(self):
        threshold = COLLISION_TILES * TILE
        for ghost in self.ghosts:
            if self.player.distance_to(ghost) > threshold:
                continue
            if ghost.is_edible:
                points = GHOST_CHAIN[min(self.ghost_chain, len(GHOST_CHAIN) - 1)]
                self.ghost_chain += 1
                self.add_score(points)
                ghost.get_eaten()
            elif ghost.is_dangerous:
                self._lose_life()
                return
            # EATEN and HOUSE ghosts are harmless and are simply passed through.

    def _lose_life(self):
        self.lives -= 1
        self.state = DYING
        self.state_timer = DYING_SECONDS

    def _after_death(self):
        if self.lives <= 0:
            self.state = GAME_OVER
            return
        self._reset_positions()                 # pellets stay eaten, score stays
        self.state = READY
        self.state_timer = READY_SECONDS

    def _check_round_clear(self):
        if self.maze.pellets_remaining == 0:
            self.state = ROUND_CLEAR
            self.state_timer = ROUND_CLEAR_SECONDS

    # --- input ---------------------------------------------------------------------------
    KEY_DIRECTIONS = {
        pygame.K_LEFT: "left", pygame.K_a: "left",
        pygame.K_RIGHT: "right", pygame.K_d: "right",
        pygame.K_UP: "up", pygame.K_w: "up",
        pygame.K_DOWN: "down", pygame.K_s: "down",
    }

    def handle_event(self, event):
        """Keyboard only submits direction requests - it never moves the player itself."""
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_r and self.state == GAME_OVER:
            self.new_game()
        elif event.key in self.KEY_DIRECTIONS:
            self.request_direction(self.KEY_DIRECTIONS[event.key])

    # --- drawing -------------------------------------------------------------------------
    def draw(self):
        self.screen.fill(BACKGROUND)
        self._draw_maze()
        self._draw_pellets()
        self._draw_player()
        for ghost in self.ghosts:
            self._draw_ghost(ghost)
        self._draw_hud()
        self._draw_banner()

    def _maze_origin(self):
        return 0, HUD_TOP

    def _draw_maze(self):
        origin_x, origin_y = self._maze_origin()
        for row in range(self.maze.rows):
            for column in range(self.maze.columns):
                cell = self.maze.layout[row][column]
                rect = pygame.Rect(origin_x + column * TILE, origin_y + row * TILE, TILE, TILE)
                if cell == "#":
                    inner = rect.inflate(-3, -3)
                    pygame.draw.rect(self.screen, WALL_COLOR, inner, border_radius=5)
                    pygame.draw.rect(self.screen, WALL_EDGE, inner, width=1, border_radius=5)
                elif cell == "-":
                    bar = pygame.Rect(rect.left, rect.centery - 2, TILE, 4)
                    pygame.draw.rect(self.screen, DOOR_COLOR, bar, border_radius=2)

    def _draw_pellets(self):
        origin_x, origin_y = self._maze_origin()
        for row, column in self.maze.pellets:
            x, y = self.maze.tile_center(row, column)
            pygame.draw.circle(self.screen, PELLET_COLOR,
                               (int(origin_x + x), int(origin_y + y)), 2)
        # Power pellets pulse so they read as different from ordinary pellets at a glance.
        pulse = 4 + 2 * abs(math.sin(pygame.time.get_ticks() / 220.0))
        for row, column in self.maze.power_pellets:
            x, y = self.maze.tile_center(row, column)
            pygame.draw.circle(self.screen, POWER_COLOR,
                               (int(origin_x + x), int(origin_y + y)), int(pulse))

    def _draw_player(self):
        origin_x, origin_y = self._maze_origin()
        centre = (origin_x + self.player.x, origin_y + self.player.y)
        radius = TILE * 0.45
        pygame.draw.circle(self.screen, PLAYER_COLOR, (int(centre[0]), int(centre[1])),
                           int(radius))

        # The mouth is a background-coloured wedge cut out of the circle.
        half_angle = 5 + 35 * self.player.mouth_openness
        if half_angle <= 6:
            return
        facing = math.radians(self.player.facing_angle)
        points = [centre]
        steps = 8
        for index in range(steps + 1):
            angle = facing - math.radians(half_angle) + \
                index * (2 * math.radians(half_angle) / steps)
            points.append((centre[0] + math.cos(angle) * radius * 1.15,
                           centre[1] - math.sin(angle) * radius * 1.15))
        pygame.draw.polygon(self.screen, BACKGROUND, points)

    def _draw_ghost(self, ghost):
        origin_x, origin_y = self._maze_origin()
        x, y = origin_x + ghost.x, origin_y + ghost.y
        radius = TILE * 0.44
        body = ghost.body_color()

        if body is not None:
            top = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
            pygame.draw.circle(self.screen, body, (int(x), int(y - radius * 0.15)),
                               int(radius))
            skirt = pygame.Rect(x - radius, y - radius * 0.15, radius * 2, radius * 1.05)
            pygame.draw.rect(self.screen, body, skirt)
            # three little feet, so the silhouette reads as a ghost rather than a blob
            foot = radius * 2 / 3.0
            for index in range(3):
                centre_x = x - radius + foot * (index + 0.5)
                pygame.draw.circle(self.screen, body,
                                   (int(centre_x), int(y + radius * 0.85)), int(foot / 2))

        # Eyes, always drawn: when the ghost is EATEN they are all that is left.
        if ghost.state == FRIGHTENED and body is not None:
            for offset in (-radius * 0.35, radius * 0.35):
                pygame.draw.circle(self.screen, EYE_COLOR,
                                   (int(x + offset), int(y - radius * 0.2)), 3)
        else:
            look_x, look_y = {"left": (-1, 0), "right": (1, 0),
                              "up": (0, -1), "down": (0, 1)}.get(ghost.direction or "left",
                                                                 (-1, 0))
            for offset in (-radius * 0.38, radius * 0.38):
                eye = (int(x + offset), int(y - radius * 0.25))
                pygame.draw.circle(self.screen, EYE_COLOR, eye, int(radius * 0.28))
                pygame.draw.circle(self.screen, (25, 25, 60),
                                   (int(eye[0] + look_x * 2.4), int(eye[1] + look_y * 2.4)),
                                   int(radius * 0.15))

    def hud_text(self):
        """The HUD's values as plain data, so the UI checks can read them without pixels."""
        return {"score": f"SCORE {self.score:>6}",
                "round": f"ROUND {self.round}",
                "lives": max(0, self.lives)}

    def _draw_hud(self):
        values = self.hud_text()
        score = self.font.render(values["score"], True, TEXT_COLOR)
        self.screen.blit(score, (12, 12))
        round_text = self.font.render(values["round"], True, TEXT_COLOR)
        self.screen.blit(round_text, (WINDOW_WIDTH - round_text.get_width() - 12, 12))

        base_y = HUD_TOP + self.maze.pixel_height + HUD_BOTTOM // 2
        label = self.small_font.render("LIVES", True, DIM_TEXT)
        self.screen.blit(label, (12, base_y - 7))
        for index in range(values["lives"]):
            pygame.draw.circle(self.screen, PLAYER_COLOR,
                               (66 + index * 26, base_y), int(TILE * 0.38))

        # The right of the bottom row carries exactly one line: the keyboard reminder when
        # the game stands alone, or the live command when a panel is carrying the rest of
        # the interface. Drawing a centred status *and* a right-aligned hint is what made
        # the old debug HUD overlap itself at common window widths.
        if self.panel_width:
            right, colour = self.input_status, PLAYER_COLOR
        else:
            right, colour = "ARROWS / WASD move   R restart   ESC quit", DIM_TEXT
        if right:
            text = self.small_font.render(right, True, colour)
            self.screen.blit(text, (WINDOW_WIDTH - text.get_width() - 12, base_y - 7))

    def banner_lines(self):
        """What the banner says in the current state: (headline, [detail lines]) or None.

        Separated from drawing so the UI checks can assert on the wording of every state
        without resorting to pixel comparisons. `lives` is already decremented by the time
        DYING begins, so it reads as the remaining count directly.
        """
        if self.state == READY:
            return "READY!", []
        if self.state == DYING:
            return "LIFE LOST", [f"Lives Remaining: {max(0, self.lives)}"]
        if self.state == ROUND_CLEAR:
            return f"ROUND {self.round} CLEARED!", []
        if self.state == GAME_OVER:
            return "GAME OVER", [f"Final Score: {self.score}",
                                 f"Round Reached: {self.round}",
                                 "",
                                 "Press R to Restart",
                                 "Press ESC to Quit"]
        return None

    def _draw_banner(self):
        lines = self.banner_lines()
        if lines is None:
            return
        headline, details = lines

        title = self.big_font.render(headline, True, PLAYER_COLOR)
        rendered = [self.font.render(line, True, TEXT_COLOR) if line else None
                    for line in details]

        line_height = self.font.get_linesize()
        height = title.get_height() + sum(line_height for _ in rendered)
        width = max([title.get_width()] + [r.get_width() for r in rendered if r])

        centre_y = HUD_TOP + self.maze.pixel_height * 0.62
        backdrop = pygame.Rect(0, 0, width + 44, height + 34)
        backdrop.center = (WINDOW_WIDTH // 2, centre_y)
        pygame.draw.rect(self.screen, BACKGROUND, backdrop, border_radius=8)
        pygame.draw.rect(self.screen, PLAYER_COLOR, backdrop, width=2, border_radius=8)

        y = backdrop.top + 17
        self.screen.blit(title, title.get_rect(midtop=(backdrop.centerx, y)))
        y += title.get_height() + 6
        for surface in rendered:
            if surface is not None:
                self.screen.blit(surface, surface.get_rect(midtop=(backdrop.centerx, y)))
            y += line_height

    # --- main loop -----------------------------------------------------------------------
    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                dt = min(dt, 0.05)               # a stall must not teleport anyone
                for event in pygame.event.get():
                    self.handle_event(event)
                self.update(dt)
                self.draw()
                pygame.display.flip()
        finally:
            pygame.quit()
