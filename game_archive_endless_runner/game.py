"""Dino-style 3-lane endless runner - cactus obstacles only.

Original Pygame shapes throughout; no Chrome/Google artwork or assets are used or copied.

Why the road scrolls vertically
-------------------------------
Three LEFT/CENTER/RIGHT lanes only mean something if obstacles travel perpendicular to them.
With horizontally scrolling cacti every obstacle eventually crosses every x position, so
sideways movement can buy time but can never avoid anything - which is exactly why the first
build played as jump-only. Cacti therefore descend the road while the dino changes lanes
across it.

Each control earns its place:
  * a single cactus in your lane  -> change lane OR jump it
  * two adjacent lanes blocked    -> you must move to the free lane
  * two cacti in the SAME lane    -> one jump cannot cover both, you must change lane
  * a full row of three           -> lane changes cannot help, you must jump

No webcam, no CNN, no torch imports here. Physics reads `Controls`, never the keyboard.
"""

import random

import pygame

# --- window ------------------------------------------------------------------------------
WIDTH, HEIGHT = 960, 540
FPS = 60

# --- road and lanes -----------------------------------------------------------------------
ROAD_LEFT, ROAD_RIGHT = 210, 750
LANE_COUNT = 3
LANE_WIDTH = (ROAD_RIGHT - ROAD_LEFT) // LANE_COUNT              # 180 px
LANE_X = [ROAD_LEFT + LANE_WIDTH // 2 + i * LANE_WIDTH for i in range(LANE_COUNT)]
CENTER_LANE = 1

# --- dino ---------------------------------------------------------------------------------
PLAYER_WIDTH, PLAYER_HEIGHT = 46, 58
PLAYER_ROW_Y = 392                  # where the dino's feet sit on the road
LANE_SLIDE_SPEED = 1150.0           # px/s while sliding between lanes (~0.16 s per lane)
LANE_COOLDOWN = 0.10                # guards against a mis-driven caller chaining lane changes

GRAVITY = 2000.0
JUMP_VELOCITY = -620.0              # ~96 px hop over ~0.62 s
LANDING_RECOVERY = 0.18             # brief pause before another jump is allowed

# --- cacti ---------------------------------------------------------------------------------
SCROLL_SPEED_START = 250.0
SCROLL_SPEED_MAX = 430.0            # capped so play stays fair with ~100 ms gesture latency
SCROLL_SPEED_GAIN = 7.0             # px/s added per second survived

GAP_SECONDS_EARLY = (1.85, 2.60)
GAP_SECONDS_LATE = (1.40, 2.00)
GAP_TIGHTEN_OVER = 90.0
SAME_LANE_GAP_SECONDS = 0.70        # tighter than jump airtime + recovery, so it forces a move

CACTUS_W, CACTUS_H = 46, 52
HITBOX_INSET = 6                    # forgiving collision, so a near miss reads as a miss

SCORE_PER_SECOND = 10.0

# --- palette ---------------------------------------------------------------------------------
SKY_TOP = (247, 250, 252)
SKY_BOTTOM = (226, 234, 242)
ROAD = (238, 232, 214)
ROAD_EDGE = (176, 166, 146)
LANE_MARK = (206, 198, 180)
DINO_BODY = (92, 104, 112)
DINO_DARK = (62, 72, 80)
SHADOW = (198, 192, 176)
CACTUS = (74, 152, 92)
CACTUS_DARK = (52, 118, 70)
TEXT = (70, 78, 88)
TEXT_DIM = (140, 150, 160)
ACCENT = (196, 84, 72)


class Controls:
    """Generic control state.

    LEFT and RIGHT are **edge-triggered lane-change requests**, not held movement: one
    request moves the dino exactly one lane. A control source that keeps asserting LEFT
    therefore cannot slide the dino across several lanes. Objective 9 maps a stable gesture
    change straight onto these calls.
    """

    def __init__(self):
        self._lane_shift = 0
        self._jump_requested = False

    def request_left(self):
        self._lane_shift = -1

    def request_right(self):
        self._lane_shift = +1

    def request_jump(self):
        self._jump_requested = True

    def consume_lane_shift(self):
        """Take the pending lane change: -1, 0 or +1. Cleared, so one request = one lane."""
        shift = self._lane_shift
        self._lane_shift = 0
        return shift

    def consume_jump(self):
        requested = self._jump_requested
        self._jump_requested = False
        return requested

    def reset(self):
        self._lane_shift = 0
        self._jump_requested = False


class Dino:
    """The player: a small dinosaur that occupies one of three lanes."""

    def __init__(self):
        self.lane = CENTER_LANE
        self.x = float(LANE_X[CENTER_LANE])
        self.hop = 0.0                  # pixels above the road while jumping
        self.velocity_y = 0.0
        self.on_ground = True
        self.lane_cooldown = 0.0
        self.recovery = 0.0
        self.step_timer = 0.0
        self.step_phase = 0
        self.rect = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
        self._sync_rect()

    def _sync_rect(self):
        self.rect.centerx = int(self.x)
        self.rect.bottom = int(PLAYER_ROW_Y - self.hop)

    @property
    def hitbox(self):
        """Ground footprint used for collisions; ignores the visual hop."""
        box = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
        box.centerx = int(self.x)
        box.bottom = PLAYER_ROW_Y
        return box.inflate(-HITBOX_INSET * 2, -HITBOX_INSET * 2)

    def update(self, dt, controls):
        self.lane_cooldown = max(0.0, self.lane_cooldown - dt)
        self.recovery = max(0.0, self.recovery - dt)

        # --- lane change: one request, one lane, clamped to the road -----------------------
        shift = controls.consume_lane_shift()
        if shift and self.lane_cooldown <= 0.0:
            target = self.lane + shift
            if 0 <= target < LANE_COUNT:
                self.lane = target
                self.lane_cooldown = LANE_COOLDOWN

        # Slide toward the lane centre rather than snapping, for readability.
        target_x = float(LANE_X[self.lane])
        if self.x < target_x:
            self.x = min(target_x, self.x + LANE_SLIDE_SPEED * dt)
        elif self.x > target_x:
            self.x = max(target_x, self.x - LANE_SLIDE_SPEED * dt)

        # --- jump: only from the ground, and only after the landing recovery ---------------
        if controls.consume_jump() and self.on_ground and self.recovery <= 0.0:
            self.velocity_y = JUMP_VELOCITY
            self.on_ground = False

        if not self.on_ground:
            self.velocity_y += GRAVITY * dt
            self.hop -= self.velocity_y * dt
            if self.hop <= 0.0:
                self.hop = 0.0
                self.velocity_y = 0.0
                self.on_ground = True
                self.recovery = LANDING_RECOVERY

        if self.on_ground:
            self.step_timer += dt
            if self.step_timer >= 0.11:
                self.step_timer = 0.0
                self.step_phase ^= 1
        else:
            self.step_phase = 0

        self._sync_rect()

    def draw(self, surface):
        # Shadow shrinks as the dino rises, which is what sells the hop from this angle.
        shrink = min(1.0, self.hop / 90.0)
        shadow_w = int(PLAYER_WIDTH * (1.0 - 0.35 * shrink))
        shadow_h = int(12 * (1.0 - 0.4 * shrink))
        pygame.draw.ellipse(surface, SHADOW,
                            (int(self.x) - shadow_w // 2, PLAYER_ROW_Y - shadow_h // 2,
                             shadow_w, max(4, shadow_h)))

        x, y = self.rect.x, self.rect.y
        pygame.draw.polygon(surface, DINO_BODY,
                            [(x, y + 30), (x + 12, y + 22), (x + 12, y + 42)])
        pygame.draw.rect(surface, DINO_BODY, (x + 8, y + 20, 24, 24), border_radius=5)
        pygame.draw.rect(surface, DINO_BODY, (x + 24, y + 6, 12, 18), border_radius=3)
        pygame.draw.rect(surface, DINO_BODY, (x + 24, y, 20, 16), border_radius=4)
        pygame.draw.rect(surface, DINO_BODY, (x + 40, y + 8, 6, 6), border_radius=2)
        pygame.draw.rect(surface, (250, 252, 255), (x + 35, y + 4, 5, 5), border_radius=1)
        pygame.draw.rect(surface, DINO_DARK, (x + 36, y + 5, 3, 3))
        pygame.draw.rect(surface, DINO_DARK, (x + 20, y + 26, 8, 4), border_radius=2)

        if self.on_ground:
            front, back = (14, 4) if self.step_phase == 0 else (4, 14)
            pygame.draw.rect(surface, DINO_DARK, (x + 12, y + 42, 7, front), border_radius=2)
            pygame.draw.rect(surface, DINO_DARK, (x + 23, y + 42, 7, back), border_radius=2)
        else:
            pygame.draw.rect(surface, DINO_DARK, (x + 12, y + 42, 7, 9), border_radius=2)
            pygame.draw.rect(surface, DINO_DARK, (x + 23, y + 44, 9, 7), border_radius=2)


class Cactus:
    """The only obstacle type: a cactus in one lane, travelling down the road."""

    def __init__(self, lane, y=-CACTUS_H - 10):
        self.lane = lane
        self.y = float(y)
        self.arms = random.choice([("l",), ("r",), ("l", "r")])
        self.rect = pygame.Rect(0, 0, CACTUS_W, CACTUS_H)
        self.rect.centerx = LANE_X[lane]
        self.rect.y = int(self.y)

    @property
    def hitbox(self):
        return self.rect.inflate(-HITBOX_INSET * 2, -HITBOX_INSET * 2)

    def update(self, dt, speed):
        self.y += speed * dt
        self.rect.y = int(self.y)

    @property
    def finished(self):
        return self.rect.top > HEIGHT + 10

    def draw(self, surface):
        trunk = pygame.Rect(self.rect.centerx - 9, self.rect.y, 18, CACTUS_H)
        pygame.draw.rect(surface, CACTUS, trunk, border_radius=9)
        pygame.draw.rect(surface, CACTUS_DARK, trunk, width=2, border_radius=9)
        for side in self.arms:
            direction = -1 if side == "l" else 1
            arm = pygame.Rect(0, 0, 16, 9)
            arm.centery = self.rect.y + CACTUS_H // 2
            arm.centerx = self.rect.centerx + direction * 14
            stub = pygame.Rect(0, 0, 9, 20)
            stub.centerx = self.rect.centerx + direction * 20
            stub.bottom = arm.centery + 5
            for shape in (arm, stub):
                pygame.draw.rect(surface, CACTUS, shape, border_radius=4)
                pygame.draw.rect(surface, CACTUS_DARK, shape, width=2, border_radius=4)


class Game:
    """Owns the loop, state and rendering. `self.controls` is the seam for Objective 9."""

    def __init__(self, caption="Gesture Dino - 3 lanes - keyboard mode"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("consolas", 52, bold=True)
        self.font = pygame.font.SysFont("consolas", 24)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.background = self._make_background()

        self.controls = Controls()
        self.running = True
        self.reset()

    # --- state -----------------------------------------------------------------------------
    def reset(self):
        """Full reset: dino, cacti, score, timers, difficulty and game-over state."""
        self.dino = Dino()
        self.cacti = []
        self.score = 0.0
        self.elapsed = 0.0
        self.scroll_speed = SCROLL_SPEED_START
        self.spawn_timer = 1.8                  # a calm moment before the first pattern
        self.pending_same_lane = None           # second half of a same-lane pair
        self.last_pattern = None
        self.dashes = [float(y) for y in range(-40, HEIGHT + 40, 80)]
        self.game_over = False
        self.controls.reset()

    def _make_background(self):
        surface = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            blend = y / HEIGHT
            surface.fill(
                tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * blend) for i in range(3)),
                pygame.Rect(0, y, WIDTH, 1))
        return surface

    # --- input ------------------------------------------------------------------------------
    def handle_events(self):
        """Keyboard only, and only as key *presses* - lane changes are edge-triggered."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and self.game_over:
                    self.reset()
                elif not self.game_over:
                    if event.key in (pygame.K_a, pygame.K_LEFT):
                        self.controls.request_left()
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        self.controls.request_right()
                    elif event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        self.controls.request_jump()

    # --- spawning ----------------------------------------------------------------------------
    def _gap_seconds(self):
        progress = min(1.0, self.elapsed / GAP_TIGHTEN_OVER)
        low = GAP_SECONDS_EARLY[0] + (GAP_SECONDS_LATE[0] - GAP_SECONDS_EARLY[0]) * progress
        high = GAP_SECONDS_EARLY[1] + (GAP_SECONDS_LATE[1] - GAP_SECONDS_EARLY[1]) * progress
        return random.uniform(low, high)

    def _choose_pattern(self):
        """Pick the next pattern, never the same demanding one twice in a row."""
        choices = ["single", "wall", "same_lane", "full_row"]
        weights = [0.30, 0.28, 0.20, 0.22]
        if self.last_pattern in ("full_row", "same_lane"):
            # Keep a demanding pattern from being followed by another, so the player is
            # never asked for a rapid left-right-left scramble.
            choices, weights = ["single", "wall"], [0.55, 0.45]
        return random.choices(choices, weights=weights, k=1)[0]

    def spawn_pattern(self, dt):
        self.spawn_timer -= dt
        if self.spawn_timer > 0.0:
            return

        if self.pending_same_lane is not None:
            self.cacti.append(Cactus(self.pending_same_lane))
            self.pending_same_lane = None
            self.spawn_timer = self._gap_seconds()
            return

        pattern = self._choose_pattern()
        self.last_pattern = pattern

        if pattern == "single":
            self.cacti.append(Cactus(random.randrange(LANE_COUNT)))

        elif pattern == "wall":
            # Two ADJACENT lanes, so the free lane is always reachable in one direction.
            first = random.choice([0, 1])
            self.cacti.append(Cactus(first))
            self.cacti.append(Cactus(first + 1))

        elif pattern == "same_lane":
            # Two cacti in one lane, closer together than jump airtime plus landing
            # recovery - a single jump cannot cover both, so a lane change is required.
            lane = random.randrange(LANE_COUNT)
            self.cacti.append(Cactus(lane))
            self.pending_same_lane = lane
            self.spawn_timer = SAME_LANE_GAP_SECONDS
            return

        else:  # full_row - lanes cannot help, this one has to be jumped
            for lane in range(LANE_COUNT):
                self.cacti.append(Cactus(lane))

        self.spawn_timer = self._gap_seconds()

    # --- update -------------------------------------------------------------------------------
    def update(self, dt):
        if self.game_over:
            return                              # score, cacti and scrolling all frozen

        self.elapsed += dt
        self.score += SCORE_PER_SECOND * dt
        self.scroll_speed = min(SCROLL_SPEED_MAX,
                                SCROLL_SPEED_START + SCROLL_SPEED_GAIN * self.elapsed)

        self.dino.update(dt, self.controls)
        self.spawn_pattern(dt)

        for cactus in self.cacti:
            cactus.update(dt, self.scroll_speed)
        self.cacti = [c for c in self.cacti if not c.finished]    # no unbounded growth

        for index, y in enumerate(self.dashes):
            y += self.scroll_speed * dt
            self.dashes[index] = y - (HEIGHT + 80) if y > HEIGHT + 40 else y

        # A jumping dino passes over whatever is in its lane.
        if self.dino.on_ground:
            hitbox = self.dino.hitbox
            for cactus in self.cacti:
                if cactus.lane == self.dino.lane and hitbox.colliderect(cactus.hitbox):
                    self.game_over = True       # set once; update() returns early afterwards
                    self.controls.reset()
                    break

    # --- draw ----------------------------------------------------------------------------------
    def draw(self):
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, ROAD,
                         pygame.Rect(ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, HEIGHT))
        pygame.draw.line(self.screen, ROAD_EDGE, (ROAD_LEFT, 0), (ROAD_LEFT, HEIGHT), 3)
        pygame.draw.line(self.screen, ROAD_EDGE, (ROAD_RIGHT, 0), (ROAD_RIGHT, HEIGHT), 3)
        for boundary in (1, 2):
            x = ROAD_LEFT + boundary * LANE_WIDTH
            for y in self.dashes:
                pygame.draw.rect(self.screen, LANE_MARK, (x - 2, int(y), 4, 34),
                                 border_radius=2)

        for cactus in self.cacti:
            cactus.draw(self.screen)
        self.dino.draw(self.screen)

        self.screen.blit(self.font.render(f"Score: {int(self.score)}", True, TEXT), (18, 14))
        self.screen.blit(self.font_small.render("GESTURE DINO", True, TEXT_DIM),
                         (WIDTH - 190, 16))
        self.screen.blit(
            self.font_small.render(f"{self.clock.get_fps():4.0f} FPS", True, TEXT_DIM),
            (WIDTH - 190, 38))
        self.screen.blit(
            self.font_small.render("A/<- lane left   D/-> lane right   SPACE jump   ESC quit",
                                   True, TEXT_DIM), (18, HEIGHT - 26))

        if self.game_over:
            veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            veil.fill((248, 250, 252, 205))
            self.screen.blit(veil, (0, 0))
            self._center(self.font_big, "GAME OVER", 170, ACCENT)
            self._center(self.font, f"Final Score: {int(self.score)}", 244, TEXT)
            self._center(self.font, "Press R to Restart", 292, (64, 140, 92))
            self._center(self.font_small, "ESC to Quit", 328, TEXT_DIM)

        pygame.display.flip()

    def _center(self, font, text, y, colour):
        surface = font.render(text, True, colour)
        self.screen.blit(surface, (WIDTH // 2 - surface.get_width() // 2, y))

    # --- loop -----------------------------------------------------------------------------------
    def run(self, frame_hook=None):
        """Main loop. `frame_hook` lets a later objective set controls once per frame."""
        try:
            while self.running:
                dt = min(self.clock.tick(FPS) / 1000.0, 0.05)   # clamp after a stall
                self.handle_events()
                if frame_hook is not None:
                    frame_hook(self)
                self.update(dt)
                self.draw()
        finally:
            pygame.quit()
