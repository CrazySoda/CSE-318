from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import pygame

# ────────────────────────── Config/Constants ────────────────────────── #
ROWS, COLS = 6, 9
CELL_SIZE = 100                  # 9 × 100 = 900 px, 6 × 100 = 600 px
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
FPS = 60

RED   = (220,  20,  60)
BLUE  = ( 30, 144, 255)
BLACK = ( 25,  25,  25)
WHITE = (250, 250, 250)
GRAY  = (200, 200, 200)

PLAYER_COLOR = {1: RED, 2: BLUE}

# Time (ms) a single explosion animation lasts
EXPLOSION_MS = 250

# ────────────────────────────── Helpers ────────────────────────────── #
def critical_mass(r: int, c: int) -> int:
    """Return the maximum orbs a cell (r,c) can hold before exploding."""
    if (r in (0, ROWS - 1)) and (c in (0, COLS - 1)):
        return 2          # corner
    if r in (0, ROWS - 1) or c in (0, COLS - 1):
        return 3          # edge
    return 4              # inner


@dataclass
class Cell:
    """Single cell on the board."""
    owner: Optional[int] = None          # 1 = Red, 2 = Blue, None = empty
    count: int = 0                      # number of orbs currently in cell

    def add_orb(self, player: int):
        self.owner = player          # always take / keep ownership
        self.count += 1


    def reset(self):
        self.owner, self.count = None, 0


@dataclass
class ExplosionAnim:
    """Track one exploding cell for drawing animation."""
    row: int
    col: int
    start_time: int                     # pygame.time.get_ticks()
    radius: int = field(init=False)

    def __post_init__(self):
        self.radius = 0  # updated frame-by-frame


class ChainReactionGame:
    # ───────────── Initialisation ───────────── #
    def __init__(self):
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        except pygame.error as e:
            print("Pygame display init failed:", e, file=sys.stderr)
            sys.exit(1)

        pygame.display.set_caption("Chain Reaction 6×9 – Press R to restart")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("Arial", 28, bold=True)

        self.board: List[List[Cell]] = [[Cell() for _ in range(COLS)] for _ in range(ROWS)]
        self.current_player: int = 1
        self.game_over: bool = False
        self.animations: List[ExplosionAnim] = []
        self.turns_played = 0

    # ───────────── Core mechanics ───────────── #
    def reset(self):
        for row in self.board:
            for cell in row:
                cell.reset()
        self.current_player = 1
        self.game_over = False
        self.animations.clear()
        self.turns_played = 0

    def neighbours(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Orthogonal neighbours inside grid."""
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                yield nr, nc

    def handle_click(self, mx: int, my: int):
        if self.game_over or self.animations:
            return
        r, c = my // CELL_SIZE, mx // CELL_SIZE
        cell = self.board[r][c]
        if cell.owner not in (None, self.current_player):
            return   
        try:
            self.place_orb(r, c, self.current_player)
        except (IndexError, ValueError):
            return  # invalid move
        self.turns_played += 1
        
        if self.turns_played >= 2 and self.check_victory():
            self.game_over = True
        else:
            # only swap turns if at least one cell is occupied (prevents infinite loop on empty board)
            if any(any(cell.owner for cell in row) for row in self.board):
                self.current_player = 3 - self.current_player  # switch player

    def place_orb(self, r: int, c: int, player: int):
        """Place an orb and resolve chain reactions."""
        q = deque([(r, c, player)])
        while q:
            cr, cc, p = q.popleft()
            cell = self.board[cr][cc]
            exploding_before = cell.count == critical_mass(cr, cc) - 1
            cell.add_orb(p)
            if exploding_before and cell.count >= critical_mass(cr, cc):
                # Record animation
                self.animations.append(ExplosionAnim(cr, cc, pygame.time.get_ticks()))
                # Explosion: distribute orbs
                cell.reset()
                for nr, nc in self.neighbours(cr, cc):
                    q.append((nr, nc, p))

    # ───────────── Victory check ───────────── #
    def check_victory(self) -> bool:
        owners = {cell.owner for row in self.board for cell in row if cell.owner}
        return len(owners) == 1 and owners.pop() == self.current_player

    # ───────────── Drawing helpers ───────────── #
    def draw_board(self):
        self.screen.fill(BLACK)
        # Grid lines
        for r in range(1, ROWS):
            pygame.draw.line(self.screen, GRAY, (0, r * CELL_SIZE), (WIDTH, r * CELL_SIZE), 2)
        for c in range(1, COLS):
            pygame.draw.line(self.screen, GRAY, (c * CELL_SIZE, 0), (c * CELL_SIZE, HEIGHT), 2)

        # Cells
        offset = CELL_SIZE * 0.25
        rad = CELL_SIZE * 0.18
        for r, row in enumerate(self.board):
            for c, cell in enumerate(row):
                if not cell.owner:
                    continue
                color = PLAYER_COLOR[cell.owner]
                cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
                positions = []
                if cell.count == 1:
                    positions = [(cx, cy)]
                elif cell.count == 2:
                    positions = [(cx - offset, cy), (cx + offset, cy)]
                elif cell.count >= 3:
                    positions = [(cx - offset, cy - offset),
                                 (cx + offset, cy - offset),
                                 (cx, cy + offset)]
                    if cell.count == 4:
                        positions.append((cx, cy))
                for px, py in positions[:cell.count]:
                    pygame.draw.circle(self.screen, color, (int(px), int(py)), int(rad))

        # Explosion animations (expanding grey circle)
        now = pygame.time.get_ticks()
        for anim in list(self.animations):
            elapsed = now - anim.start_time
            if elapsed >= EXPLOSION_MS:
                self.animations.remove(anim)
                continue
            progress = elapsed / EXPLOSION_MS
            anim.radius = int(progress * CELL_SIZE * 0.5)
            center = (anim.col * CELL_SIZE + CELL_SIZE // 2,
                      anim.row * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, WHITE, center, anim.radius, width=2)

        # UI: current player / winner
        if self.game_over:
            txt = f"Player {self.current_player} ({'Red' if self.current_player==1 else 'Blue'}) wins! – R to restart"
            surf = self.font.render(txt, True, PLAYER_COLOR[self.current_player])
        else:
            txt = f"Turn: Player {self.current_player}  ({'Red' if self.current_player==1 else 'Blue'}) – R to restart"
            surf = self.font.render(txt, True, PLAYER_COLOR[self.current_player])
        self.screen.blit(surf, (10, HEIGHT - 40))

    # ───────────── Main loop ───────────── #
    def run(self):
        while True:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    case pygame.KEYDOWN if event.key == pygame.K_r:
                        self.reset()
                    case pygame.MOUSEBUTTONDOWN if event.button == 1:
                        self.handle_click(*event.pos)

            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)


# ────────────────────────────── Tests ────────────────────────────── #
def _self_tests():
    """Light sanity checks to catch obvious logic bugs quickly."""
    # Corners
    assert critical_mass(0, 0) == 2
    assert critical_mass(ROWS-1, COLS-1) == 2
    # Edges
    assert critical_mass(0, 4) == 3
    assert critical_mass(3, COLS-1) == 3
    # Inner
    assert critical_mass(3, 4) == 4
    print("Critical-mass tests passed.")


# ────────────────────────────── Entrypoint ────────────────────────────── #
if __name__ == "__main__":
    _self_tests()
    ChainReactionGame().run()
