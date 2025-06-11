"""
gui.py – supports TWO modes:

1. PVP   (two humans on one screen, no file I/O, no engine.py needed)
2. PVAI  (Human vs AI process via gamestate.txt, as before)

HOW TO RUN
----------
Terminal 1 (AI backend, only if you want PVAI):
    python engine.py

Terminal 2 (GUI):
    python gui.py
    → Choose mode with 1 or 2
    → Press  M  at any time to toggle modes (board resets)
"""

from __future__ import annotations
import sys, time, os, pygame
from dataclasses import dataclass, field
from typing import List, Tuple
import core

# ──────────────────────────────────────────────────────────────────────
# File-protocol helpers (used only in PVAI mode)
# ──────────────────────────────────────────────────────────────────────
FILE = "gamestate.txt"

def _write_state(header: str, state: core.GameState):
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(state.to_lines()))

def _read_until(header: str) -> core.GameState:
    while True:
        with open(FILE, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lines and lines[0].strip() == header:
            return core.GameState.from_file(lines[1:])
        time.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────
# GUI constants
# ──────────────────────────────────────────────────────────────────────
CELL_SIZE = 100
WIDTH, HEIGHT = 6 * CELL_SIZE, 9 * CELL_SIZE      # 9×6 protocol board
FPS = 60
GRAY  = (200, 200, 200)
BLACK = (25, 25, 25)
WHITE = (250, 250, 250)
EXPLOSION_MS = 250

MODE_PVP  = "PVP"
MODE_PVAI = "PVAI"

@dataclass
class ExplosionAnim:
    row: int
    col: int
    start_time: int
    radius: int = field(init=False, default=0)

# ──────────────────────────────────────────────────────────────────────
# Main GUI class
# ──────────────────────────────────────────────────────────────────────
class ChainReactionGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chain Reaction")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("Arial", 28, bold=True)

        self.state = core.GameState(rows=9, cols=6)
        self.mode  = None                 # chosen in self._menu()
        self.animations: List[ExplosionAnim] = []

    # ─────────────── Mode selection / toggle ───────────────
    def _menu(self):
        """Blocking start screen – user chooses mode."""
        choosing = True
        while choosing:
            self.screen.fill((30, 30, 30))
            title = self.font.render("Choose Game Mode", True, WHITE)
            pvp   = self.font.render("1  –  Two Players", True, GRAY)
            pvai  = self.font.render("2  –  Play vs AI",  True, GRAY)
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3))
            self.screen.blit(pvp,   (WIDTH//2 - pvp.get_width()//2,   HEIGHT//2))
            self.screen.blit(pvai,  (WIDTH//2 - pvai.get_width()//2,  HEIGHT//2 + 40))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.mode = MODE_PVP;  choosing = False
                    elif event.key == pygame.K_2:
                        self.mode = MODE_PVAI; choosing = False

        self._reset_board()

    def _toggle_mode(self):
        self.mode = MODE_PVP if self.mode == MODE_PVAI else MODE_PVAI
        self._reset_board()

    # ─────────────── Game / file helpers ───────────────
    def _reset_board(self):
        self.state.reset()
        self.animations.clear()
        # Ensure the file exists for PVAI so engine.py is happy
        if self.mode == MODE_PVAI and not os.path.exists(FILE):
            _write_state("AI Move:", self.state)

    def _write_human_move(self):
        _write_state("Human Move:", self.state)

    def _wait_ai(self):
        self.state = _read_until("AI Move:")

    # ─────────────── Event handlers ───────────────
    def handle_click(self, mx: int, my: int):
        if self.state.game_over or self.animations: return
        r, c = my // CELL_SIZE, mx // CELL_SIZE

        try:
            exploded = self.state.apply_move(r, c)        # always Red/Blue logic
        except ValueError:
            return

        now = pygame.time.get_ticks()
        for er, ec in exploded:
            self.animations.append(ExplosionAnim(er, ec, now))

        # If PVAI and Red (human) just moved, hand off to AI
        if self.mode == MODE_PVAI and self.state.current_player == 2 and not self.state.game_over:
            self._write_human_move()
            self._wait_ai()                               # blocks until file flips
            # (optional) we could parse AI explosions here for animation

    # ─────────────── Draw helpers ───────────────
    def _draw_grid(self):
        for r in range(1, 9):
            pygame.draw.line(self.screen, GRAY, (0, r*CELL_SIZE), (WIDTH, r*CELL_SIZE), 2)
        for c in range(1, 6):
            pygame.draw.line(self.screen, GRAY, (c*CELL_SIZE, 0), (c*CELL_SIZE, HEIGHT), 2)

    def _draw_orbs(self):
        offset, rad = CELL_SIZE*0.25, CELL_SIZE*0.18
        for r in range(9):
            for c in range(6):
                cell = self.state.board[r][c]
                if not cell.owner: continue
                color = core.PLAYER_COLOR[cell.owner]
                cx, cy = c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2
                positions: List[Tuple[float, float]] = []
                if cell.count == 1: positions.append((cx, cy))
                elif cell.count == 2: positions.extend([(cx-offset, cy), (cx+offset, cy)])
                else:
                    positions.extend([(cx-offset, cy-offset), (cx+offset, cy-offset), (cx, cy+offset)])
                    if cell.count == 4: positions.append((cx, cy))
                for px, py in positions[:cell.count]:
                    pygame.draw.circle(self.screen, color, (int(px), int(py)), int(rad))

    def _draw_ui(self):
        if self.state.game_over:
            win = self.state.get_winner()
            if win:
                txt = f"Player {win} ({'Red' if win==1 else 'Blue'}) wins!"
                color = core.PLAYER_COLOR[win]
            else:
                txt, color = "Game over!", WHITE
        else:
            if self.mode == MODE_PVP:
                turn = 'Red' if self.state.current_player == 1 else 'Blue'
                txt = f"PVP  –  {turn}'s turn   (M to switch mode)"
            else:
                txt = "PVAI  –  Your turn (Red)   (M to switch mode)"
            color = WHITE
        surf = self.font.render(txt, True, color)
        self.screen.blit(surf, (10, HEIGHT - 40))

    # ─────────────── Main loop ───────────────
    def run(self):
        self._menu()                                      # choose mode first
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self._toggle_mode()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            # animations update (simple radial flash)
            now = pygame.time.get_ticks()
            self.animations[:] = [anim for anim in self.animations
                                   if now - anim.start_time < EXPLOSION_MS]

            self.screen.fill(BLACK)
            self._draw_grid()
            self._draw_orbs()
            self._draw_ui()
            pygame.display.flip()
            self.clock.tick(FPS)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ChainReactionGUI().run()
