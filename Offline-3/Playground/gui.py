from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import pygame

import core  

# ────────────────────────────── UI constants ─────────────────────────── #
CELL_SIZE = 100
WIDTH, HEIGHT = core.COLS * CELL_SIZE, core.ROWS * CELL_SIZE
FPS = 60
GRAY = (200, 200, 200)
BLACK = (25, 25, 25)
WHITE = (250, 250, 250)
EXPLOSION_MS = 250     # animation length in milliseconds


# ────────────────────── Explosion animation helper ───────────────────── #
@dataclass
class ExplosionAnim:
    row: int
    col: int
    start_time: int                 
    radius: int = field(init=False)

    def __post_init__(self):
        self.radius = 0


# ───────────────────────────── Main GUI class ────────────────────────── #
class ChainReactionGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chain Reaction 6×9 – Press R to restart")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("Arial", 28, bold=True)

        self.state = core.GameState()         # ← backend instance
        self.animations: List[ExplosionAnim] = []

    # ──────────────── Event handlers ──────────────── #
    def reset(self):
        self.state.reset()
        self.animations.clear()

    def handle_click(self, mx: int, my: int):
        if self.state.game_over or self.animations:
            return
        r, c = my // CELL_SIZE, mx // CELL_SIZE
        try:
            exploded = self.state.apply_move(r, c)
        except ValueError:
            return  # illegal click: silently ignore

        # enqueue explosions for animation
        now = pygame.time.get_ticks()
        for er, ec in exploded:
            self.animations.append(ExplosionAnim(er, ec, now))

    # ──────────────── Drawing helpers ─────────────── #
    def _draw_grid(self):
        for r in range(1, core.ROWS):
            pygame.draw.line(self.screen, GRAY, (0, r * CELL_SIZE),
                             (WIDTH, r * CELL_SIZE), 2)
        for c in range(1, core.COLS):
            pygame.draw.line(self.screen, GRAY, (c * CELL_SIZE, 0),
                             (c * CELL_SIZE, HEIGHT), 2)

    def _draw_orbs(self):
        offset = CELL_SIZE * 0.25
        rad    = CELL_SIZE * 0.18
        for r, row in enumerate(self.state.board):
            for c, cell in enumerate(row):
                if not cell.owner:
                    continue
                color = core.PLAYER_COLOR[cell.owner]
                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = r * CELL_SIZE + CELL_SIZE // 2

                positions: List[Tuple[float, float]] = []
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
                    pygame.draw.circle(self.screen, color,
                                       (int(px), int(py)), int(rad))

    def _draw_animations(self):
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
            pygame.draw.circle(self.screen, WHITE, center,
                               anim.radius, width=2)

    def _draw_ui(self):
        if self.state.game_over:
            winner = self.state.get_winner()
            if winner is not None:
                txt = f"Player {winner} ({'Red' if winner==1 else 'Blue'}) wins! – R to restart"
                surf = self.font.render(txt, True, core.PLAYER_COLOR[winner])
            else:
                txt = "Game Over – R to restart"
                surf = self.font.render(txt, True, WHITE)
        else:
            p = self.state.current_player
            txt = f"Turn: Player {p}  ({'Red' if p==1 else 'Blue'}) – R to restart"
            surf = self.font.render(txt, True, core.PLAYER_COLOR[p])
        self.screen.blit(surf, (10, HEIGHT - 40))

    # ──────────────── Main loop ─────────────── #
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            self.screen.fill(BLACK)
            self._draw_grid()
            self._draw_orbs()
            self._draw_animations()
            self._draw_ui()

            pygame.display.flip()
            self.clock.tick(FPS)


# ───────────────────────── Entrypoint ───────────────────────── #
if __name__ == "__main__":
    ChainReactionGUI().run()
