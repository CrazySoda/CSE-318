from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ────────────────────────── Config / Constants ───────────────────────── #
ROWS, COLS   = 6, 9
RED          = (220,  20,  60)
BLUE         = ( 30, 144, 255)
PLAYER_COLOR = {1: RED, 2: BLUE}


# ────────────────────────────── Helpers ──────────────────────────────── #
def critical_mass(r: int, c: int) -> int:
    """Return the maximum orbs a cell (r,c) can hold before exploding."""
    if (r in (0, ROWS - 1)) and (c in (0, COLS - 1)):
        return 2
    if r in (0, ROWS - 1) or c in (0, COLS - 1):
        return 3
    return 4


# ───────────────────────────── Data classes ──────────────────────────── #
@dataclass
class Cell:
    owner: Optional[int] = None   
    count: int = 0                

    def add_orb(self, player: int) -> None:
        """Adds an orb and captures / keeps this cell for *player*."""
        self.owner = player
        self.count += 1

    def reset(self) -> None:
        self.owner = None
        self.count = 0


# ───────────────────────────── Game engine ───────────────────────────── #
class GameState:
    """
    Pure game-state container.  No rendering, no Pygame.
    Public API
    ----------
    • reset()
    • apply_move(row, col) -> list[(row, col)]  # exploded cells this turn
    • get_winner() -> Optional[int]
    • board[r][c] gives the Cell at that position
    """
    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows, self.cols = rows, cols
        self.board: List[List[Cell]] = [[Cell() for _ in range(cols)]
                                        for _ in range(rows)]
        self.current_player: int = 1
        self.turns_played: int   = 0
        self.game_over: bool     = False

    # ─────────────── Core public methods ─────────────── #
    def reset(self) -> None:
        for row in self.board:
            for cell in row:
                cell.reset()
        self.current_player = 1
        self.turns_played   = 0
        self.game_over      = False

    def apply_move(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Perform a click at (r,c) for the current player.
        Returns a list of (row,col) coordinates that exploded this turn
        (frontend can animate them). Raises ValueError on illegal moves.
        """
        if self.game_over:
            raise ValueError("Game already finished")

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError("Move off board")

        target = self.board[r][c]
        if target.owner not in (None, self.current_player):
            raise ValueError("Cannot click opponent's cell")

        explosions: List[Tuple[int, int]] = []
        self._place_orb(r, c, self.current_player, explosions)

        # Switch player & check victory
        self.turns_played += 1
        if self.turns_played >= 2 and self._only_one_owner_left():
            self.game_over = True
        elif any(any(cell.owner for cell in row) for row in self.board):
            self.current_player = 3 - self.current_player

        return explosions

    def get_winner(self) -> Optional[int]:
        """Return 1 or 2 if someone has won, else None."""
        return self.current_player if self.game_over else None

    # ─────────────── Internal helpers ─────────────── #
    def _neighbours(self, r: int, c: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield nr, nc

    def _place_orb(self, r: int, c: int, player: int,
                   explosions: List[Tuple[int, int]]) -> None:
        """BFS queue that performs placement & chain reactions."""
        q = deque([(r, c, player)])
        while q:
            cr, cc, p = q.popleft()
            cell = self.board[cr][cc]
            will_explode = cell.count == critical_mass(cr, cc) - 1
            cell.add_orb(p)
            if will_explode and cell.count >= critical_mass(cr, cc):
                explosions.append((cr, cc))
                cell.reset()
                for nr, nc in self._neighbours(cr, cc):
                    q.append((nr, nc, p))

    def _only_one_owner_left(self) -> bool:
        owners = {cell.owner for row in self.board for cell in row if cell.owner}
        return len(owners) == 1
