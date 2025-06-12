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
def critical_mass(rows: int, cols: int, r: int, c: int) -> int:
    """
    Corner  → 2   •   Edge  → 3   •   Inner → 4
    Works for any board dimensions.
    """
    if (r in (0, rows - 1)) and (c in (0, cols - 1)):
        return 2
    if r in (0, rows - 1) or c in (0, cols - 1):
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
    ------------------------------------------------------------------------
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
        
        # ─────────────────── File-protocol helpers ─────────────────── #
    @staticmethod
    def from_file(lines: list[str]) -> "GameState":
        """Build a GameState from metadata + 9 lines of 6 tokens each."""
        if len(lines) != 10:  # Changed from 9 to 10
            raise ValueError("File must have metadata + 9 board rows.")
        
        # Parse metadata
        turns_played, game_over_int, current_player = map(int, lines[0].split())
        
        gs = GameState(rows=9, cols=6)
        gs.turns_played = turns_played      # Restore persistent data
        gs.game_over = bool(game_over_int)  # Restore persistent data  
        gs.current_player = current_player  # Restore persistent data
        
        # Parse board (skip first line now)
        for r, line in enumerate(lines[1:]):
            tokens = line.strip().split()
            if len(tokens) != 6:
                raise ValueError("Each row must have 6 cells.")
            for c, tok in enumerate(tokens):
                if tok == "0":
                    continue
                n, col = int(tok[:-1]), tok[-1]
                gs.board[r][c].count = n
                gs.board[r][c].owner = 1 if col == "R" else 2
        return gs

    def to_lines(self) -> list[str]:
        """Return 9 lines of 6 tokens matching the file protocol."""
        out = [f"{self.turns_played} {int(self.game_over)} {self.current_player}"]
        for r in range(9):
            row_tokens = []
            for c in range(6):
                cell = self.board[r][c]
                if cell.owner is None or cell.count == 0:
                    row_tokens.append("0")
                else:
                    row_tokens.append(f"{cell.count}{'R' if cell.owner==1 else 'B'}")
            out.append(" ".join(row_tokens))
        return out


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
        Returns a list of (row,col) coordinates that exploded this turn.
        Raises ValueError on illegal moves.
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

        # Debug state after move
        #print(f"[DEBUG] Turns played (before increment): {self.turns_played}")
        self.turns_played += 1
        #print(f"[DEBUG] Turns played (after increment): {self.turns_played}")

        owners = {cell.owner for row in self.board for cell in row if cell.owner}
        #print(f"[DEBUG] Owners after move: {owners}")

        if self.turns_played >= 2 and self._only_one_owner_left():
            #print("[DEBUG] Only one player left. Game Over!")
            self.game_over = True
        elif any(owners):
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

            # ─── NEW: look up the correct critical-mass for *this* board size ───
            crit = critical_mass(self.rows, self.cols, cr, cc)
            will_explode = cell.count == crit - 1
            # ────────────────────────────────────────────────────────────────────

            cell.add_orb(p)
            if will_explode and cell.count >= crit:     
                explosions.append((cr, cc))
                cell.reset()
                for nr, nc in self._neighbours(cr, cc):
                    q.append((nr, nc, p))


    def _only_one_owner_left(self) -> bool:
        owners = {cell.owner for row in self.board for cell in row if cell.owner}
        #print(f"[DEBUG] Owners on board: {owners}")
        return len(owners) == 1
    
        # ─────────────────── Utility for AI ─────────────────── #
    def clone(self) -> "GameState":
        """Deep copy of this state (wrapper around copy.deepcopy)."""
        import copy
        return copy.deepcopy(self)

    def generate_moves(self, player: int) -> list[tuple[int, int]]:
        """Return all legal (row, col) moves for *player* in this state."""
        moves = []
        for r, row in enumerate(self.board):
            for c, cell in enumerate(row):
                if cell.owner in (None, player):
                    moves.append((r, c))
        return moves
