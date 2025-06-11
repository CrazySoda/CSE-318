"""
ai.py – Minimax + alpha-beta agent for Chain Reaction 6×9
Depends only on core.py (no Pygame).
"""

from __future__ import annotations
import math
import copy
import core                       # backend game logic

# ───────────────────────────── Heuristic ───────────────────────────── #
def evaluate(state: core.GameState, me: int) -> float:
    """
    +∞  if I have already won
    -∞  if opponent has already won
    Otherwise: weighted material & mobility balance.
    """
    winner = state.get_winner()
    if winner == me:
        return math.inf
    if winner and winner != me:
        return -math.inf

    my_orbs = opp_orbs = 0
    my_cells = opp_cells = 0
    for row in state.board:
        for cell in row:
            if cell.owner == me:
                my_orbs += cell.count
                my_cells += 1
            elif cell.owner == 3 - me:
                opp_orbs += cell.count
                opp_cells += 1

    # simple linear combination – tweak weights as you like
    return (my_orbs - opp_orbs) * 5 + (my_cells - opp_cells) * 2


# ─────────────────────────── Minimax agent ─────────────────────────── #
class MinimaxAgent:
    def __init__(self, player: int, depth: int = 3):
        self.player = player          # 1 = Red, 2 = Blue
        self.depth  = depth

    # public ----------------------------------------------------------- #
    def choose_move(self, state: core.GameState) -> tuple[int, int]:
        """Return (row, col) of best move for self.player."""
        _, move = self._alphabeta(copy.deepcopy(state),
                                  depth=self.depth,
                                  alpha=-math.inf,
                                  beta= math.inf,
                                  maximizing=True)
        if move is None:
            # Fallback: pick the first legal move or raise an error
            legal_moves = state.generate_moves(self.player)
            if legal_moves:
                return legal_moves[0]
            else:
                raise RuntimeError("No legal moves available for player.")
        return move

    # internal --------------------------------------------------------- #
    def _alphabeta(self, state: core.GameState, depth: int,
                   alpha: float, beta: float, maximizing: bool):
        current_player = state.current_player
        if depth == 0 or state.game_over:
            return evaluate(state, self.player), None

        best_move = None
        legal = state.generate_moves(current_player)
        if not legal:                                # no legal moves?
            return evaluate(state, self.player), None

        if maximizing:
            value = -math.inf
            for r, c in legal:
                child = copy.deepcopy(state)
                child.apply_move(r, c)               # mutates child
                score, _ = self._alphabeta(child, depth - 1,
                                            alpha, beta, False)
                if score > value:
                    value, best_move = score, (r, c)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break                            # β-cutoff
            return value, best_move
        else:                                        # minimizing
            value = math.inf
            for r, c in legal:
                child = copy.deepcopy(state)
                child.apply_move(r, c)
                score, _ = self._alphabeta(child, depth - 1,
                                            alpha, beta, True)
                if score < value:
                    value, best_move = score, (r, c)
                beta = min(beta, value)
                if beta <= alpha:
                    break                            # α-cutoff
            return value, best_move
