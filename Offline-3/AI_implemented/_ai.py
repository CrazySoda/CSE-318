"""
ai.py – Enhanced Minimax + alpha-beta agent for Chain Reaction 6×9
Includes multiple domain-inspired heuristic functions and comprehensive evaluation.
Depends only on core.py (no Pygame).
"""

from __future__ import annotations
import math
import copy
import core
from typing import Tuple, Optional, Dict, Any

# ───────────────────────────── Enhanced Heuristics ───────────────────────────── #

class ChainReactionHeuristics:
    """Collection of domain-inspired heuristic evaluation functions."""
    
    @staticmethod
    def material_advantage(state: core.GameState, player: int) -> float:
        """
        Heuristic 1: Material Advantage
        Rationale: Having more orbs gives better position and more options.
        Weights orbs by their strategic value (near-critical orbs are more valuable).
        """
        my_score = opp_score = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner == player:
                    crit_mass = core.critical_mass(state.rows, state.cols, r, c)
                    proximity_bonus = cell.count / crit_mass
                    my_score += cell.count * (1 + proximity_bonus)
                elif cell.owner == opponent:
                    crit_mass = core.critical_mass(state.rows, state.cols, r, c)
                    proximity_bonus = cell.count / crit_mass
                    opp_score += cell.count * (1 + proximity_bonus)
        
        return my_score - opp_score
    
    @staticmethod
    def territorial_control(state: core.GameState, player: int) -> float:
        """
        Heuristic 2: Territorial Control
        Rationale: Controlling more cells provides strategic advantage and more move options.
        Corner and edge cells are weighted higher due to lower critical mass.
        """
        my_cells = opp_cells = 0
        my_weighted = opp_weighted = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                crit_mass = core.critical_mass(state.rows, state.cols, r, c)
                cell_weight = 5 - crit_mass  # corners=3, edges=2, center=1
                
                if cell.owner == player:
                    my_cells += 1
                    my_weighted += cell_weight
                elif cell.owner == opponent:
                    opp_cells += 1
                    opp_weighted += cell_weight
        
        return (my_weighted - opp_weighted) * 2 + (my_cells - opp_cells)
    
    @staticmethod
    def critical_mass_proximity(state: core.GameState, player: int) -> float:
        """
        Heuristic 3: Critical Mass Proximity
        Rationale: Cells close to exploding create immediate threats and opportunities.
        Higher weight for cells that can explode next turn.
        """
        my_threats = opp_threats = 0
        my_near_critical = opp_near_critical = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    crit_mass = core.critical_mass(state.rows, state.cols, r, c)
                    proximity = cell.count / crit_mass
                    
                    if cell.owner == player:
                        if cell.count == crit_mass - 1:
                            my_threats += 10  # About to explode
                        my_near_critical += proximity * 5
                    else:
                        if cell.count == crit_mass - 1:
                            opp_threats += 10
                        opp_near_critical += proximity * 5
        
        return (my_threats - opp_threats) + (my_near_critical - opp_near_critical)
    
    @staticmethod
    def mobility_freedom(state: core.GameState, player: int) -> float:
        """
        Heuristic 4: Mobility and Freedom
        Rationale: Having more legal moves provides tactical flexibility.
        Also considers potential future moves after explosions.
        """
        my_moves = len(state.generate_moves(player))
        opp_moves = len(state.generate_moves(3 - player))
        
        # Weight moves by their potential impact
        my_weighted_moves = opp_weighted_moves = 0
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [None, player]:
                    move_value = 2 if cell.owner is None else 1
                    my_weighted_moves += move_value
                if cell.owner in [None, 3 - player]:
                    move_value = 2 if cell.owner is None else 1
                    opp_weighted_moves += move_value
        
        return (my_weighted_moves - opp_weighted_moves) * 0.5 + (my_moves - opp_moves)
    
    @staticmethod
    def chain_reaction_potential(state: core.GameState, player: int) -> float:
        """
        Heuristic 5: Chain Reaction Potential
        Rationale: Positions that can trigger large chain reactions are highly valuable.
        Evaluates potential cascade effects from current board positions.
        """
        my_potential = opp_potential = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    crit_mass = core.critical_mass(state.rows, state.cols, r, c)
                    
                    if cell.count >= crit_mass - 1:  # Close to exploding
                        neighbor_count = 0
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < state.rows and 0 <= nc < state.cols:
                                neighbor = state.board[nr][nc]
                                neighbor_crit = core.critical_mass(state.rows, state.cols, nr, nc)
                                if neighbor.count >= neighbor_crit - 2:
                                    neighbor_count += 2
                                else:
                                    neighbor_count += 1
                        
                        chain_value = neighbor_count * cell.count
                        if cell.owner == player:
                            my_potential += chain_value
                        else:
                            opp_potential += chain_value
        
        return my_potential - opp_potential
    
    @staticmethod
    def positional_advantage(state: core.GameState, player: int) -> float:
        """
        Heuristic 6: Positional Advantage
        Rationale: Some board positions are inherently more valuable.
        Center control and formation of clusters are important.
        """
        my_positional = opp_positional = 0
        opponent = 3 - player
        center_r, center_c = state.rows // 2, state.cols // 2
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    # Distance from center (closer is better for some strategies)
                    center_distance = abs(r - center_r) + abs(c - center_c)
                    center_bonus = max(0, 5 - center_distance) * 0.5
                    
                    # Cluster bonus - reward adjacent friendly cells
                    cluster_bonus = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < state.rows and 0 <= nc < state.cols and 
                            state.board[nr][nc].owner == cell.owner):
                            cluster_bonus += 1
                    
                    total_positional = center_bonus + cluster_bonus
                    if cell.owner == player:
                        my_positional += total_positional
                    else:
                        opp_positional += total_positional
        
        return my_positional - opp_positional


# ─────────────────────────── Enhanced Minimax Agent ─────────────────────────── #

class MinimaxAgent:
    """Enhanced Minimax agent with multiple heuristics and comprehensive evaluation."""
    
    def __init__(self, player: int, depth: int = 4, heuristic_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the enhanced minimax agent.
        
        Args:
            player: Player number (1 for Red, 2 for Blue)
            depth: Search depth limit
            heuristic_weights: Weights for combining different heuristics
        """
        self.player = player
        self.depth = depth
        self.heuristics = ChainReactionHeuristics()
        
        # Default heuristic weights (can be tuned)
        self.weights = heuristic_weights or {
            'material': 3.0,
            'territorial': 2.0,
            'critical_mass': 4.0,
            'mobility': 1.5,
            'chain_potential': 2.5,
            'positional': 1.0
        }
        
        # Statistics for analysis
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
    
    def evaluate_state(self, state: core.GameState) -> float:
        """
        Comprehensive state evaluation using all heuristic functions.
        
        Returns:
            Evaluation score from the perspective of self.player
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner == self.player:
            return math.inf
        elif winner and winner != self.player:
            return -math.inf
        
        # Combine all heuristics with their weights
        total_score = 0.0
        
        total_score += self.weights['material'] * self.heuristics.material_advantage(state, self.player)
        total_score += self.weights['territorial'] * self.heuristics.territorial_control(state, self.player)
        total_score += self.weights['critical_mass'] * self.heuristics.critical_mass_proximity(state, self.player)
        total_score += self.weights['mobility'] * self.heuristics.mobility_freedom(state, self.player)
        total_score += self.weights['chain_potential'] * self.heuristics.chain_reaction_potential(state, self.player)
        total_score += self.weights['positional'] * self.heuristics.positional_advantage(state, self.player)
        
        return total_score
    
    def minimax_search(self, state: core.GameState, depth_limit: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Main minimax search function with alpha-beta pruning.
        
        Args:
            state: Current game state
            depth_limit: Maximum search depth
            
        Returns:
            Tuple of (evaluation_value, best_action)
        """
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
        
        value, action = self._alpha_beta(
            state=copy.deepcopy(state),
            depth=depth_limit,
            alpha=-math.inf,
            beta=math.inf,
            maximizing_player=True
        )
        
        return value, action
    
    def _alpha_beta(self, state: core.GameState, depth: int, alpha: float, beta: float, 
                   maximizing_player: bool) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Alpha-beta pruning implementation.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: True if maximizing, False if minimizing
            
        Returns:
            Tuple of (evaluation_value, best_action)
        """
        self.nodes_explored += 1
        
        # Terminal conditions
        if depth == 0 or state.game_over:
            return self.evaluate_state(state), None
        
        current_player = state.current_player
        legal_moves = state.generate_moves(current_player)
        
        if not legal_moves:
            return self.evaluate_state(state), None
        
        best_action = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in legal_moves:
                r, c = move
                child_state = copy.deepcopy(state)
                child_state.apply_move(r, c)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    break  # Beta cutoff
            
            return max_eval, best_action
        
        else:  # Minimizing player
            min_eval = math.inf
            for move in legal_moves:
                r, c = move
                child_state = copy.deepcopy(state)
                child_state.apply_move(r, c)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    break  # Alpha cutoff
            
            return min_eval, best_action
    
    def choose_move(self, state: core.GameState) -> Tuple[int, int]:
        """
        Public interface to choose the best move.
        
        Args:
            state: Current game state
            
        Returns:
            Best move as (row, col) tuple
        """
        value, move = self.minimax_search(state, self.depth)
        
        if move is None:
            # Fallback to first legal move
            legal_moves = state.generate_moves(self.player)
            if legal_moves:
                return legal_moves[0]
            else:
                raise RuntimeError("No legal moves available")
        
        return move
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Return statistics about the last search performed."""
        return {
            'nodes_explored': self.nodes_explored,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs,
            'search_depth': self.depth,
            'heuristic_weights': self.weights.copy()
        }


# ─────────────────────────── Original Simple Heuristic ─────────────────────────── #

def evaluate(state: core.GameState, me: int) -> float:
    """
    Original simple heuristic for backward compatibility.
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


# ─────────────────────────── Backward Compatible Agent ─────────────────────────── #

class MinimaxAgent:
    """Backward compatible wrapper using the enhanced agent with simpler weights."""
    
    def __init__(self, player: int, depth: int = 3):
        # Use simpler weights for backward compatibility with original behavior
        simple_weights = {
            'material': 5.0,
            'territorial': 2.0,
            'critical_mass': 1.0,
            'mobility': 0.5,
            'chain_potential': 0.5,
            'positional': 0.1
        }
        self.agent = MinimaxAgent(player, depth, simple_weights)
        self.player = player
        self.depth = depth

    def choose_move(self, state: core.GameState) -> tuple[int, int]:
        """Return (row, col) of best move for self.player."""
        return self.agent.choose_move(state)
    
    def _alphabeta(self, state: core.GameState, depth: int,
                   alpha: float, beta: float, maximizing: bool):
        """Legacy method - delegates to enhanced agent."""
        return self.agent._alpha_beta(state, depth, alpha, beta, maximizing)