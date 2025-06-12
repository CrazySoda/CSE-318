"""
ai.py – Configurable Minimax agent for Chain Reaction 6×9
Allows selection of heuristics and easy fine-tuning.
Depends only on core.py (no Pygame).
"""

from __future__ import annotations
import math
import copy
import core
from typing import Tuple, Optional, Dict, Any, List

# ═══════════════════════════════════════════════════════════════════════════════
# TUNING SECTION - Modify these values to experiment with your AI
# ═══════════════════════════════════════════════════════════════════════════════

# Default heuristic weights - TUNE THESE VALUES
DEFAULT_WEIGHTS = {
    'material': 3.0,        # Higher = prioritize having more orbs
    'territorial': 2.0,     # Higher = prioritize controlling more cells
    'critical_mass': 4.0,   # Higher = prioritize immediate threats/opportunities
    'mobility': 1.5,        # Higher = prioritize having more move options
    'chain_potential': 2.5, # Higher = prioritize chain reaction setups
    'positional': 1.0       # Higher = prioritize good board positions
}

# Search depth settings - TUNE THESE VALUES
DEFAULT_DEPTH = 3           # Increase for stronger play (but slower)
MAX_DEPTH = 6              # Maximum allowed depth
MIN_DEPTH = 1              # Minimum allowed depth

# Performance settings - TUNE THESE VALUES
USE_TRANSPOSITION_TABLE = True  # Set to False to disable caching
MAX_TABLE_SIZE = 5000          # Increase for more caching (uses more memory)
USE_MOVE_ORDERING = True       # Set to False to disable move ordering

# ═══════════════════════════════════════════════════════════════════════════════

class Heuristics:
    """Collection of heuristic evaluation functions."""
    
    @staticmethod
    def material_advantage(state: core.GameState, player: int) -> float:
        """Count orbs with proximity-to-explosion bonus."""
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
        """Count controlled cells with positional weighting."""
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
        """Evaluate immediate explosion threats and opportunities."""
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
        """Evaluate move options and tactical flexibility."""
        my_moves = len(state.generate_moves(player))
        opp_moves = len(state.generate_moves(3 - player))
        
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
        """Evaluate potential for chain reactions."""
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
        """Evaluate board position quality."""
        my_positional = opp_positional = 0
        opponent = 3 - player
        center_r, center_c = state.rows // 2, state.cols // 2
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    # Distance from center
                    center_distance = abs(r - center_r) + abs(c - center_c)
                    center_bonus = max(0, 5 - center_distance) * 0.5
                    
                    # Cluster bonus
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


class AIConfig:
    """Configuration class for AI settings."""
    
    def __init__(self):
        self.enabled_heuristics = {
            'material': True,
            'territorial': True,
            'critical_mass': True,
            'mobility': True,
            'chain_potential': True,
            'positional': True
        }
        self.weights = DEFAULT_WEIGHTS.copy()
        self.depth = DEFAULT_DEPTH
        self.use_transposition_table = USE_TRANSPOSITION_TABLE
        self.use_move_ordering = USE_MOVE_ORDERING
    
    def disable_heuristic(self, heuristic_name: str):
        """Disable a specific heuristic."""
        if heuristic_name in self.enabled_heuristics:
            self.enabled_heuristics[heuristic_name] = False
    
    def enable_heuristic(self, heuristic_name: str):
        """Enable a specific heuristic."""
        if heuristic_name in self.enabled_heuristics:
            self.enabled_heuristics[heuristic_name] = True
    
    def set_weight(self, heuristic_name: str, weight: float):
        """Set weight for a specific heuristic."""
        if heuristic_name in self.weights:
            self.weights[heuristic_name] = weight
    
    def set_depth(self, depth: int):
        """Set search depth."""
        self.depth = max(MIN_DEPTH, min(MAX_DEPTH, depth))
    
    def get_active_weights(self) -> Dict[str, float]:
        """Get weights for only enabled heuristics."""
        return {name: weight for name, weight in self.weights.items() 
                if self.enabled_heuristics.get(name, False)}


class MinimaxAgent:
    """Configurable Minimax agent with selectable heuristics."""
    
    def __init__(self, player: int, config: Optional[AIConfig] = None):
        """
        Initialize the minimax agent.
        
        Args:
            player: Player number (1 for Red, 2 for Blue)
            config: AI configuration object (uses defaults if None)
        """
        self.player = player
        self.config = config or AIConfig()
        self.heuristics = Heuristics()
        
        # Performance optimizations
        self.transposition_table: Dict[str, Tuple[float, int, Optional[Tuple[int, int]]]] = {}
        self.move_history: Dict[Tuple[int, int], int] = {}
        
        # Statistics
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
        self.table_hits = 0
    
    def evaluate_state(self, state: core.GameState) -> float:
        """Evaluate game state using enabled heuristics."""
        # Check terminal states first
        winner = state.get_winner()
        if winner == self.player:
            return math.inf
        elif winner and winner != self.player:
            return -math.inf
        
        # Calculate enabled heuristics
        total_score = 0.0
        active_weights = self.config.get_active_weights()
        
        if active_weights.get('material', 0) > 0:
            total_score += active_weights['material'] * self.heuristics.material_advantage(state, self.player)
        
        if active_weights.get('territorial', 0) > 0:
            total_score += active_weights['territorial'] * self.heuristics.territorial_control(state, self.player)
        
        if active_weights.get('critical_mass', 0) > 0:
            total_score += active_weights['critical_mass'] * self.heuristics.critical_mass_proximity(state, self.player)
        
        if active_weights.get('mobility', 0) > 0:
            total_score += active_weights['mobility'] * self.heuristics.mobility_freedom(state, self.player)
        
        if active_weights.get('chain_potential', 0) > 0:
            total_score += active_weights['chain_potential'] * self.heuristics.chain_reaction_potential(state, self.player)
        
        if active_weights.get('positional', 0) > 0:
            total_score += active_weights['positional'] * self.heuristics.positional_advantage(state, self.player)
        
        return total_score
    
    def _board_hash(self, state: core.GameState) -> str:
        """Create hash for transposition table."""
        if not self.config.use_transposition_table:
            return ""
        
        parts = []
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner is None:
                    parts.append('0')
                else:
                    parts.append(f"{cell.count}{cell.owner}")
        return ''.join(parts) + f"_{state.current_player}"
    
    def _order_moves(self, moves: List[Tuple[int, int]], state: core.GameState) -> List[Tuple[int, int]]:
        """Order moves for better alpha-beta pruning."""
        if not self.config.use_move_ordering:
            return moves
        
        def move_score(move):
            r, c = move
            score = self.move_history.get(move, 0)
            
            # Prefer corners and edges
            crit_mass = core.critical_mass(state.rows, state.cols, r, c)
            if crit_mass == 2:      # Corner
                score += 5
            elif crit_mass == 3:    # Edge
                score += 3
            
            # Prefer empty cells
            if state.board[r][c].owner is None:
                score += 2
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def minimax_search(self, state: core.GameState, depth_limit: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Main minimax search with alpha-beta pruning."""
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
        self.table_hits = 0
        
        # Clear table if too large
        if len(self.transposition_table) > MAX_TABLE_SIZE:
            self.transposition_table.clear()
        
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
        """Alpha-beta pruning implementation."""
        self.nodes_explored += 1
        
        # Terminal conditions
        if depth == 0 or state.game_over:
            return self.evaluate_state(state), None
        
        # Transposition table lookup
        state_hash = self._board_hash(state)
        if state_hash and state_hash in self.transposition_table:
            cached_value, cached_depth, cached_move = self.transposition_table[state_hash]
            if cached_depth >= depth:
                self.table_hits += 1
                return cached_value, cached_move
        
        current_player = state.current_player
        legal_moves = state.generate_moves(current_player)
        
        if not legal_moves:
            return self.evaluate_state(state), None
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(legal_moves, state)
        best_action = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in ordered_moves:
                r, c = move
                child_state = copy.deepcopy(state)
                child_state.apply_move(r, c)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = move
                    if self.config.use_move_ordering:
                        self.move_history[move] = self.move_history.get(move, 0) + 1
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    if self.config.use_move_ordering:
                        self.move_history[move] = self.move_history.get(move, 0) + 2
                    break
            
            # Store in transposition table
            if state_hash and len(self.transposition_table) < MAX_TABLE_SIZE:
                self.transposition_table[state_hash] = (max_eval, depth, best_action)
            
            return max_eval, best_action
        
        else:  # Minimizing player
            min_eval = math.inf
            for move in ordered_moves:
                r, c = move
                child_state = copy.deepcopy(state)
                child_state.apply_move(r, c)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = move
                    if self.config.use_move_ordering:
                        self.move_history[move] = self.move_history.get(move, 0) + 1
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    if self.config.use_move_ordering:
                        self.move_history[move] = self.move_history.get(move, 0) + 2
                    break
            
            # Store in transposition table
            if state_hash and len(self.transposition_table) < MAX_TABLE_SIZE:
                self.transposition_table[state_hash] = (min_eval, depth, best_action)
            
            return min_eval, best_action
    
    def choose_move(self, state: core.GameState) -> Tuple[int, int]:
        """Choose the best move."""
        value, move = self.minimax_search(state, self.config.depth)
        
        if move is None:
            legal_moves = state.generate_moves(self.player)
            if legal_moves:
                return legal_moves[0]
            else:
                raise RuntimeError("No legal moves available")
        
        return move
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        total_lookups = self.table_hits + max(1, self.nodes_explored - self.table_hits)
        hit_rate = (self.table_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        return {
            'nodes_explored': self.nodes_explored,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs,
            'search_depth': self.config.depth,
            'table_hits': self.table_hits,
            'hit_rate_percent': hit_rate,
            'enabled_heuristics': [name for name, enabled in self.config.enabled_heuristics.items() if enabled],
            'heuristic_weights': self.config.get_active_weights()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS - Add your own presets here for easy testing
# ═══════════════════════════════════════════════════════════════════════════════

def create_aggressive_config() -> AIConfig:
    """Aggressive AI focusing on immediate threats and chain reactions."""
    config = AIConfig()
    config.weights = {
        'material': 2.0,
        'territorial': 1.0,
        'critical_mass': 6.0,      # Very high
        'mobility': 1.0,
        'chain_potential': 4.0,    # High
        'positional': 0.5
    }
    return config

def create_defensive_config() -> AIConfig:
    """Defensive AI focusing on territory and material advantage."""
    config = AIConfig()
    config.weights = {
        'material': 5.0,           # Very high
        'territorial': 4.0,        # High
        'critical_mass': 2.0,
        'mobility': 2.0,
        'chain_potential': 1.0,
        'positional': 3.0          # High
    }
    return config

def create_balanced_config() -> AIConfig:
    """Balanced AI with default weights."""
    return AIConfig()

def create_material_only_config() -> AIConfig:
    """AI using only material advantage heuristic."""
    config = AIConfig()
    config.enabled_heuristics = {
        'material': True,
        'territorial': False,
        'critical_mass': False,
        'mobility': False,
        'chain_potential': False,
        'positional': False
    }
    config.weights['material'] = 1.0
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(state: core.GameState, me: int) -> float:
    """Simple evaluation for backward compatibility."""
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

    return (my_orbs - opp_orbs) * 5 + (my_cells - opp_cells) * 2