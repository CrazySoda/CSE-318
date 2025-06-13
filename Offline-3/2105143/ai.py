from __future__ import annotations
import math
import time
import core
from typing import Tuple, Optional, Dict, Any, List
from collections import defaultdict

# Default heuristic weights
DEFAULT_WEIGHTS = {
    'material': 3.0,        
    'territorial': 2.0,     
    'critical_mass': 4.0,   
    'mobility': 1.5,        
    'chain_potential': 2.5, 
    'positional': 1.0       
}

# Search depth settings
DEFAULT_DEPTH = 3           # Increased default depth due to optimizations
MAX_DEPTH = 6              # Increased maximum depth
MIN_DEPTH = 1              # Minimum allowed depth

# Performance settings
USE_TRANSPOSITION_TABLE = True  # Set to False to disable caching
MAX_TABLE_SIZE = 10000         # Increased for better caching
CACHE_CLEANUP_THRESHOLD = 8000  # When to start cleaning cache
USE_MOVE_ORDERING = True       # Set to False to disable move ordering
USE_ASPIRATION_WINDOWS = True  # Set to False to disable aspiration windows

# Explosion limit settings 
DEFAULT_EXPLOSION_LIMIT = 1000   # Maximum explosions per minimax search
EXPLOSION_LIMIT_ENABLED = True # Enable/disable explosion limiting

# Timeout settings
DEFAULT_TIMEOUT = 5.0          # Default timeout in seconds
MIN_TIMEOUT = 2.0              # Minimum timeout

# ═══════════════════════════════════════════════════════════════════════════════

class Heuristics:
    """Optimized heuristic evaluation functions with caching."""
    
    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_cache_key(self, state: core.GameState, player: int, heuristic: str) -> str:
        """Fast cache key generation."""
        # Use a more efficient hash based on board state
        board_hash = 0
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner:
                    board_hash = board_hash * 31 + (r * state.cols + c) * 10 + cell.count * 2 + cell.owner
        return f"{heuristic}_{player}_{board_hash}"
    
    def material_advantage(self, state: core.GameState, player: int) -> float:
        """Count orbs with proximity-to-explosion bonus - optimized."""
        cache_key = self._get_cache_key(state, player, "material")
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        my_score = opp_score = 0
        opponent = 3 - player
        
        # Single pass through board
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner == player:
                    # Pre-calculate critical mass (corners=2, edges=3, center=4)
                    if (r in (0, state.rows-1)) and (c in (0, state.cols-1)):
                        crit_mass = 2
                    elif r in (0, state.rows-1) or c in (0, state.cols-1):
                        crit_mass = 3
                    else:
                        crit_mass = 4
                    
                    proximity_bonus = cell.count / crit_mass
                    my_score += cell.count * (1 + proximity_bonus)
                elif cell.owner == opponent:
                    if (r in (0, state.rows-1)) and (c in (0, state.cols-1)):
                        crit_mass = 2
                    elif r in (0, state.rows-1) or c in (0, state.cols-1):
                        crit_mass = 3
                    else:
                        crit_mass = 4
                    
                    proximity_bonus = cell.count / crit_mass
                    opp_score += cell.count * (1 + proximity_bonus)
        
        result = my_score - opp_score
        self._cache[cache_key] = result
        self._cache_misses += 1
        return result
    
    def territorial_control(self, state: core.GameState, player: int) -> float:
        """Count controlled cells with positional weighting - optimized."""
        my_cells = opp_cells = 0
        my_weighted = opp_weighted = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner:
                    # Fast critical mass calculation
                    if (r in (0, state.rows-1)) and (c in (0, state.cols-1)):
                        cell_weight = 3  # corners
                    elif r in (0, state.rows-1) or c in (0, state.cols-1):
                        cell_weight = 2  # edges
                    else:
                        cell_weight = 1  # center
                    
                    if cell.owner == player:
                        my_cells += 1
                        my_weighted += cell_weight
                    else:
                        opp_cells += 1
                        opp_weighted += cell_weight
        
        return (my_weighted - opp_weighted) * 2 + (my_cells - opp_cells)
    
    def critical_mass_proximity(self, state: core.GameState, player: int) -> float:
        """Evaluate immediate explosion threats and opportunities - optimized."""
        my_threats = opp_threats = 0
        my_near_critical = opp_near_critical = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in (player, opponent):
                    # Fast critical mass calculation
                    if (r in (0, state.rows-1)) and (c in (0, state.cols-1)):
                        crit_mass = 2
                    elif r in (0, state.rows-1) or c in (0, state.cols-1):
                        crit_mass = 3
                    else:
                        crit_mass = 4
                    
                    proximity = cell.count / crit_mass
                    
                    if cell.owner == player:
                        if cell.count == crit_mass - 1:
                            my_threats += 10
                        my_near_critical += proximity * 5
                    else:
                        if cell.count == crit_mass - 1:
                            opp_threats += 10
                        opp_near_critical += proximity * 5
        
        return (my_threats - opp_threats) + (my_near_critical - opp_near_critical)
    
    def mobility_freedom(self, state: core.GameState, player: int) -> float:
        """Evaluate move options - simplified for speed."""
        my_moves = opp_moves = 0
        my_weighted = opp_weighted = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner is None:
                    my_weighted += 2
                    opp_weighted += 2
                elif cell.owner == player:
                    my_moves += 1
                    my_weighted += 1
                else:
                    opp_moves += 1
                    opp_weighted += 1
        
        return (my_weighted - opp_weighted) * 0.5 + (my_moves - opp_moves)
    
    def chain_reaction_potential(self, state: core.GameState, player: int) -> float:
        """Evaluate potential for chain reactions - optimized."""
        my_potential = opp_potential = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in (player, opponent):
                    # Fast critical mass
                    if (r in (0, state.rows-1)) and (c in (0, state.cols-1)):
                        crit_mass = 2
                    elif r in (0, state.rows-1) or c in (0, state.cols-1):
                        crit_mass = 3
                    else:
                        crit_mass = 4
                    
                    if cell.count >= crit_mass - 1:
                        # Count adjacent cells quickly
                        neighbor_count = 0
                        for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < state.rows and 0 <= nc < state.cols:
                                neighbor = state.board[nr][nc]
                                # Fast neighbor critical mass
                                if (nr in (0, state.rows-1)) and (nc in (0, state.cols-1)):
                                    neighbor_crit = 2
                                elif nr in (0, state.rows-1) or nc in (0, state.cols-1):
                                    neighbor_crit = 3
                                else:
                                    neighbor_crit = 4
                                
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
    
    def positional_advantage(self, state: core.GameState, player: int) -> float:
        """Evaluate board position quality - simplified."""
        my_positional = opp_positional = 0
        opponent = 3 - player
        center_r, center_c = state.rows // 2, state.cols // 2
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in (player, opponent):
                    # Simplified distance calculation
                    center_distance = abs(r - center_r) + abs(c - center_c)
                    center_bonus = max(0, 5 - center_distance) * 0.5
                    
                    # Fast cluster bonus
                    cluster_bonus = 0
                    for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
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
        self.timeout = DEFAULT_TIMEOUT
        self.explosion_limit = DEFAULT_EXPLOSION_LIMIT  # NEW: Explosion limit
        self.explosion_limit_enabled = EXPLOSION_LIMIT_ENABLED  # NEW: Enable/disable explosion limiting
        self.use_transposition_table = USE_TRANSPOSITION_TABLE
        self.use_move_ordering = USE_MOVE_ORDERING
        self.use_aspiration_windows = USE_ASPIRATION_WINDOWS
    
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
    
    def set_timeout(self, timeout: float):
        """Set search timeout in seconds."""
        self.timeout = max(MIN_TIMEOUT, timeout)
    
    def set_explosion_limit(self, limit: int):
        """Set explosion limit for minimax search."""
        self.explosion_limit = max(1, limit)
    
    def enable_explosion_limiting(self, enabled: bool):
        """Enable or disable explosion limiting."""
        self.explosion_limit_enabled = enabled
    
    def get_active_weights(self) -> Dict[str, float]:
        """Get weights for only enabled heuristics."""
        return {name: weight for name, weight in self.weights.items() 
                if self.enabled_heuristics.get(name, False)}

class FastGameState:
    """Optimized game state for fast copying and move application."""
    
    def __init__(self, state: core.GameState):
        self.rows = state.rows
        self.cols = state.cols
        self.current_player = state.current_player
        self.game_over = state.game_over
        
        # Compact board representation
        self.board_owners = [[cell.owner for cell in row] for row in state.board]
        self.board_counts = [[cell.count for cell in row] for row in state.board]
        self._move_history = []
    
    def apply_move_fast(self, r: int, c: int, explosion_counter: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """Fast move application with undo capability and explosion counting."""
        # Store state for undo
        old_player = self.current_player
        move_info = [(r, c, self.board_owners[r][c], self.board_counts[r][c])]
        
        # Apply move using simplified logic
        explosions = []
        queue = [(r, c, self.current_player)]
        
        while queue:
            cr, cc, player = queue.pop(0)
            
            # Get critical mass quickly
            if (cr in (0, self.rows-1)) and (cc in (0, self.cols-1)):
                crit = 2
            elif cr in (0, self.rows-1) or cc in (0, self.cols-1):
                crit = 3
            else:
                crit = 4
            
            will_explode = self.board_counts[cr][cc] == crit - 1
            
            # Store original state
            move_info.append((cr, cc, self.board_owners[cr][cc], self.board_counts[cr][cc]))
            
            # Add orb
            self.board_owners[cr][cc] = player
            self.board_counts[cr][cc] += 1
            
            if will_explode and self.board_counts[cr][cc] >= crit:
                explosions.append((cr, cc))
                self.board_owners[cr][cc] = None
                self.board_counts[cr][cc] = 0
                
                # Increment explosion counter if provided
                if explosion_counter is not None:
                    explosion_counter[0] += 1
                
                # Add neighbors to queue
                for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        queue.append((nr, nc, player))
        
        # Update game state
        self.current_player = 3 - self.current_player
        
        # Check game over (simplified)
        owners = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board_owners[r][c]:
                    owners.add(self.board_owners[r][c])
        
        if len(owners) <= 1:
            self.game_over = True
        
        # Store move for undo
        self._move_history.append((old_player, move_info, explosions))
        
        return explosions
    
    def undo_move(self):
        """Undo the last move."""
        if not self._move_history:
            return
        
        old_player, move_info, explosions = self._move_history.pop()
        
        # Restore board state
        for r, c, owner, count in reversed(move_info):
            self.board_owners[r][c] = owner
            self.board_counts[r][c] = count
        
        self.current_player = old_player
        self.game_over = False
    
    def generate_moves_fast(self, player: int) -> List[Tuple[int, int]]:
        """Fast move generation."""
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board_owners[r][c] in (None, player):
                    moves.append((r, c))
        return moves
    
    def get_hash(self) -> int:
        """Fast board hash for transposition table."""
        h = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board_owners[r][c]:
                    h = h * 31 + (r * self.cols + c) * 10 + self.board_counts[r][c] * 2 + self.board_owners[r][c]
        return h * 2 + self.current_player

class MinimaxAgent:
    """Highly optimized Minimax agent with timeout and explosion limiting."""
    
    def __init__(self, player: int, config: Optional[AIConfig] = None):
        self.player = player
        self.config = config or AIConfig()
        self.heuristics = Heuristics()
        
        # Enhanced transposition table with heuristic values for tie-breaking
        self.transposition_table: Dict[int, Tuple[float, int, Optional[Tuple[int, int]], str, float]] = {}
        self.killer_moves: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.history_heuristic: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Timeout and explosion limiting
        self.start_time = 0
        self.timeout_reached = False
        self.explosion_counter = [0]  # NEW: Use list for mutable reference
        self.explosion_limit_reached = False  # NEW: Track if explosion limit was reached
        
        # Statistics
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
        self.table_hits = 0
        self.killer_cutoffs = 0
        self.cache_cleanups = 0
        
        # Move evaluation tracking for local best move selection
        self.evaluated_moves: List[Tuple[Tuple[int, int], float]] = []  # NEW: (move, score) pairs
        
    def _cleanup_cache(self):
        """Clean up cache keeping local optimas and high-value entries."""
        if len(self.transposition_table) < CACHE_CLEANUP_THRESHOLD:
            return
            
        # Sort by combined score: depth + heuristic value for tie-breaking
        entries = []
        for hash_key, (value, depth, move, bound_type, heuristic_score) in self.transposition_table.items():
            combined_score = depth * 100 + abs(heuristic_score)  # Prioritize depth, then heuristic value
            entries.append((combined_score, hash_key, value, depth, move, bound_type, heuristic_score))
        
        # Keep top entries (local optimas) and remove the rest
        entries.sort(reverse=True)
        keep_count = MAX_TABLE_SIZE // 2
        
        # Clear table and repopulate with best entries
        self.transposition_table.clear()
        for i in range(min(keep_count, len(entries))):
            _, hash_key, value, depth, move, bound_type, heuristic_score = entries[i]
            self.transposition_table[hash_key] = (value, depth, move, bound_type, heuristic_score)
        
        self.cache_cleanups += 1
    
    def _is_timeout(self) -> bool:
        """Check if timeout has been reached."""
        if self.timeout_reached:
            return True
        
        if time.time() - self.start_time >= self.config.timeout:
            self.timeout_reached = True
            return True
        return False
    
    def _is_explosion_limit_reached(self) -> bool:
        """Check if explosion limit has been reached."""
        if not self.config.explosion_limit_enabled:
            return False
            
        if self.explosion_limit_reached:
            return True
            
        if self.explosion_counter[0] >= self.config.explosion_limit:
            self.explosion_limit_reached = True
            return True
        return False
    
    def _should_stop_search(self) -> bool:
        """Check if search should stop due to timeout or explosion limit."""
        return self._is_timeout() or self._is_explosion_limit_reached()
    
    def evaluate_state_fast(self, fast_state: FastGameState) -> float:
        """Fast state evaluation using optimized heuristics."""
        # Convert to regular state for heuristics (can be optimized further)
        state = core.GameState(fast_state.rows, fast_state.cols)
        state.current_player = fast_state.current_player
        state.game_over = fast_state.game_over
        
        for r in range(fast_state.rows):
            for c in range(fast_state.cols):
                state.board[r][c].owner = fast_state.board_owners[r][c]
                state.board[r][c].count = fast_state.board_counts[r][c]
        
        # Check terminal states first
        if fast_state.game_over:
            if len({fast_state.board_owners[r][c] for r in range(fast_state.rows) 
                   for c in range(fast_state.cols) 
                   if fast_state.board_owners[r][c]}) == 1:
                # Find winner
                for r in range(fast_state.rows):
                    for c in range(fast_state.cols):
                        if fast_state.board_owners[r][c]:
                            winner = fast_state.board_owners[r][c]
                            return math.inf if winner == self.player else -math.inf
            return 0  # Draw
        
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
    
    def order_moves_advanced(self, moves: List[Tuple[int, int]], fast_state: FastGameState, depth: int) -> List[Tuple[int, int]]:
        """Advanced move ordering with killer moves and history heuristic."""
        if not self.config.use_move_ordering or len(moves) <= 1:
            return moves
        
        def move_score(move):
            r, c = move
            score = 0
            
            # Killer moves get highest priority
            if move in self.killer_moves.get(depth, []):
                score += 10000
            
            # History heuristic
            score += self.history_heuristic.get(move, 0)
            
            # Positional bonuses
            if (r in (0, fast_state.rows-1)) and (c in (0, fast_state.cols-1)):
                score += 100  # Corner
            elif r in (0, fast_state.rows-1) or c in (0, fast_state.cols-1):
                score += 50   # Edge
            
            # Empty cells preferred
            if fast_state.board_owners[r][c] is None:
                score += 20
            
            # Near critical mass
            owner = fast_state.board_owners[r][c]
            count = fast_state.board_counts[r][c]
            if owner == fast_state.current_player and count >= 2:
                score += count * 10
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def alpha_beta_optimized(self, fast_state: FastGameState, depth: int, alpha: float, beta: float, 
                           maximizing_player: bool) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Highly optimized alpha-beta search with timeout and explosion limiting."""
        # Check termination conditions first
        if self._should_stop_search():
            return self.evaluate_state_fast(fast_state), None
            
        self.nodes_explored += 1
        
        # Terminal conditions
        if depth == 0 or fast_state.game_over:
            eval_score = self.evaluate_state_fast(fast_state)
            return eval_score, None
        
        # Transposition table lookup with tie-breaking
        state_hash = fast_state.get_hash()
        if self.config.use_transposition_table and state_hash in self.transposition_table:
            cached_value, cached_depth, cached_move, bound_type, cached_heuristic = self.transposition_table[state_hash]
            if cached_depth >= depth:
                self.table_hits += 1
                if bound_type == 'EXACT':
                    return cached_value, cached_move
                elif bound_type == 'LOWER' and cached_value >= beta:
                    return cached_value, cached_move
                elif bound_type == 'UPPER' and cached_value <= alpha:
                    return cached_value, cached_move
        
        current_player = fast_state.current_player
        legal_moves = fast_state.generate_moves_fast(current_player)
        
        if not legal_moves:
            eval_score = self.evaluate_state_fast(fast_state)
            return eval_score, None
        
        # Advanced move ordering
        ordered_moves = self.order_moves_advanced(legal_moves, fast_state, depth)
        best_action = None
        original_alpha = alpha
        
        if maximizing_player:
            max_eval = -math.inf
            for i, move in enumerate(ordered_moves):
                if self._should_stop_search():  # Check termination conditions during search
                    break
                    
                r, c = move
                
                # Apply move with explosion counting
                fast_state.apply_move_fast(r, c, self.explosion_counter)
                eval_score, _ = self.alpha_beta_optimized(fast_state, depth - 1, alpha, beta, False)
                fast_state.undo_move()  # Undo move
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = move
                    
                    # Update history heuristic for good moves
                    if self.config.use_move_ordering:
                        self.history_heuristic[move] += depth * depth
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    
                    # Store killer move
                    if self.config.use_move_ordering and move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:  # Keep only 2 killer moves per depth
                            self.killer_moves[depth].pop()
                        self.killer_cutoffs += 1
                    break
            
            # Store in transposition table with heuristic value for tie-breaking
            if self.config.use_transposition_table and not self._should_stop_search():
                # Clean cache if needed
                if len(self.transposition_table) >= MAX_TABLE_SIZE:
                    self._cleanup_cache()
                
                if len(self.transposition_table) < MAX_TABLE_SIZE:
                    heuristic_value = abs(max_eval) if max_eval != math.inf and max_eval != -math.inf else 0
                    if max_eval <= original_alpha:
                        bound_type = 'UPPER'
                    elif max_eval >= beta:
                        bound_type = 'LOWER'
                    else:
                        bound_type = 'EXACT'
                    self.transposition_table[state_hash] = (max_eval, depth, best_action, bound_type, heuristic_value)
            
            return max_eval, best_action
        
        else:  # Minimizing player
            min_eval = math.inf
            for move in ordered_moves:
                if self._should_stop_search():  # Check termination conditions during search
                    break
                    
                r, c = move
                
                # Apply move with explosion counting
                fast_state.apply_move_fast(r, c, self.explosion_counter)
                eval_score, _ = self.alpha_beta_optimized(fast_state, depth - 1, alpha, beta, True)
                fast_state.undo_move()  # Undo move
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = move
                    
                    # Update history heuristic for good moves
                    if self.config.use_move_ordering:
                        self.history_heuristic[move] += depth * depth
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    
                    # Store killer move
                    if self.config.use_move_ordering and move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                        self.killer_cutoffs += 1
                    break
            
            # Store in transposition table with heuristic value for tie-breaking
            if self.config.use_transposition_table and not self._should_stop_search():
                # Clean cache if needed
                if len(self.transposition_table) >= MAX_TABLE_SIZE:
                    self._cleanup_cache()
                
                if len(self.transposition_table) < MAX_TABLE_SIZE:
                    heuristic_value = abs(min_eval) if min_eval != math.inf and min_eval != -math.inf else 0
                    if min_eval <= original_alpha:
                        bound_type = 'UPPER'
                    elif min_eval >= beta:
                        bound_type = 'LOWER'
                    else:
                        bound_type = 'EXACT'
                    self.transposition_table[state_hash] = (min_eval, depth, best_action, bound_type, heuristic_value)
            
            return min_eval, best_action
    
    def search_with_aspiration_windows(self, fast_state: FastGameState, depth: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Search with aspiration windows for better pruning."""
        if not self.config.use_aspiration_windows or self._should_stop_search():
            return self.alpha_beta_optimized(fast_state, depth, -math.inf, math.inf, True)
        
        # Initial search with narrow window
        alpha, beta = -50, 50
        value, move = self.alpha_beta_optimized(fast_state, depth, alpha, beta, True)
        
        # If search failed and we have time, re-search with wider window
        if not self._should_stop_search():
            if value <= alpha:
                value, move = self.alpha_beta_optimized(fast_state, depth, -math.inf, beta, True)
            elif value >= beta:
                value, move = self.alpha_beta_optimized(fast_state, depth, alpha, math.inf, True)
        
        return value, move
    
    def _get_best_local_move(self, fast_state: FastGameState) -> Tuple[int, int]:
        """Get the best move from evaluated moves or quick local evaluation."""
        # If we have evaluated moves, return the best one
        if self.evaluated_moves:
            self.evaluated_moves.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
            best_move, best_score = self.evaluated_moves[0]
            return best_move
        
        # Fallback: Quick local evaluation of available moves
        legal_moves = fast_state.generate_moves_fast(fast_state.current_player)
        if not legal_moves:
            raise RuntimeError("No legal moves available")
        
        best_move = legal_moves[0]
        best_score = -math.inf
        
        # Quick evaluation of each move (depth 1)
        for move in legal_moves:
            r, c = move
            fast_state.apply_move_fast(r, c, self.explosion_counter)
            score = self.evaluate_state_fast(fast_state)
            fast_state.undo_move()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def choose_move(self, state: core.GameState) -> Tuple[int, int]:
        """Choose the best move using optimized search with explosion limiting."""
        # Reset statistics and limits
        self.nodes_explored = 0
        self.alpha_beta_cutoffs = 0
        self.table_hits = 0
        self.killer_cutoffs = 0
        self.start_time = time.time()
        self.timeout_reached = False
        self.explosion_counter = [0]  # Reset explosion counter
        self.explosion_limit_reached = False
        self.evaluated_moves = []  # Reset evaluated moves
        
        # Convert to fast state
        fast_state = FastGameState(state)
        
        # Iterative deepening with timeout and explosion limiting
        best_move = None
        last_complete_depth = 0
        
        for d in range(1, self.config.depth + 1):
            if self._should_stop_search():
                break
                
            try:
                value, move = self.search_with_aspiration_windows(fast_state, d)
                if move and not self._should_stop_search():
                    best_move = move
                    last_complete_depth = d
                    # Store this move evaluation
                    self.evaluated_moves.append((move, value))
                else:
                    break  # Timeout or explosion limit reached during this depth
            except KeyboardInterrupt:
                break
        
        # If explosion limit was reached, get the best local move
        if self.explosion_limit_reached and self.config.explosion_limit_enabled:
            if best_move is None or len(self.evaluated_moves) == 0:
                best_move = self._get_best_local_move(fast_state)
        
        # Fallback to legal move if no move found
        if best_move is None:
            legal_moves = state.generate_moves(self.player)
            if legal_moves:
                return legal_moves[0]
            else:
                raise RuntimeError("No legal moves available")
        
        return best_move
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics including explosion data."""
        total_lookups = self.table_hits + max(1, self.nodes_explored - self.table_hits)
        hit_rate = (self.table_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        return {
            'nodes_explored': self.nodes_explored,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs,
            'killer_cutoffs': self.killer_cutoffs,
            'search_depth': self.config.depth,
            'timeout_seconds': self.config.timeout,
            'timeout_reached': self.timeout_reached,
            'explosion_limit': self.config.explosion_limit,
            'explosion_limit_enabled': self.config.explosion_limit_enabled,
            'explosions_processed': self.explosion_counter[0],
            'explosion_limit_reached': self.explosion_limit_reached,
            'evaluated_moves_count': len(self.evaluated_moves),
            'search_time': time.time() - self.start_time if self.start_time > 0 else 0,
            'table_hits': self.table_hits,
            'hit_rate_percent': hit_rate,
            'cache_cleanups': self.cache_cleanups,
            'table_size': len(self.transposition_table),
            'enabled_heuristics': [name for name, enabled in self.config.enabled_heuristics.items() if enabled],
            'heuristic_weights': self.config.get_active_weights(),
            'cache_hits': self.heuristics._cache_hits,
            'cache_misses': self.heuristics._cache_misses
        }

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_aggressive_config(timeout: float = 3.0, explosion_limit: int = 50) -> AIConfig:
    """Aggressive AI focusing on immediate threats and chain reactions."""
    config = AIConfig()
    config.weights = {
        'material': 2.0,
        'territorial': 1.0,
        'critical_mass': 6.0,
        'mobility': 1.0,
        'chain_potential': 4.0,
        'positional': 0.5
    }
    config.set_timeout(timeout)
    config.set_explosion_limit(explosion_limit)
    return config

def create_defensive_config(timeout: float = 3.0, explosion_limit: int = 50) -> AIConfig:
    """Defensive AI focusing on territory and material advantage."""
    config = AIConfig()
    config.weights = {
        'material': 5.0,
        'territorial': 4.0,
        'critical_mass': 2.0,
        'mobility': 2.0,
        'chain_potential': 1.0,
        'positional': 3.0
    }
    config.set_timeout(timeout)
    config.set_explosion_limit(explosion_limit)
    return config

def create_balanced_config(timeout: float = 5.0, explosion_limit: int = 50) -> AIConfig:
    """Balanced AI with default weights."""
    config = AIConfig()
    config.set_timeout(timeout)
    config.set_explosion_limit(explosion_limit)
    return config

def create_fast_config(timeout: float = 1.0, explosion_limit: int = 25) -> AIConfig:
    """Fast AI for quick responses with lower explosion limit."""
    config = AIConfig()
    config.set_depth(2)
    config.set_timeout(timeout)
    config.set_explosion_limit(explosion_limit)
    return config

def create_material_only_config(timeout: float = 3.0, explosion_limit: int = 50) -> AIConfig:
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
    config.set_timeout(timeout)
    config.set_explosion_limit(explosion_limit)
    return config

def create_unlimited_explosions_config(timeout: float = 5.0) -> AIConfig:
    """AI with explosion limiting disabled."""
    config = AIConfig()
    config.enable_explosion_limiting(False)
    config.set_timeout(timeout)
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