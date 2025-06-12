#!/usr/bin/env python3
"""
Chain Reaction Game - Unified AI Engine
Combines game logic, AI algorithms, and file protocol into a single file.
Reads/writes gamestate.txt and tracks moves until game completion.
"""

from __future__ import annotations

import sys
import time
import os
import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# ────────────────────────── Config / Constants ───────────────────────── #
ROWS, COLS = 6, 9
RED = (220, 20, 60)
BLUE = (30, 144, 255)
PLAYER_COLOR = {1: RED, 2: BLUE}

# File configuration
GAMESTATE_FILE = "gamestate.txt"
CONFIG_FILE = "ai_config.txt"

# AI Configuration Constants
DEFAULT_WEIGHTS = {
    'material': 3.0,
    'territorial': 2.0,
    'critical_mass': 4.0,
    'mobility': 1.5,
    'chain_potential': 2.5,
    'positional': 1.0
}

DEFAULT_DEPTH = 3
MAX_DEPTH = 6
MIN_DEPTH = 1
USE_TRANSPOSITION_TABLE = True
MAX_TABLE_SIZE = 5000
USE_MOVE_ORDERING = True


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


# ───────────────────────────── Game Engine ───────────────────────────── #
class GameState:
    """
    Pure game-state container with file I/O and game logic.
    """
    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows, self.cols = rows, cols
        self.board: List[List[Cell]] = [[Cell() for _ in range(cols)]
                                        for _ in range(rows)]
        self.current_player: int = 1
        self.turns_played: int = 0
        self.game_over: bool = False

    # ─────────────────── File-protocol helpers ─────────────────── #
    @staticmethod
    def from_file(lines: list[str]) -> "GameState":
        """
        Build a GameState from 9 lines of 6 tokens each (no header).
        Tokens: 0  or  <n>R / <n>B .
        """
        if len(lines) != 9:
            raise ValueError("File must have exactly 9 board rows.")
        gs = GameState(rows=9, cols=6)  # protocol is 9×6
        for r, line in enumerate(lines):
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
        out: list[str] = []
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
        self.turns_played = 0
        self.game_over = False

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

        self.turns_played += 1

        owners = {cell.owner for row in self.board for cell in row if cell.owner}

        if self.turns_played >= 2 and self._only_one_owner_left():
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

            crit = critical_mass(self.rows, self.cols, cr, cc)
            will_explode = cell.count == crit - 1

            cell.add_orb(p)
            if will_explode and cell.count >= crit:
                explosions.append((cr, cc))
                cell.reset()
                for nr, nc in self._neighbours(cr, cc):
                    q.append((nr, nc, p))

    def _only_one_owner_left(self) -> bool:
        owners = {cell.owner for row in self.board for cell in row if cell.owner}
        return len(owners) == 1

    # ─────────────────── Utility for AI ─────────────────── #
    def clone(self) -> "GameState":
        """Deep copy of this state."""
        return copy.deepcopy(self)

    def generate_moves(self, player: int) -> list[tuple[int, int]]:
        """Return all legal (row, col) moves for *player* in this state."""
        moves = []
        for r, row in enumerate(self.board):
            for c, cell in enumerate(row):
                if cell.owner in (None, player):
                    moves.append((r, c))
        return moves


# ───────────────────────────── AI Heuristics ──────────────────────────── #
class Heuristics:
    """Collection of heuristic evaluation functions."""
    
    @staticmethod
    def material_advantage(state: GameState, player: int) -> float:
        """Count orbs with proximity-to-explosion bonus."""
        my_score = opp_score = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner == player:
                    crit_mass = critical_mass(state.rows, state.cols, r, c)
                    proximity_bonus = cell.count / crit_mass
                    my_score += cell.count * (1 + proximity_bonus)
                elif cell.owner == opponent:
                    crit_mass = critical_mass(state.rows, state.cols, r, c)
                    proximity_bonus = cell.count / crit_mass
                    opp_score += cell.count * (1 + proximity_bonus)
        
        return my_score - opp_score
    
    @staticmethod
    def territorial_control(state: GameState, player: int) -> float:
        """Count controlled cells with positional weighting."""
        my_cells = opp_cells = 0
        my_weighted = opp_weighted = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                crit_mass = critical_mass(state.rows, state.cols, r, c)
                cell_weight = 5 - crit_mass  # corners=3, edges=2, center=1
                
                if cell.owner == player:
                    my_cells += 1
                    my_weighted += cell_weight
                elif cell.owner == opponent:
                    opp_cells += 1
                    opp_weighted += cell_weight
        
        return (my_weighted - opp_weighted) * 2 + (my_cells - opp_cells)
    
    @staticmethod
    def critical_mass_proximity(state: GameState, player: int) -> float:
        """Evaluate immediate explosion threats and opportunities."""
        my_threats = opp_threats = 0
        my_near_critical = opp_near_critical = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    crit_mass = critical_mass(state.rows, state.cols, r, c)
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
    
    @staticmethod
    def mobility_freedom(state: GameState, player: int) -> float:
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
    def chain_reaction_potential(state: GameState, player: int) -> float:
        """Evaluate potential for chain reactions."""
        my_potential = opp_potential = 0
        opponent = 3 - player
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.board[r][c]
                if cell.owner in [player, opponent]:
                    crit_mass = critical_mass(state.rows, state.cols, r, c)
                    
                    if cell.count >= crit_mass - 1:
                        neighbor_count = 0
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < state.rows and 0 <= nc < state.cols:
                                neighbor = state.board[nr][nc]
                                neighbor_crit = critical_mass(state.rows, state.cols, nr, nc)
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
    def positional_advantage(state: GameState, player: int) -> float:
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


# ───────────────────────────── AI Configuration ──────────────────────────── #
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


# ───────────────────────────── AI Agent ──────────────────────────── #
class MinimaxAgent:
    """Configurable Minimax agent with selectable heuristics."""
    
    def __init__(self, player: int, config: Optional[AIConfig] = None):
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
    
    def evaluate_state(self, state: GameState) -> float:
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
    
    def _board_hash(self, state: GameState) -> str:
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
    
    def _order_moves(self, moves: List[Tuple[int, int]], state: GameState) -> List[Tuple[int, int]]:
        """Order moves for better alpha-beta pruning."""
        if not self.config.use_move_ordering:
            return moves
        
        def move_score(move):
            r, c = move
            score = self.move_history.get(move, 0)
            
            # Prefer corners and edges
            crit_mass = critical_mass(state.rows, state.cols, r, c)
            if crit_mass == 2:      # Corner
                score += 5
            elif crit_mass == 3:    # Edge
                score += 3
            
            # Prefer empty cells
            if state.board[r][c].owner is None:
                score += 2
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def minimax_search(self, state: GameState, depth_limit: int) -> Tuple[float, Optional[Tuple[int, int]]]:
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
    
    def _alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, 
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
    
    def choose_move(self, state: GameState) -> Tuple[int, int]:
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


# ───────────────────────────── AI Presets ──────────────────────────── #
def create_aggressive_config() -> AIConfig:
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
    return config

def create_defensive_config() -> AIConfig:
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


# ───────────────────────────── File I/O Engine ──────────────────────────── #
def read_file_waiting_for(header: str) -> list[str]:
    """Block until the file starts with the desired header."""
    while True:
        try:
            with open(GAMESTATE_FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            time.sleep(0.1)
            continue

        if lines and lines[0].strip() == header:
            return lines[1:]
        time.sleep(0.1)

def write_state(header: str, state: GameState):
    """Write game state to file."""
    data = [header] + state.to_lines()
    with open(GAMESTATE_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(data))

def load_ai_config() -> AIConfig:
    """Load AI configuration from file or use defaults."""
    try:
        if os.path.exists(CONFIG_FILE):
            config = AIConfig()
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith("depth="):
                    config.set_depth(int(line.split("=")[1]))
                elif line.startswith("preset="):
                    preset_name = line.split("=")[1]
                    if preset_name == "aggressive":
                        config = create_aggressive_config()
                    elif preset_name == "defensive":
                        config = create_defensive_config()
                    elif preset_name == "material_only":
                        config = create_material_only_config()
                    else:
                        config = create_balanced_config()
                elif "=" in line:
                    # Handle individual heuristic settings
                    key, value = line.split("=", 1)
                    if key.endswith("_enabled"):
                        heuristic = key.replace("_enabled", "")
                        if heuristic in config.enabled_heuristics:
                            config.enabled_heuristics[heuristic] = value.lower() == "true"
                    elif key.endswith("_weight"):
                        heuristic = key.replace("_weight", "")
                        if heuristic in config.weights:
                            config.weights[heuristic] = float(value)
            
            print(f"Loaded AI configuration from {CONFIG_FILE}")
            return config
        else:
            print("No AI configuration file found, using defaults")
            return AIConfig()
            
    except Exception as e:
        print(f"Error loading AI config: {e}, using defaults")
        return AIConfig()

def create_ai_agent(config: AIConfig) -> MinimaxAgent:
    """Create AI agent with the given configuration."""
    agent = MinimaxAgent(player=2, config=config)  # Blue player
    
    # Print configuration summary
    print("AI Agent Configuration:")
    print(f"  Search depth: {config.depth}")
    print(f"  Enabled heuristics: {[h for h, enabled in config.enabled_heuristics.items() if enabled]}")
    print(f"  Active weights: {config.get_active_weights()}")
    
    return agent


# ───────────────────────────── Main Engine ──────────────────────────── #
def main():
    """Main engine loop."""
    print("Chain Reaction AI Engine starting...")
    
    # Load AI configuration
    config = load_ai_config()
    agent = create_ai_agent(config)
    
    # If file doesn't exist, create an empty board
    if not os.path.exists(GAMESTATE_FILE):
        empty = GameState(rows=9, cols=6)
        write_state("AI Move:", empty)
        print("Created initial game state file")

    print("AI Engine ready – waiting for human moves…")
    
    move_count = 0
    game_start_time = time.time()
    
    while True:
        try:
            # Wait for human move
            board_lines = read_file_waiting_for("Human Move:")
            state = GameState.from_file(board_lines)
            state.current_player = 2  # AI's turn (Blue)

            if state.game_over:
                write_state("AI Move:", state)
                winner = state.get_winner()
                game_end_time = time.time()
                total_game_time = game_end_time - game_start_time
                
                print(f"\n=== GAME COMPLETED ===")
                print(f"Total moves played: {state.turns_played}")
                print(f"Total game time: {total_game_time:.2f} seconds")
                print(f"Average time per move: {total_game_time/max(1, state.turns_played):.2f} seconds")
                
                if winner == 1:
                    print("Result: Human won!")
                elif winner == 2:
                    print("Result: AI won!")
                else:
                    print("Result: Game ended in draw")
                break

            move_count += 1
            print(f"\nMove {move_count}: AI thinking...")
            
            # AI chooses move
            start_time = time.time()
            r, c = agent.choose_move(state.clone())
            end_time = time.time()
            
            # Apply move
            state.apply_move(r, c)
            
            # Get and display statistics
            stats = agent.get_search_statistics()
            print(f"  Chosen move: ({r}, {c})")
            print(f"  Time taken: {end_time - start_time:.3f}s")
            print(f"  Nodes explored: {stats['nodes_explored']}")
            print(f"  Alpha-beta cutoffs: {stats['alpha_beta_cutoffs']}")
            
            if 'hit_rate_percent' in stats:
                print(f"  Table hit rate: {stats['hit_rate_percent']:.1f}%")

            # Save result
            write_state("AI Move:", state)

            # Check if AI won
            if state.game_over:
                winner = state.get_winner()
                game_end_time = time.time()
                total_game_time = game_end_time - game_start_time
                
                print(f"\n=== GAME COMPLETED ===")
                print(f"Total moves played: {state.turns_played}")
                print(f"Total game time: {total_game_time:.2f} seconds")
                print(f"Average time per move: {total_game_time/max(1, state.turns_played):.2f} seconds")
                
                if winner == 2:
                    print("Result: AI wins!")
                else:
                    print("Result: Game ended")
                break
                
        except KeyboardInterrupt:
            print("\nAI Engine interrupted by user")
            break
        except Exception as e:
            print(f"Error in AI Engine: {e}")
            # Try to continue with a random move
            try:
                legal_moves = state.generate_moves(2)
                if legal_moves:
                    r, c = legal_moves[0]
                    state.apply_move(r, c)
                    write_state("AI Move:", state)
                    print(f"Made fallback move: ({r}, {c})")
                else:
                    print("No legal moves available")
                    break
            except:
                print("Critical error - exiting")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down AI Engine")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)