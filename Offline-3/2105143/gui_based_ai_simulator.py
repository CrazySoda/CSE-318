#!/usr/bin/env python3
"""
GUI Battle Controller
Uses gui.py's exact AI vs AI system to simulate battles programmatically
"""

import time
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import threading

# Import GUI and disable pygame for headless operation
import pygame
pygame.init = lambda: None  # Disable pygame initialization
pygame.display.set_mode = lambda *args, **kwargs: None
pygame.display.set_caption = lambda *args: None
pygame.font.SysFont = lambda *args, **kwargs: None

import gui
import core
import ai

@dataclass
class BattleResult:
    """Result of a single battle."""
    config1_name: str
    config2_name: str
    player1_wins: int
    player2_wins: int
    draws: int
    total_games: int
    individual_results: List[Tuple[int, int, float]]  # (winner, moves, duration)
    
    @property
    def player1_win_rate(self) -> float:
        return self.player1_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def player2_win_rate(self) -> float:
        return self.player2_wins / self.total_games if self.total_games > 0 else 0.0

class GUIBattleController:
    """Controls GUI's AI vs AI system for automated battles."""
    
    def __init__(self):
        self.gui_instance = None
        self.available_configs = {
            'balanced': ai.create_balanced_config,
            'aggressive': ai.create_aggressive_config,
            'defensive': ai.create_defensive_config,
            'tactical': ai.create_tactical_config,
            'strategic': ai.create_strategic_config,
            'fast': ai.create_fast_config,
            'material_only': ai.create_material_only_config,
        }
    
    def create_headless_gui(self) -> gui.ChainReactionGUI:
        """Create a GUI instance without pygame display."""
        # Create GUI instance with mocked pygame
        gui_instance = gui.ChainReactionGUI()
        
        # Mock the fonts to avoid pygame errors
        gui_instance.fonts = {
            'title': None, 'header': None, 'normal': None, 'small': None, 'tiny': None
        }
        
        # Mock the screen
        gui_instance.screen = None
        gui_instance.clock = None
        
        return gui_instance
    
    def setup_battle(self, gui_instance: gui.ChainReactionGUI, 
                    config1_name: str, config2_name: str) -> bool:
        """Set up a battle between two AI configurations."""
        try:
            # Set AI configurations exactly like GUI does
            gui_instance.ai_player1_config = self.available_configs[config1_name]()
            gui_instance.ai_player2_config = self.available_configs[config2_name]()
            
            # Set mode to AI vs AI
            gui_instance.mode = gui.MODE_AI_VS_AI
            
            # Disable auto-restart and set fast move delay for testing
            gui_instance.auto_restart = False
            gui_instance.ai_move_delay = 0.0
            
            # Start AI vs AI exactly like GUI does
            gui_instance._start_ai_vs_ai()
            
            return True
        except Exception as e:
            print(f"Error setting up battle: {e}")
            return False
    
    def run_single_game(self, gui_instance: gui.ChainReactionGUI, 
                       game_id: int, verbose: bool = True) -> Tuple[int, int, float]:
        """Run a single game and return (winner, moves, duration)."""
        if verbose:
            print(f"  Game {game_id}: ", end="", flush=True)
        
        game_start_time = time.time()
        moves_count = 0
        max_moves = 1000  # Safety limit
        timeout = 300  # 5 minutes timeout
        
        # Game loop using GUI's exact update_ai_vs_ai method
        while (gui_instance.ai_vs_ai_running and 
               not gui_instance.state.game_over and 
               moves_count < max_moves):
            
            current_time = time.time()
            if current_time - game_start_time > timeout:
                if verbose:
                    print("TIMEOUT")
                break
            
            # Store move count before update
            prev_moves = gui_instance.match_stats.current_game_moves
            
            # Use GUI's exact update method
            gui_instance.update_ai_vs_ai()
            
            # Check if a move was made
            if gui_instance.match_stats.current_game_moves > prev_moves:
                moves_count = gui_instance.match_stats.current_game_moves
                if verbose and moves_count % 25 == 0:
                    print(f"{moves_count}...", end="", flush=True)
            
            # Small delay to prevent busy waiting
            time.sleep(0.001)
        
        game_duration = time.time() - game_start_time
        winner = gui_instance.state.get_winner() if gui_instance.state.game_over else None
        
        if verbose:
            if winner:
                print(f" Winner = Player {winner} in {moves_count} moves ({game_duration:.1f}s)")
            else:
                print(f" Draw in {moves_count} moves ({game_duration:.1f}s)")
        
        return winner, moves_count, game_duration
    
    def battle_configs(self, config1_name: str, config2_name: str, 
                      num_games: int = 20, verbose: bool = True) -> BattleResult:
        """Battle two AI configurations using GUI's exact system."""
        print(f"\n🎮 GUI Battle: {config1_name} vs {config2_name}")
        print(f"Running {num_games} games using GUI's exact AI vs AI system...")
        print("="*70)
        
        if config1_name not in self.available_configs:
            raise ValueError(f"Unknown config: {config1_name}")
        if config2_name not in self.available_configs:
            raise ValueError(f"Unknown config: {config2_name}")
        
        player1_wins = 0
        player2_wins = 0
        draws = 0
        individual_results = []
        
        for game in range(1, num_games + 1):
            # Create fresh GUI instance for each game to avoid state pollution
            gui_instance = self.create_headless_gui()
            
            # Set up the battle
            if not self.setup_battle(gui_instance, config1_name, config2_name):
                print(f"  Failed to set up game {game}")
                continue
            
            # Run the game
            winner, moves, duration = self.run_single_game(gui_instance, game, verbose)
            individual_results.append((winner, moves, duration))
            
            # Update counters
            if winner == 1:
                player1_wins += 1
            elif winner == 2:
                player2_wins += 1
            else:
                draws += 1
            
            # Progress update
            if game % 5 == 0:
                print(f"\nProgress: {game}/{num_games} completed")
                print(f"Current: {config1_name} {player1_wins}-{player2_wins} {config2_name} (Draws: {draws})")
        
        # Create result
        result = BattleResult(
            config1_name=config1_name,
            config2_name=config2_name,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_games=num_games,
            individual_results=individual_results
        )
        
        # Print final results
        print(f"\n📊 FINAL RESULTS: {config1_name} vs {config2_name}")
        print("="*70)
        print(f"Player 1 ({config1_name}): {player1_wins}/{num_games} wins ({result.player1_win_rate:.1%})")
        print(f"Player 2 ({config2_name}): {player2_wins}/{num_games} wins ({result.player2_win_rate:.1%})")
        print(f"Draws: {draws}/{num_games} ({draws/num_games:.1%})")
        
        # Game statistics
        if individual_results:
            move_counts = [r[1] for r in individual_results]
            durations = [r[2] for r in individual_results]
            print(f"Average moves per game: {sum(move_counts)/len(move_counts):.1f}")
            print(f"Average game duration: {sum(durations)/len(durations):.2f}s")
            print(f"Move range: {min(move_counts)} - {max(move_counts)} moves")
        
        return result
    
    def tournament(self, configs_to_test: List[str] = None, 
                  num_games_per_matchup: int = 15) -> Dict[str, Any]:
        """Run a tournament between multiple configurations."""
        if configs_to_test is None:
            configs_to_test = list(self.available_configs.keys())
        
        print(f"\n🏆 GUI Tournament")
        print(f"Configurations: {configs_to_test}")
        print(f"Games per matchup: {num_games_per_matchup}")
        print("="*80)
        
        all_battles = []
        standings = {config: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0} for config in configs_to_test}
        
        total_matchups = len(configs_to_test) * (len(configs_to_test) - 1) // 2
        current_matchup = 0
        
        for i, config1 in enumerate(configs_to_test):
            for j, config2 in enumerate(configs_to_test[i+1:], i+1):
                current_matchup += 1
                print(f"\nMatchup {current_matchup}/{total_matchups}")
                
                result = self.battle_configs(config1, config2, num_games_per_matchup, verbose=False)
                all_battles.append(result)
                
                # Update standings (3 points for win, 1 for draw)
                standings[config1]['wins'] += result.player1_wins
                standings[config1]['losses'] += result.player2_wins
                standings[config1]['draws'] += result.draws
                standings[config1]['points'] += result.player1_wins * 3 + result.draws
                
                standings[config2]['wins'] += result.player2_wins
                standings[config2]['losses'] += result.player1_wins
                standings[config2]['draws'] += result.draws
                standings[config2]['points'] += result.player2_wins * 3 + result.draws
        
        # Calculate final standings
        for config in standings:
            total_games = standings[config]['wins'] + standings[config]['losses'] + standings[config]['draws']
            standings[config]['total_games'] = total_games
            standings[config]['win_rate'] = standings[config]['wins'] / total_games if total_games > 0 else 0
        
        # Sort by points, then by win rate
        sorted_standings = sorted(standings.items(), 
                                key=lambda x: (x[1]['points'], x[1]['win_rate']), 
                                reverse=True)
        
        # Print final standings
        print(f"\n🏆 TOURNAMENT FINAL STANDINGS")
        print("="*80)
        print(f"{'Rank':<4} {'Configuration':<15} {'W-L-D':<12} {'Points':<8} {'Win Rate':<10}")
        print("-"*80)
        
        for rank, (config, stats) in enumerate(sorted_standings, 1):
            w_l_d = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"
            print(f"{rank:<4} {config:<15} {w_l_d:<12} {stats['points']:<8} {stats['win_rate']:.1%}")
        
        return {
            'battles': all_battles,
            'standings': standings,
            'sorted_standings': sorted_standings
        }
    
    def analyze_config_details(self, config_name: str):
        """Analyze the details of a specific configuration."""
        if config_name not in self.available_configs:
            print(f"Unknown configuration: {config_name}")
            return
        
        config = self.available_configs[config_name]()
        
        print(f"\n🔍 Configuration Analysis: {config_name}")
        print("="*50)
        print(f"Search Depth: {config.depth}")
        print(f"Timeout: {config.timeout}s")
        print(f"Explosion Limit: {config.explosion_limit}")
        print(f"Explosion Limit Enabled: {config.explosion_limit_enabled}")
        
        print(f"\nEnabled Heuristics:")
        for heuristic, enabled in config.enabled_heuristics.items():
            if enabled:
                weight = config.weights.get(heuristic, 0)
                print(f"  ✓ {heuristic.replace('_', ' ').title()}: {weight:.1f}")
            else:
                print(f"  ✗ {heuristic.replace('_', ' ').title()}: Disabled")
        
        print(f"\nOptimizations:")
        print(f"  Transposition Table: {'ON' if config.use_transposition_table else 'OFF'}")
        print(f"  Move Ordering: {'ON' if config.use_move_ordering else 'OFF'}")
        print(f"  Aspiration Windows: {'ON' if config.use_aspiration_windows else 'OFF'}")

def main():
    """Main function with menu system."""
    controller = GUIBattleController()
    
    print("🎮 GUI Battle Controller")
    print("Uses gui.py's exact AI vs AI system for perfect consistency")
    print("="*60)
    
    available_configs = list(controller.available_configs.keys())
    print(f"Available configurations: {', '.join(available_configs)}")
    
    while True:
        print(f"\nChoose an option:")
        print("1. Battle two configurations")
        print("2. Reproduce aggressive vs balanced test")
        print("3. Quick tournament (top 4 configs)")
        print("4. Full tournament (all configs)")
        print("5. Analyze configuration details")
        print("6. Custom battle with detailed results")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (0-6): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == "0":
            break
        
        elif choice == "1":
            config1 = input(f"Enter first config ({'/'.join(available_configs)}): ").strip()
            config2 = input(f"Enter second config ({'/'.join(available_configs)}): ").strip()
            num_games = int(input("Enter number of games (default 20): ").strip() or "20")
            
            if config1 in available_configs and config2 in available_configs:
                controller.battle_configs(config1, config2, num_games, verbose=True)
            else:
                print("Invalid configuration names!")
        
        elif choice == "2":
            print("\n🔄 Reproducing the exact test: aggressive vs balanced")
            print("This uses GUI's exact AI vs AI system...")
            controller.battle_configs("aggressive", "balanced", num_games=20, verbose=True)
        
        elif choice == "3":
            top_configs = ['balanced', 'aggressive', 'defensive', 'tactical']
            controller.tournament(top_configs, num_games_per_matchup=10)
        
        elif choice == "4":
            controller.tournament(num_games_per_matchup=12)
        
        elif choice == "5":
            config = input(f"Enter config to analyze ({'/'.join(available_configs)}): ").strip()
            if config in available_configs:
                controller.analyze_config_details(config)
            else:
                print("Invalid configuration name!")
        
        elif choice == "6":
            config1 = input(f"Enter first config ({'/'.join(available_configs)}): ").strip()
            config2 = input(f"Enter second config ({'/'.join(available_configs)}): ").strip()
            num_games = int(input("Enter number of games (default 50): ").strip() or "50")
            
            if config1 in available_configs and config2 in available_configs:
                result = controller.battle_configs(config1, config2, num_games, verbose=True)
                
                # Show detailed analysis
                print(f"\n📈 Detailed Analysis:")
                print("Individual game results:")
                for i, (winner, moves, duration) in enumerate(result.individual_results[:10], 1):
                    winner_name = config1 if winner == 1 else config2 if winner == 2 else "Draw"
                    print(f"  Game {i}: {winner_name} in {moves} moves ({duration:.1f}s)")
                if len(result.individual_results) > 10:
                    print(f"  ... and {len(result.individual_results) - 10} more games")
            else:
                print("Invalid configuration names!")
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()