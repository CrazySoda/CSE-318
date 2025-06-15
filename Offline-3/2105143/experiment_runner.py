#!/usr/bin/env python3
"""
AI Configuration Experiment Runner
Comprehensive testing script for Task 4 & 5 of Chain Reaction AI assignment
Generates data for statistical analysis and reporting
"""

import time
import json
import csv
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import statistics
from datetime import datetime

# Import your existing modules
import core
import ai

@dataclass
class ExperimentResult:
    """Detailed result of a single experiment."""
    config1_name: str
    config2_name: str
    config1_details: Dict[str, Any]
    config2_details: Dict[str, Any]
    player1_wins: int
    player2_wins: int
    draws: int
    total_games: int
    avg_moves_per_game: float
    avg_game_duration: float
    min_moves: int
    max_moves: int
    std_moves: float
    avg_nodes_explored_p1: float
    avg_nodes_explored_p2: float
    avg_search_time_p1: float
    avg_search_time_p2: float
    total_experiment_time: float
    individual_games: List[Dict[str, Any]]
    experiment_type: str = ""
    parameter_tested: str = ""
    
    @property
    def player1_win_rate(self) -> float:
        return self.player1_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def player2_win_rate(self) -> float:
        return self.player2_wins / self.total_games if self.total_games > 0 else 0.0

class ExperimentRunner:
    """Runs comprehensive AI experiments for assignment analysis."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Available configurations for testing
        self.ai_configs = {
            'random': None,  # Special case for random agent
            'balanced': ai.create_balanced_config,
            'aggressive': ai.create_aggressive_config,
            'defensive': ai.create_defensive_config,
            'tactical': ai.create_tactical_config,
            'strategic': ai.create_strategic_config,
            'fast': ai.create_fast_config,
            'material_only': ai.create_material_only_config,
            # Single heuristic focus configurations
            'material_focus': ai.create_material_focus_config,
            'territorial_focus': ai.create_territorial_focus_config,
            'critical_mass_focus': ai.create_critical_mass_focus_config,
            'mobility_focus': ai.create_mobility_focus_config,
            'chain_focus': ai.create_chain_focus_config,
            'positional_focus': ai.create_positional_focus_config,
            # Hybrid configurations
            'tactical_plus': ai.create_tactical_plus_config,
            'strategic_plus': ai.create_strategic_plus_config,
        }
        
        # Experiment configurations
        self.depth_tests = [2, 3, 4, 5]
        self.timeout_tests = [1.0, 3.0, 5.0, 10.0]
        self.games_per_experiment = 20  # Adjust based on time constraints
        
        self.results: List[ExperimentResult] = []
    
    def create_random_agent(self, player: int):
        """Create a random move agent."""
        return ai.RandomAgent(player)
    
    def create_ai_agent(self, config_name: str, player: int, depth: Optional[int] = None, 
                       timeout: Optional[float] = None):
        """Create an AI agent with optional parameter overrides."""
        if config_name == 'random':
            return self.create_random_agent(player)
        
        config = self.ai_configs[config_name]()
        
        if depth is not None:
            config.set_depth(depth)
        if timeout is not None:
            config.set_timeout(timeout)
            
        return ai.MinimaxAgent(player, config)
    
    def get_config_details(self, config_name: str, agent=None) -> Dict[str, Any]:
        """Extract configuration details for reporting."""
        if config_name == 'random':
            return {
                'type': 'random',
                'depth': 0,
                'timeout': 0,
                'heuristics': [],
                'weights': {}
            }
        
        if agent and hasattr(agent, 'config'):
            config = agent.config
            return {
                'type': config_name,
                'depth': config.depth,
                'timeout': config.timeout,
                'explosion_limit': getattr(config, 'explosion_limit', 0),
                'explosion_limit_enabled': getattr(config, 'explosion_limit_enabled', False),
                'enabled_heuristics': [h for h, enabled in getattr(config, 'enabled_heuristics', {}).items() if enabled],
                'weights': getattr(config, 'get_active_weights', lambda: {})(),
                'use_move_ordering': getattr(config, 'use_move_ordering', False)
            }
        
        # Fallback - create config to get details
        try:
            config = self.ai_configs[config_name]()
            return {
                'type': config_name,
                'depth': config.depth,
                'timeout': config.timeout,
                'explosion_limit': getattr(config, 'explosion_limit', 0),
                'explosion_limit_enabled': getattr(config, 'explosion_limit_enabled', False),
                'enabled_heuristics': [h for h, enabled in getattr(config, 'enabled_heuristics', {}).items() if enabled],
                'weights': getattr(config, 'get_active_weights', lambda: {})(),
                'use_move_ordering': getattr(config, 'use_move_ordering', False)
            }
        except Exception as e:
            print(f"Warning: Could not get config details for {config_name}: {e}")
            return {'type': config_name, 'error': str(e)}
    
    def run_single_experiment(self, config1_name: str, config2_name: str, 
                             num_games: int = None, depth1: int = None, depth2: int = None, # type: ignore
                             timeout1: float = None, timeout2: float = None, # type: ignore
                             show_progress: bool = True) -> ExperimentResult:
        """Run a single experiment between two configurations."""
        if num_games is None:
            num_games = self.games_per_experiment
            
        if show_progress:
            print(f"\n🧪 EXPERIMENT: {config1_name} vs {config2_name}")
            print(f"Parameters: depth1={depth1}, depth2={depth2}, timeout1={timeout1}, timeout2={timeout2}")
            print(f"Games: {num_games}")
        
        experiment_start = time.time()
        
        # Create agents
        agent1 = self.create_ai_agent(config1_name, 1, depth1, timeout1)
        agent2 = self.create_ai_agent(config2_name, 2, depth2, timeout2)
        
        # Get configuration details
        config1_details = self.get_config_details(config1_name, agent1)
        config2_details = self.get_config_details(config2_name, agent2)
        
        # Game statistics
        player1_wins = player2_wins = draws = 0
        game_moves = []
        game_durations = []
        nodes_explored_p1 = []
        nodes_explored_p2 = []
        search_times_p1 = []
        search_times_p2 = []
        individual_games = []
        
        # Headless experiments for speed
        for game_num in range(num_games):
            if show_progress:
                print(f"  Game {game_num + 1}/{num_games}...", end=" ")
            
            game_start = time.time()
            state = core.GameState(rows=9, cols=6)
            move_count = 0
            max_moves = 1000  # Safety limit
            
            # Game statistics for this game
            game_stats_p1 = {'nodes': [], 'search_time': []}
            game_stats_p2 = {'nodes': [], 'search_time': []}
            
            while not state.game_over and move_count < max_moves:
                current_agent = agent1 if state.current_player == 1 else agent2
                
                try:
                    move_start = time.time()
                    move = current_agent.choose_move(state.clone())
                    move_time = time.time() - move_start
                    
                    # Collect statistics
                    if hasattr(current_agent, 'get_search_statistics'):
                        stats = current_agent.get_search_statistics()
                        if state.current_player == 1:
                            game_stats_p1['nodes'].append(stats.get('nodes_explored', 0))
                            game_stats_p1['search_time'].append(move_time)
                        else:
                            game_stats_p2['nodes'].append(stats.get('nodes_explored', 0))
                            game_stats_p2['search_time'].append(move_time)
                    
                    if move is None:
                        if show_progress:
                            print(f"Warning: Agent returned None move in game {game_num + 1}")
                        break
                        
                    state.apply_move(move[0], move[1])
                    move_count += 1
                    
                except Exception as e:
                    if show_progress:
                        print(f"Error in game {game_num + 1}: {e}")
                    break
            
            game_duration = time.time() - game_start
            winner = state.get_winner() if state.game_over else None
            
            # Update statistics
            if winner == 1:
                player1_wins += 1
            elif winner == 2:
                player2_wins += 1
            else:
                draws += 1
            
            game_moves.append(move_count)
            game_durations.append(game_duration)
            
            # Collect average statistics for this game
            if game_stats_p1['nodes']:
                nodes_explored_p1.append(statistics.mean(game_stats_p1['nodes']))
                search_times_p1.append(statistics.mean(game_stats_p1['search_time']))
            
            if game_stats_p2['nodes']:
                nodes_explored_p2.append(statistics.mean(game_stats_p2['nodes']))
                search_times_p2.append(statistics.mean(game_stats_p2['search_time']))
            
            individual_games.append({
                'game_number': game_num + 1,
                'winner': winner,
                'moves': move_count,
                'duration': game_duration,
                'avg_nodes_p1': statistics.mean(game_stats_p1['nodes']) if game_stats_p1['nodes'] else 0,
                'avg_nodes_p2': statistics.mean(game_stats_p2['nodes']) if game_stats_p2['nodes'] else 0,
                'avg_search_time_p1': statistics.mean(game_stats_p1['search_time']) if game_stats_p1['search_time'] else 0,
                'avg_search_time_p2': statistics.mean(game_stats_p2['search_time']) if game_stats_p2['search_time'] else 0
            })
            
            if show_progress:
                print(f"Winner: {winner or 'Draw'}, Moves: {move_count}, Duration: {game_duration:.1f}s")
        
        experiment_duration = time.time() - experiment_start
        
        # Calculate statistics
        avg_moves = statistics.mean(game_moves) if game_moves else 0
        avg_duration = statistics.mean(game_durations) if game_durations else 0
        min_moves = min(game_moves) if game_moves else 0
        max_moves = max(game_moves) if game_moves else 0
        std_moves = statistics.stdev(game_moves) if len(game_moves) > 1 else 0
        
        avg_nodes_p1 = statistics.mean(nodes_explored_p1) if nodes_explored_p1 else 0
        avg_nodes_p2 = statistics.mean(nodes_explored_p2) if nodes_explored_p2 else 0
        avg_search_time_p1 = statistics.mean(search_times_p1) if search_times_p1 else 0
        avg_search_time_p2 = statistics.mean(search_times_p2) if search_times_p2 else 0
        
        # Create result
        result = ExperimentResult(
            config1_name=config1_name,
            config2_name=config2_name,
            config1_details=config1_details,
            config2_details=config2_details,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_games=num_games,
            avg_moves_per_game=avg_moves,
            avg_game_duration=avg_duration,
            min_moves=min_moves,
            max_moves=max_moves,
            std_moves=std_moves,
            avg_nodes_explored_p1=avg_nodes_p1,
            avg_nodes_explored_p2=avg_nodes_p2,
            avg_search_time_p1=avg_search_time_p1,
            avg_search_time_p2=avg_search_time_p2,
            total_experiment_time=experiment_duration,
            individual_games=individual_games
        )
        
        if show_progress:
            print(f"✅ EXPERIMENT COMPLETE: {config1_name} vs {config2_name}")
            print(f"   Results: {player1_wins}-{player2_wins}-{draws}")
            print(f"   Win rates: {result.player1_win_rate:.1%} - {result.player2_win_rate:.1%}")
            print(f"   Avg moves: {avg_moves:.1f}, Avg duration: {avg_duration:.1f}s")
            print(f"   Total time: {experiment_duration:.1f}s")
        
        return result
    
    def run_task4_experiments(self):
        """Run Task 4 experiments: depth and time limit analysis."""
        print("\n" + "="*60)
        print("TASK 4: DEPTH AND TIME LIMIT EXPERIMENTS")
        print("="*60)
        
        task4_results = []
        
        # 1. Random vs AI at different depths
        print("\n1. Testing different search depths against random agent:")
        for depth in self.depth_tests:
            result = self.run_single_experiment(
                'random', 'balanced',
                num_games=15,
                depth2=depth,
                timeout2=5.0
            )
            result.experiment_type = 'depth_analysis'
            result.parameter_tested = f'depth_{depth}'
            task4_results.append(result)
        
        # 2. Time limit analysis
        print("\n2. Testing different time limits:")
        for timeout in self.timeout_tests:
            result = self.run_single_experiment(
                'balanced', 'aggressive',
                num_games=10,
                timeout1=timeout,
                timeout2=timeout
            )
            result.experiment_type = 'timeout_analysis'
            result.parameter_tested = f'timeout_{timeout}'
            task4_results.append(result)
        
        # 3. Heuristic effectiveness
        print("\n3. Testing individual heuristics against random:")
        heuristic_configs = [
            ('material_only', 'Material Only'),
            ('tactical', 'Tactical'),
            ('defensive', 'Defensive'),
            ('aggressive', 'Aggressive')
        ]
        
        for config_name, display_name in heuristic_configs:
            result = self.run_single_experiment(
                'random', config_name,
                num_games=15
            )
            result.experiment_type = 'heuristic_analysis'
            result.parameter_tested = config_name
            task4_results.append(result)
        
        self.save_task4_results(task4_results)
        self.generate_task4_summary(task4_results)
        
        return task4_results
    
    def run_task5_experiments(self):
        """Run Task 5 (bonus) experiments: AI vs AI comprehensive analysis."""
        print("\n" + "="*60)
        print("TASK 5: AI VS AI COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        task5_results = []
        
        # 1. Round-robin tournament
        print("\n1. Round-robin tournament between all AI configurations:")
        ai_configs = ['balanced', 'aggressive', 'defensive', 'tactical', 'strategic', 'fast']
        
        for i, config1 in enumerate(ai_configs):
            for config2 in ai_configs[i+1:]:  # Avoid duplicate matchups
                result = self.run_single_experiment(
                    config1, config2,
                    num_games=20
                )
                result.experiment_type = 'tournament'
                result.parameter_tested = f'{config1}_vs_{config2}'
                task5_results.append(result)
        
        # 2. Depth comparison with best heuristics
        print("\n2. Depth comparison between top performers:")
        top_configs = ['balanced', 'tactical']
        
        for config in top_configs:
            for depth in [2, 3, 4]:
                result = self.run_single_experiment(
                    config, config,
                    num_games=10,
                    depth1=depth,
                    depth2=3  # Baseline
                )
                result.experiment_type = 'depth_comparison'
                result.parameter_tested = f'{config}_depth_{depth}_vs_3'
                task5_results.append(result)
        
        # 3. Performance vs Random baseline
        print("\n3. All configurations vs random agent:")
        for config in ai_configs:
            result = self.run_single_experiment(
                'random', config,
                num_games=25
            )
            result.experiment_type = 'vs_random'
            result.parameter_tested = f'{config}_vs_random'
            task5_results.append(result)
        
        self.save_task5_results(task5_results)
        self.generate_task5_summary(task5_results)
        
        return task5_results
    
    def run_heuristic_comparison_experiments(self):
        """Run comprehensive heuristic comparison experiments."""
        print("\n" + "="*60)
        print("HEURISTIC COMPARISON EXPERIMENTS")
        print("="*60)
        
        heuristic_results = []
        
        # Single heuristic configurations
        single_heuristics = [
            'material_focus',
            'territorial_focus', 
            'critical_mass_focus',
            'mobility_focus',
            'chain_focus',
            'positional_focus'
        ]
        
        # 1. Single heuristics vs Random agent
        print("\n1. Single heuristics vs Random agent:")
        for heuristic in single_heuristics:
            result = self.run_single_experiment(
                'random', heuristic,
                num_games=20
            )
            result.experiment_type = 'heuristic_vs_random'
            result.parameter_tested = f'{heuristic}_vs_random'
            heuristic_results.append(result)
        
        # 2. Round-robin tournament between single heuristics
        print("\n2. Single heuristic round-robin tournament:")
        for i, heuristic1 in enumerate(single_heuristics):
            for heuristic2 in single_heuristics[i+1:]:
                result = self.run_single_experiment(
                    heuristic1, heuristic2,
                    num_games=15
                )
                result.experiment_type = 'heuristic_vs_heuristic'
                result.parameter_tested = f'{heuristic1}_vs_{heuristic2}'
                heuristic_results.append(result)
        
        # 3. Single heuristics vs hybrid configurations
        print("\n3. Single heuristics vs hybrid strategies:")
        hybrid_configs = ['tactical_plus', 'strategic_plus', 'balanced']
        
        for heuristic in single_heuristics:
            for hybrid in hybrid_configs:
                result = self.run_single_experiment(
                    heuristic, hybrid,
                    num_games=12
                )
                result.experiment_type = 'heuristic_vs_hybrid'
                result.parameter_tested = f'{heuristic}_vs_{hybrid}'
                heuristic_results.append(result)
        
        # 4. Depth impact on individual heuristics
        print("\n4. Depth impact on top-performing heuristics:")
        # Find top 3 heuristics from vs_random results
        vs_random_results = [r for r in heuristic_results if r.experiment_type == 'heuristic_vs_random']
        top_heuristics = sorted(vs_random_results, key=lambda x: x.player2_win_rate, reverse=True)[:3]
        
        for result in top_heuristics:
            heuristic = result.config2_name
            for depth in [2, 3, 4]:
                depth_result = self.run_single_experiment(
                    'random', heuristic,
                    num_games=8,
                    depth2=depth
                )
                depth_result.experiment_type = 'heuristic_depth_test'
                depth_result.parameter_tested = f'{heuristic}_depth_{depth}'
                heuristic_results.append(depth_result)
        
        self.save_heuristic_results(heuristic_results)
        self.generate_heuristic_summary(heuristic_results)
        
        return heuristic_results
    
    def save_task4_results(self, results: List[ExperimentResult]):
        """Save Task 4 results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_file = os.path.join(self.output_dir, f"task4_detailed_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary CSV
        csv_file = os.path.join(self.output_dir, f"task4_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment_Type', 'Parameter', 'Config1', 'Config2',
                'P1_Wins', 'P2_Wins', 'Draws', 'P1_Win_Rate', 'P2_Win_Rate',
                'Avg_Moves', 'Avg_Duration', 'Avg_Nodes_P1', 'Avg_Nodes_P2',
                'Total_Time'
            ])
            
            for r in results:
                writer.writerow([
                    r.experiment_type,
                    r.parameter_tested,
                    r.config1_name, r.config2_name,
                    r.player1_wins, r.player2_wins, r.draws,
                    f"{r.player1_win_rate:.3f}", f"{r.player2_win_rate:.3f}",
                    f"{r.avg_moves_per_game:.1f}", f"{r.avg_game_duration:.1f}",
                    f"{r.avg_nodes_explored_p1:.0f}", f"{r.avg_nodes_explored_p2:.0f}",
                    f"{r.total_experiment_time:.1f}"
                ])
        
        print(f"\n📊 Task 4 results saved:")
        print(f"   Detailed: {json_file}")
        print(f"   Summary:  {csv_file}")
    
    def save_task5_results(self, results: List[ExperimentResult]):
        """Save Task 5 results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_file = os.path.join(self.output_dir, f"task5_detailed_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary CSV
        csv_file = os.path.join(self.output_dir, f"task5_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment_Type', 'Parameter', 'Config1', 'Config2',
                'P1_Wins', 'P2_Wins', 'Draws', 'P1_Win_Rate', 'P2_Win_Rate',
                'Avg_Moves', 'Avg_Duration', 'Avg_Nodes_P1', 'Avg_Nodes_P2',
                'Avg_Search_Time_P1', 'Avg_Search_Time_P2', 'Total_Time'
            ])
            
            for r in results:
                writer.writerow([
                    r.experiment_type,
                    r.parameter_tested,
                    r.config1_name, r.config2_name,
                    r.player1_wins, r.player2_wins, r.draws,
                    f"{r.player1_win_rate:.3f}", f"{r.player2_win_rate:.3f}",
                    f"{r.avg_moves_per_game:.1f}", f"{r.avg_game_duration:.1f}",
                    f"{r.avg_nodes_explored_p1:.0f}", f"{r.avg_nodes_explored_p2:.0f}",
                    f"{r.avg_search_time_p1:.3f}", f"{r.avg_search_time_p2:.3f}",
                    f"{r.total_experiment_time:.1f}"
                ])
        
        print(f"\n📊 Task 5 results saved:")
        print(f"   Detailed: {json_file}")
        print(f"   Summary:  {csv_file}")
    
    def save_heuristic_results(self, results: List[ExperimentResult]):
        """Save heuristic comparison results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_file = os.path.join(self.output_dir, f"heuristic_comparison_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary CSV
        csv_file = os.path.join(self.output_dir, f"heuristic_comparison_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment_Type', 'Parameter', 'Config1', 'Config2',
                'P1_Wins', 'P2_Wins', 'Draws', 'P1_Win_Rate', 'P2_Win_Rate',
                'Avg_Moves', 'Avg_Duration', 'Avg_Nodes_P1', 'Avg_Nodes_P2',
                'Avg_Search_Time_P1', 'Avg_Search_Time_P2', 'Total_Time'
            ])
            
            for r in results:
                writer.writerow([
                    r.experiment_type,
                    r.parameter_tested,
                    r.config1_name, r.config2_name,
                    r.player1_wins, r.player2_wins, r.draws,
                    f"{r.player1_win_rate:.3f}", f"{r.player2_win_rate:.3f}",
                    f"{r.avg_moves_per_game:.1f}", f"{r.avg_game_duration:.1f}",
                    f"{r.avg_nodes_explored_p1:.0f}", f"{r.avg_nodes_explored_p2:.0f}",
                    f"{r.avg_search_time_p1:.3f}", f"{r.avg_search_time_p2:.3f}",
                    f"{r.total_experiment_time:.1f}"
                ])
        
        print(f"\n📊 Heuristic comparison results saved:")
        print(f"   Detailed: {json_file}")
        print(f"   Summary:  {csv_file}")
    
    def generate_task4_summary(self, results: List[ExperimentResult]):
        """Generate Task 4 analysis summary."""
        print("\n" + "="*60)
        print("TASK 4 ANALYSIS SUMMARY")
        print("="*60)
        
        # Depth analysis
        depth_results = [r for r in results if r.experiment_type == 'depth_analysis']
        if depth_results:
            print("\n📈 DEPTH ANALYSIS (vs Random Agent):")
            print("Depth | Win Rate | Avg Moves | Avg Nodes | Search Time")
            print("-" * 55)
            for r in sorted(depth_results, key=lambda x: int(x.parameter_tested.split('_')[1])):
                depth = r.parameter_tested.split('_')[1]
                print(f"  {depth}   |  {r.player2_win_rate:6.1%}  |   {r.avg_moves_per_game:6.1f}  |   {r.avg_nodes_explored_p2:7.0f} |    {r.avg_search_time_p2:.3f}s")
        
        # Timeout analysis
        timeout_results = [r for r in results if r.experiment_type == 'timeout_analysis']
        if timeout_results:
            print("\n⏱️  TIMEOUT ANALYSIS (Balanced vs Aggressive):")
            print("Timeout | P1 Win Rate | P2 Win Rate | Avg Duration")
            print("-" * 50)
            for r in sorted(timeout_results, key=lambda x: float(x.parameter_tested.split('_')[1])):
                timeout = r.parameter_tested.split('_')[1]
                print(f"  {timeout}s   |    {r.player1_win_rate:6.1%}   |    {r.player2_win_rate:6.1%}   |    {r.avg_game_duration:6.1f}s")
        
        # Heuristic analysis
        heuristic_results = [r for r in results if r.experiment_type == 'heuristic_analysis']
        if heuristic_results:
            print("\n🧠 HEURISTIC ANALYSIS (vs Random Agent):")
            print("Heuristic    | Win Rate | Avg Moves | Performance")
            print("-" * 50)
            for r in sorted(heuristic_results, key=lambda x: x.player2_win_rate, reverse=True):
                performance = "Excellent" if r.player2_win_rate > 0.8 else "Good" if r.player2_win_rate > 0.6 else "Fair"
                print(f"{r.parameter_tested:12} |  {r.player2_win_rate:6.1%}  |   {r.avg_moves_per_game:6.1f}  | {performance}")
    
    def generate_task5_summary(self, results: List[ExperimentResult]):
        """Generate Task 5 analysis summary."""
        print("\n" + "="*60)
        print("TASK 5 ANALYSIS SUMMARY")
        print("="*60)
        
        # Tournament results
        tournament_results = [r for r in results if r.experiment_type == 'tournament']
        if tournament_results:
            print("\n🏆 TOURNAMENT RESULTS:")
            print("Matchup                    | Winner     | Score    | Win Rate")
            print("-" * 65)
            for r in tournament_results:
                winner = r.config1_name if r.player1_wins > r.player2_wins else r.config2_name if r.player2_wins > r.player1_wins else "Draw"
                score = f"{r.player1_wins}-{r.player2_wins}-{r.draws}"
                win_rate = max(r.player1_win_rate, r.player2_win_rate)
                print(f"{r.config1_name:12} vs {r.config2_name:12} | {winner:10} | {score:8} | {win_rate:6.1%}")
        
        # Performance vs Random
        vs_random_results = [r for r in results if r.experiment_type == 'vs_random']
        if vs_random_results:
            print("\n🎯 PERFORMANCE VS RANDOM AGENT:")
            print("Config       | Win Rate | Avg Moves | Efficiency")
            print("-" * 50)
            for r in sorted(vs_random_results, key=lambda x: x.player2_win_rate, reverse=True):
                config = r.parameter_tested.split('_vs_')[0]
                efficiency = f"{r.avg_moves_per_game:.0f} moves"
                print(f"{config:12} |  {r.player2_win_rate:6.1%}  |   {r.avg_moves_per_game:6.1f}  | {efficiency}")
    
    def generate_heuristic_summary(self, results: List[ExperimentResult]):
        """Generate comprehensive heuristic comparison summary."""
        print("\n" + "="*70)
        print("HEURISTIC COMPARISON ANALYSIS SUMMARY")
        print("="*70)
        
        # 1. Performance vs Random
        vs_random_results = [r for r in results if r.experiment_type == 'heuristic_vs_random']
        if vs_random_results:
            print("\n🎯 INDIVIDUAL HEURISTIC PERFORMANCE VS RANDOM:")
            print("Heuristic         | Win Rate | Avg Moves | Efficiency | Rank")
            print("-" * 65)
            sorted_results = sorted(vs_random_results, key=lambda x: x.player2_win_rate, reverse=True)
            for i, r in enumerate(sorted_results, 1):
                heuristic = r.config2_name.replace('_focus', '').replace('_', ' ').title()
                efficiency = "Excellent" if r.player2_win_rate > 0.8 else "Good" if r.player2_win_rate > 0.6 else "Fair"
                print(f"{heuristic:17} |  {r.player2_win_rate:6.1%}  |   {r.avg_moves_per_game:6.1f}  | {efficiency:10} | #{i}")
    
    def run_interactive_experiment(self):
        """Run a single interactive experiment."""
        print("\n🎮 INTERACTIVE EXPERIMENT MODE")
        print("Available configurations:", list(self.ai_configs.keys()))
        
        config1 = input("Enter first configuration: ").strip()
        config2 = input("Enter second configuration: ").strip()
        
        if config1 not in self.ai_configs or config2 not in self.ai_configs:
            print("Invalid configuration!")
            return
        
        try:
            num_games = int(input("Number of games (default 5): ").strip() or "5")
        except ValueError:
            num_games = 5
        
        result = self.run_single_experiment(config1, config2, num_games)
        
        print(f"\n📊 EXPERIMENT RESULTS:")
        print(f"Matchup: {result.config1_name} vs {result.config2_name}")
        print(f"Games: {result.total_games}")
        print(f"Score: {result.player1_wins}-{result.player2_wins}-{result.draws}")
        print(f"Win rates: {result.player1_win_rate:.1%} - {result.player2_win_rate:.1%}")
        print(f"Average moves: {result.avg_moves_per_game:.1f}")
        print(f"Average duration: {result.avg_game_duration:.1f}s")


def main():
    """Main experiment runner with menu."""
    print("🧪 AI CONFIGURATION EXPERIMENT RUNNER")
    print("For Chain Reaction Assignment Tasks 4 & 5")
    print("=" * 50)
    
    runner = ExperimentRunner()
    
    while True:
        print(f"\nExperiment Options:")
        print("1. Run Task 4 Experiments (Depth & Time Analysis)")
        print("2. Run Task 5 Experiments (AI vs AI Tournament)")
        print("3. Run Heuristic Comparison Experiments")
        print("4. Run Both Tasks (Full Experiment Suite)")
        print("5. Interactive Single Experiment")
        print("6. Quick Demo (3 games)")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (0-6): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == "0":
            break
        
        elif choice == "1":
            print("\n🚀 Starting Task 4 Experiments...")
            print("This will test different depths and time limits.")
            confirm = input("This may take 20-30 minutes. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    runner.run_task4_experiments()
                    print("\n✅ Task 4 experiments complete!")
                except Exception as e:
                    print(f"\n❌ Task 4 experiments failed: {e}")
        
        elif choice == "2":
            print("\n🚀 Starting Task 5 Experiments...")
            print("This will run comprehensive AI vs AI analysis.")
            confirm = input("This may take 30-60 minutes. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    runner.run_task5_experiments()
                    print("\n✅ Task 5 experiments complete!")
                except Exception as e:
                    print(f"\n❌ Task 5 experiments failed: {e}")
        
        elif choice == "3":
            print("\n🧠 Starting Heuristic Comparison Experiments...")
            print("This will compare individual heuristics head-to-head.")
            confirm = input("This may take 45-75 minutes. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    runner.run_heuristic_comparison_experiments()
                    print("\n✅ Heuristic comparison experiments complete!")
                except Exception as e:
                    print(f"\n❌ Heuristic comparison experiments failed: {e}")
        
        elif choice == "4":
            print("\n🚀 Starting Full Experiment Suite...")
            confirm = input("This may take 60-90 minutes. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    runner.run_task4_experiments()
                    runner.run_task5_experiments()
                    print("\n✅ All experiments complete!")
                except Exception as e:
                    print(f"\n❌ Experiments failed: {e}")
        
        elif choice == "5":
            try:
                runner.run_interactive_experiment()
            except Exception as e:
                print(f"\n❌ Interactive experiment failed: {e}")
        
        elif choice == "6":
            print("\n🎮 Quick Demo: Balanced vs Aggressive")
            try:
                result = runner.run_single_experiment('balanced', 'aggressive', 3)
                print(f"Demo complete! Winner: {result.config1_name if result.player1_wins > result.player2_wins else result.config2_name}")
            except Exception as e:
                print(f"\n❌ Quick demo failed: {e}")
        
        else:
            print("Invalid choice!")
    
    print(f"\n📁 Results saved in: {runner.output_dir}/")
    print("Import CSV files into Excel/Google Sheets for analysis and charting.")


if __name__ == "__main__":
    main()