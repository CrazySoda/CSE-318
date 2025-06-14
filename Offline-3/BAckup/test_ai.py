"""
Test script for the Configurable Minimax AI Agent
Updated to work with your current ai.py implementation.
Demonstrates the capabilities of the six heuristic functions and alpha-beta pruning.
"""

import sys
import time
from typing import Dict, List
import core
import ai  # Import your current ai module

def create_test_scenario_1() -> core.GameState:
    """Create a test scenario with some pieces on the board."""
    state = core.GameState(rows=6, cols=9)
    
    # Set up a scenario with various pieces
    # Player 1 (Red) pieces
    state.board[0][0].owner = 1
    state.board[0][0].count = 1  # Corner piece
    
    state.board[2][3].owner = 1
    state.board[2][3].count = 3  # Center piece near critical mass
    
    state.board[4][1].owner = 1
    state.board[4][1].count = 2  # Edge piece
    
    # Player 2 (Blue) pieces
    state.board[1][1].owner = 2
    state.board[1][1].count = 1
    
    state.board[3][4].owner = 2
    state.board[3][4].count = 2
    
    state.board[5][7].owner = 2
    state.board[5][7].count = 1  # Corner piece
    
    state.current_player = 1
    return state

def create_test_scenario_2() -> core.GameState:
    """Create a scenario focusing on chain reaction potential."""
    state = core.GameState(rows=6, cols=9)
    
    # Create a setup where chain reactions are possible
    # Player 1 pieces in a cluster
    state.board[2][2].owner = 1
    state.board[2][2].count = 3  # About to explode
    
    state.board[2][3].owner = 1
    state.board[2][3].count = 2
    
    state.board[3][2].owner = 1
    state.board[3][2].count = 2
    
    state.board[1][2].owner = 1
    state.board[1][2].count = 1
    
    # Player 2 pieces
    state.board[4][5].owner = 2
    state.board[4][5].count = 2
    
    state.board[0][8].owner = 2
    state.board[0][8].count = 1  # Corner
    
    state.current_player = 1
    return state

def create_test_scenario_3() -> core.GameState:
    """Create a complex mid-game scenario."""
    state = core.GameState(rows=6, cols=9)
    
    # More complex setup
    positions = [
        (0, 1, 1, 1), (1, 2, 1, 2), (2, 0, 1, 2), (3, 3, 1, 3),
        (1, 5, 2, 1), (2, 6, 2, 2), (4, 4, 2, 3), (5, 1, 2, 1)
    ]
    
    for r, c, owner, count in positions:
        state.board[r][c].owner = owner
        state.board[r][c].count = count
    
    state.current_player = 1
    return state

def test_individual_heuristics(state: core.GameState) -> Dict[str, float]:
    """Test each heuristic function individually using your current ai.py."""
    heuristics = ai.Heuristics()  # Use ai.Heuristics instead of ChainReactionHeuristics
    
    results = {
        'material_advantage': heuristics.material_advantage(state, 1),
        'territorial_control': heuristics.territorial_control(state, 1),
        'critical_mass_proximity': heuristics.critical_mass_proximity(state, 1),
        'mobility_freedom': heuristics.mobility_freedom(state, 1),
        'chain_reaction_potential': heuristics.chain_reaction_potential(state, 1),
        'positional_advantage': heuristics.positional_advantage(state, 1)
    }
    
    return results

def test_minimax_performance(agent: ai.MinimaxAgent, state: core.GameState, test_name: str = "") -> Dict:
    """Test the minimax agent's performance and gather statistics."""
    print(f"Testing minimax performance{' - ' + test_name if test_name else ''}...")
    
    start_time = time.time()
    value, move = agent.minimax_search(state, agent.config.depth)  # Use agent.config.depth
    end_time = time.time()
    
    stats = agent.get_search_statistics()
    stats['search_time'] = end_time - start_time
    stats['evaluation_value'] = value
    stats['chosen_move'] = move
    
    return stats

def compare_agent_configurations():
    """Compare different agent configurations using your preset functions."""
    state = create_test_scenario_1()
    
    # Use your actual preset functions
    configurations = [
        ('Balanced', ai.create_balanced_config),
        ('Aggressive', ai.create_aggressive_config),
        ('Defensive', ai.create_defensive_config),
        ('Material Only', ai.create_material_only_config)
    ]
    
    print("\n=== Agent Configuration Comparison ===")
    for name, config_func in configurations:
        config = config_func()
        agent = ai.MinimaxAgent(player=1, config=config)  # Use ai.MinimaxAgent with config
        
        start_time = time.time()
        value, move = agent.minimax_search(state, 3)
        end_time = time.time()
        
        stats = agent.get_search_statistics()
        
        print(f"\n{name} Agent:")
        print(f"  Chosen move: {move}")
        print(f"  Evaluation: {value:.2f}")
        print(f"  Time: {end_time - start_time:.4f}s")
        print(f"  Nodes explored: {stats['nodes_explored']}")
        print(f"  Alpha-beta cutoffs: {stats['alpha_beta_cutoffs']}")
        if 'hit_rate_percent' in stats:
            print(f"  Table hit rate: {stats['hit_rate_percent']:.1f}%")

def test_individual_heuristic_impact():
    """Test the impact of individual heuristics by enabling only one at a time."""
    state = create_test_scenario_1()
    
    print("\n=== Individual Heuristic Impact Test ===")
    
    heuristic_names = list(ai.DEFAULT_WEIGHTS.keys())
    
    for heuristic_name in heuristic_names:
        # Create config with only this heuristic enabled
        config = ai.AIConfig()
        
        # Disable all heuristics
        for h in config.enabled_heuristics:
            config.enabled_heuristics[h] = False
        
        # Enable only the current one
        config.enabled_heuristics[heuristic_name] = True
        config.weights[heuristic_name] = 1.0  # Set to 1.0 for fair comparison
        
        agent = ai.MinimaxAgent(player=1, config=config)
        
        start_time = time.time()
        value, move = agent.minimax_search(state, 3)
        end_time = time.time()
        
        print(f"\n{heuristic_name.replace('_', ' ').title()} Only:")
        print(f"  Move: {move}")
        print(f"  Value: {value:.2f}")
        print(f"  Time: {end_time - start_time:.4f}s")

def test_depth_scaling():
    """Test how performance scales with search depth."""
    state = create_test_scenario_2()
    
    print("\n=== Search Depth Scaling Test ===")
    
    for depth in range(1, 6):
        config = ai.create_balanced_config()
        config.set_depth(depth)
        agent = ai.MinimaxAgent(player=1, config=config)
        
        start_time = time.time()
        try:
            value, move = agent.minimax_search(state, depth)
            end_time = time.time()
            stats = agent.get_search_statistics()
            
            print(f"Depth {depth}: Move {move}, Value {value:.2f}, "
                  f"Nodes {stats['nodes_explored']}, Time {end_time-start_time:.4f}s")
            
            # Stop if it takes too long
            if end_time - start_time > 5.0:
                print(f"  (Stopping depth test - took too long)")
                break
                
        except Exception as e:
            print(f"Depth {depth}: Error - {e}")
            break

def test_weight_sensitivity():
    """Test sensitivity to weight changes."""
    state = create_test_scenario_1()
    
    print("\n=== Weight Sensitivity Test ===")
    
    # Test different weight values for critical_mass heuristic
    weights_to_test = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    for weight in weights_to_test:
        config = ai.create_balanced_config()
        config.weights['critical_mass'] = weight
        
        agent = ai.MinimaxAgent(player=1, config=config)
        
        start_time = time.time()
        value, move = agent.minimax_search(state, 3)
        end_time = time.time()
        
        print(f"Critical Mass Weight {weight}: Move {move}, Value {value:.2f}, "
              f"Time {end_time-start_time:.4f}s")

def benchmark_performance():
    """Benchmark the AI performance on different scenarios."""
    scenarios = [
        ("Early Game", create_test_scenario_1),
        ("Chain Reaction Setup", create_test_scenario_2),
        ("Complex Mid-Game", create_test_scenario_3)
    ]
    
    print("\n=== Performance Benchmark ===")
    
    config = ai.create_balanced_config()
    config.set_depth(4)
    agent = ai.MinimaxAgent(player=1, config=config)
    
    total_time = 0
    total_nodes = 0
    
    for scenario_name, scenario_func in scenarios:
        state = scenario_func()
        
        start_time = time.time()
        value, move = agent.minimax_search(state, 4)
        end_time = time.time()
        
        stats = agent.get_search_statistics()
        search_time = end_time - start_time
        
        total_time += search_time
        total_nodes += stats['nodes_explored']
        
        print(f"\n{scenario_name}:")
        print(f"  Move: {move}")
        print(f"  Time: {search_time:.4f}s")
        print(f"  Nodes: {stats['nodes_explored']}")
        print(f"  Nodes/sec: {stats['nodes_explored']/search_time:.0f}")
        if 'hit_rate_percent' in stats:
            print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
    
    print(f"\nOverall Performance:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Average nodes/sec: {total_nodes/total_time:.0f}")

def print_board_state(state: core.GameState):
    """Print a visual representation of the board state."""
    print("\nBoard State:")
    print("   ", end="")
    for c in range(state.cols):
        print(f"{c:3}", end="")
    print()
    
    for r in range(state.rows):
        print(f"{r:2} ", end="")
        for c in range(state.cols):
            cell = state.board[r][c]
            if cell.owner is None:
                print("  .", end="")
            else:
                color = "R" if cell.owner == 1 else "B"
                print(f"{cell.count:2}{color}", end="")
        print()

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🧠 Comprehensive Chain Reaction AI Test Suite")
    print("=" * 60)
    
    # Test Scenario 1: Basic position evaluation
    print("\n1️⃣ Testing Basic Position Evaluation")
    state1 = create_test_scenario_1()
    print_board_state(state1)
    
    heuristic_values = test_individual_heuristics(state1)
    print("\nIndividual Heuristic Values (for Player 1):")
    for name, value in heuristic_values.items():
        print(f"  {name.replace('_', ' ').title()}: {value:.2f}")
    
    # Test agent with default config
    agent = ai.MinimaxAgent(player=1)  # Use default config
    overall_eval = agent.evaluate_state(state1)
    print(f"\nOverall Evaluation: {overall_eval:.2f}")
    
    # Test minimax performance
    performance_stats = test_minimax_performance(agent, state1, "Basic Scenario")
    print(f"\nMinimax Performance:")
    print(f"  Chosen move: {performance_stats['chosen_move']}")
    print(f"  Evaluation value: {performance_stats['evaluation_value']:.2f}")
    print(f"  Search time: {performance_stats['search_time']:.4f} seconds")
    print(f"  Nodes explored: {performance_stats['nodes_explored']}")
    print(f"  Alpha-beta cutoffs: {performance_stats['alpha_beta_cutoffs']}")
    if 'hit_rate_percent' in performance_stats:
        print(f"  Table hit rate: {performance_stats['hit_rate_percent']:.1f}%")
    
    # Test Scenario 2: Chain reaction potential
    print("\n\n2️⃣ Testing Chain Reaction Scenario")
    state2 = create_test_scenario_2()
    print_board_state(state2)
    
    heuristic_values2 = test_individual_heuristics(state2)
    print("\nIndividual Heuristic Values (for Player 1):")
    for name, value in heuristic_values2.items():
        print(f"  {name.replace('_', ' ').title()}: {value:.2f}")
    
    # Run all comparison tests
    compare_agent_configurations()
    test_depth_scaling()
    test_individual_heuristic_impact()
    test_weight_sensitivity()
    benchmark_performance()
    
    print("\n✅ All tests completed!")

def quick_test():
    """Quick test for immediate feedback."""
    print("⚡ Quick AI Test")
    print("-" * 30)
    
    # Simple test
    state = create_test_scenario_1()
    agent = ai.MinimaxAgent(player=1)
    
    start_time = time.time()
    move = agent.choose_move(state)
    end_time = time.time()
    
    stats = agent.get_search_statistics()
    
    print(f"✅ Move chosen: {move}")
    print(f"⏱️  Time: {end_time - start_time:.4f} seconds")
    print(f"🧠 Nodes explored: {stats['nodes_explored']}")
    if 'hit_rate_percent' in stats:
        print(f"📊 Hit rate: {stats['hit_rate_percent']:.1f}%")
    
    if end_time - start_time < 0.5:
        print("🚀 Great! Fast response time.")
    elif end_time - start_time < 2.0:
        print("✅ Good response time.")
    else:
        print("⚠️  Slow response - consider reducing depth.")

def main():
    """Main test function with menu."""
    print("Chain Reaction AI Testing Suite")
    print("=" * 40)
    print("Choose test type:")
    print("1. Quick test (fast)")
    print("2. Comprehensive test (thorough)")
    print("3. Performance benchmark")
    print("4. Configuration comparison")
    print("5. Heuristic impact analysis")
    print("6. Depth scaling test")
    print("7. Weight sensitivity test")
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        run_comprehensive_test()
    elif choice == "3":
        benchmark_performance()
    elif choice == "4":
        compare_agent_configurations()
    elif choice == "5":
        test_individual_heuristic_impact()
    elif choice == "6":
        test_depth_scaling()
    elif choice == "7":
        test_weight_sensitivity()
    else:
        print("Invalid choice, running quick test...")
        quick_test()

if __name__ == "__main__":
    main()