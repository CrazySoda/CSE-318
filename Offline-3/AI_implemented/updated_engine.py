"""
engine.py – Updated backend process with configurable AI:
• waits for a 'Human Move:' in gamestate.txt
• replies with its own move using configurable AI agent
• supports different AI configurations
"""

import time, os, sys
import core, ai

FILE = "gamestate.txt"
CONFIG_FILE = "ai_config.txt"

def read_file_waiting_for(header: str) -> list[str]:
    """Block until the file starts with the desired header."""
    while True:
        try:
            with open(FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            time.sleep(0.1)
            continue

        if lines and lines[0].strip() == header:
            return lines[1:]
        time.sleep(0.1)

def write_state(header: str, state: core.GameState):
    """Write game state to file."""
    data = [header] + state.to_lines()
    with open(FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(data))

def load_ai_config() -> ai.AIConfig:
    """Load AI configuration from file or use defaults."""
    try:
        if os.path.exists(CONFIG_FILE):
            config = ai.AIConfig()
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith("depth="):
                    config.set_depth(int(line.split("=")[1]))
                elif line.startswith("preset="):
                    preset_name = line.split("=")[1]
                    if preset_name == "aggressive":
                        config = ai.create_aggressive_config()
                    elif preset_name == "defensive":
                        config = ai.create_defensive_config()
                    elif preset_name == "material_only":
                        config = ai.create_material_only_config()
                    else:
                        config = ai.create_balanced_config()
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
            return ai.AIConfig()
            
    except Exception as e:
        print(f"Error loading AI config: {e}, using defaults")
        return ai.AIConfig()

def save_ai_config(config: ai.AIConfig):
    """Save AI configuration to file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(f"depth={config.depth}\n")
            f.write(f"transposition_table={config.use_transposition_table}\n")
            f.write(f"move_ordering={config.use_move_ordering}\n")
            
            for heuristic, enabled in config.enabled_heuristics.items():
                f.write(f"{heuristic}_enabled={enabled}\n")
            
            for heuristic, weight in config.weights.items():
                f.write(f"{heuristic}_weight={weight}\n")
        
        print(f"Saved AI configuration to {CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving AI config: {e}")

def create_ai_agent(config: ai.AIConfig) -> ai.MinimaxAgent:
    """Create AI agent with the given configuration."""
    agent = ai.MinimaxAgent(player=2, config=config)  # Blue player
    
    # Print configuration summary
    print("AI Agent Configuration:")
    print(f"  Search depth: {config.depth}")
    print(f"  Enabled heuristics: {[h for h, enabled in config.enabled_heuristics.items() if enabled]}")
    print(f"  Active weights: {config.get_active_weights()}")
    
    return agent

def main():
    """Main engine loop."""
    print("AI Engine starting...")
    
    # Load AI configuration
    config = load_ai_config()
    agent = create_ai_agent(config)
    
    # If file doesn't exist, create an empty board
    if not os.path.exists(FILE):
        empty = core.GameState(rows=9, cols=6)
        write_state("AI Move:", empty)
        print("Created initial game state file")

    print("AI Engine ready – waiting for human moves…")
    
    move_count = 0
    
    while True:
        try:
            # Wait for human move
            board_lines = read_file_waiting_for("Human Move:")
            state = core.GameState.from_file(board_lines)
            state.current_player = 2  # AI's turn (Blue)

            if state.game_over:
                write_state("AI Move:", state)
                winner = state.get_winner()
                if winner == 1:
                    print("Human won – game over.")
                else:
                    print("Game ended in draw – exiting.")
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
                if winner == 2:
                    print("AI wins – game over!")
                else:
                    print("Game ended – exiting.")
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