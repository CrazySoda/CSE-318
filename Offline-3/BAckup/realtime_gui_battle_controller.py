#!/usr/bin/env python3
"""
Real-time GUI Battle Controller
Shows actual GUI battles in real-time so you can watch what's happening
Uses ChainReactionGUI class directly and removes artificial time limits
"""

import time
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import pygame

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

class RealtimeGUIBattleController:
    """Shows GUI battles in real-time with enhanced statistics display."""
    
    def __init__(self):
        self.available_configs = {
            'balanced': ai.create_balanced_config,
            'aggressive': ai.create_aggressive_config,
            'defensive': ai.create_defensive_config,
            'tactical': ai.create_tactical_config,
            'strategic': ai.create_strategic_config,
            'fast': ai.create_fast_config,
            'material_only': ai.create_material_only_config,
        }
        
        # Battle control settings
        self.game_speed_multiplier = 1.0  # 1.0 = normal, 0.5 = half speed, 2.0 = double speed
        self.show_detailed_stats = True
        self.auto_advance = False
        
        # Current GUI instance
        self.gui_instance: Optional[gui.ChainReactionGUI] = None
        
        # Battle tracking
        self.current_battle_result = None
        self.games_completed = 0
        self.target_games = 0
    
    def setup_gui_for_battle(self, config1_name: str, config2_name: str, auto_restart: bool = False) -> gui.ChainReactionGUI:
        """Set up GUI instance for real-time battle viewing using existing GUI class."""
        # Create GUI instance
        gui_instance = gui.ChainReactionGUI()
        
        # Configure AI vs AI mode using existing GUI methods
        gui_instance.ai_player1_config = self.available_configs[config1_name]()
        gui_instance.ai_player2_config = self.available_configs[config2_name]()
        gui_instance.mode = gui.MODE_AI_VS_AI
        
        # Set battle speed
        gui_instance.ai_move_delay = max(0.1, 1.0 / self.game_speed_multiplier)
        gui_instance.auto_restart = auto_restart
        
        # Reset match stats
        gui_instance.match_stats = gui.MatchStats()
        
        # Start the AI vs AI match using existing method
        gui_instance._start_ai_vs_ai()
        
        return gui_instance
    
    def draw_enhanced_battle_overlay(self, gui_instance: gui.ChainReactionGUI, 
                                   current_game: int, total_games: int,
                                   config1_name: str, config2_name: str, 
                                   wins1: int, wins2: int, draws: int):
        """Draw enhanced battle statistics overlay on the GUI."""
        if not gui_instance.screen:
            return
        
        # Enhanced overlay with more information
        overlay_height = 140
        overlay_rect = (0, gui.HEIGHT - overlay_height, gui.WIDTH, overlay_height)
        overlay_surf = pygame.Surface((gui.WIDTH, overlay_height))
        overlay_surf.set_alpha(240)  # More opaque for better readability
        overlay_surf.fill(gui.COLORS['bg_menu'])
        gui_instance.screen.blit(overlay_surf, (0, gui.HEIGHT - overlay_height))
        
        # Draw border
        pygame.draw.rect(gui_instance.screen, gui.COLORS['accent'], overlay_rect, 3)
        
        # Battle title
        title_text = f"REAL-TIME BATTLE: {config1_name} vs {config2_name}"
        title_surf = gui_instance.fonts['header'].render(title_text, True, gui.COLORS['text_primary'])
        title_x = (gui.WIDTH - title_surf.get_width()) // 2
        gui_instance.screen.blit(title_surf, (title_x, gui.HEIGHT - overlay_height + 8))
        
        # Game progress
        progress_text = f"Game {current_game}/{total_games}"
        progress_surf = gui_instance.fonts['normal'].render(progress_text, True, gui.COLORS['text_secondary'])
        gui_instance.screen.blit(progress_surf, (10, gui.HEIGHT - overlay_height + 35))
        
        # Current standings with colors
        p1_color = gui.COLORS['player1']
        p2_color = gui.COLORS['player2']
        
        standings_text = f"Score: "
        standings_surf = gui_instance.fonts['normal'].render(standings_text, True, gui.COLORS['text_primary'])
        gui_instance.screen.blit(standings_surf, (10, gui.HEIGHT - overlay_height + 55))
        
        # Player 1 score in red
        p1_text = f"{config1_name} {wins1}"
        p1_surf = gui_instance.fonts['normal'].render(p1_text, True, p1_color)
        p1_x = 10 + standings_surf.get_width()
        gui_instance.screen.blit(p1_surf, (p1_x, gui.HEIGHT - overlay_height + 55))
        
        # Separator
        sep_text = " - "
        sep_surf = gui_instance.fonts['normal'].render(sep_text, True, gui.COLORS['text_primary'])
        sep_x = p1_x + p1_surf.get_width()
        gui_instance.screen.blit(sep_surf, (sep_x, gui.HEIGHT - overlay_height + 55))
        
        # Player 2 score in blue
        p2_text = f"{wins2} {config2_name}"
        p2_surf = gui_instance.fonts['normal'].render(p2_text, True, p2_color)
        p2_x = sep_x + sep_surf.get_width()
        gui_instance.screen.blit(p2_surf, (p2_x, gui.HEIGHT - overlay_height + 55))
        
        # Draws if any
        if draws > 0:
            draws_text = f" (Draws: {draws})"
            draws_surf = gui_instance.fonts['normal'].render(draws_text, True, gui.COLORS['text_secondary'])
            draws_x = p2_x + p2_surf.get_width()
            gui_instance.screen.blit(draws_surf, (draws_x, gui.HEIGHT - overlay_height + 55))
        
        # Win percentages
        total_completed = wins1 + wins2 + draws
        if total_completed > 0:
            win_rate1 = wins1 / total_completed * 100
            win_rate2 = wins2 / total_completed * 100
            rates_text = f"Win Rates: {win_rate1:.1f}% - {win_rate2:.1f}%"
            rates_surf = gui_instance.fonts['small'].render(rates_text, True, gui.COLORS['text_secondary'])
            gui_instance.screen.blit(rates_surf, (10, gui.HEIGHT - overlay_height + 75))
        
        # Auto-restart status
        auto_status = "Auto-Restart: ON" if gui_instance.auto_restart else "Auto-Restart: OFF"
        auto_color = gui.COLORS['success'] if gui_instance.auto_restart else gui.COLORS['warning']
        auto_surf = gui_instance.fonts['small'].render(auto_status, True, auto_color)
        gui_instance.screen.blit(auto_surf, (10, gui.HEIGHT - overlay_height + 95))
        
        # Enhanced controls
        controls_text = "SPACE: Pause | ↑↓: Speed | A: Auto-Restart | R: Restart Game | ESC: Stop"
        controls_surf = gui_instance.fonts['small'].render(controls_text, True, gui.COLORS['text_muted'])
        controls_x = gui.WIDTH - controls_surf.get_width() - 10
        gui_instance.screen.blit(controls_surf, (controls_x, gui.HEIGHT - overlay_height + 115))
        
        # Current game stats
        moves_text = f"Moves: {gui_instance.match_stats.current_game_moves}"
        moves_surf = gui_instance.fonts['small'].render(moves_text, True, gui.COLORS['text_secondary'])
        gui_instance.screen.blit(moves_surf, (gui.WIDTH - 120, gui.HEIGHT - overlay_height + 35))
        
        # Game duration
        if hasattr(gui_instance, 'current_game_start_time') and gui_instance.current_game_start_time > 0:
            duration = time.time() - gui_instance.current_game_start_time
            duration_text = f"Time: {duration:.1f}s"
            duration_surf = gui_instance.fonts['small'].render(duration_text, True, gui.COLORS['text_secondary'])
            gui_instance.screen.blit(duration_surf, (gui.WIDTH - 120, gui.HEIGHT - overlay_height + 55))
        
        # Speed indicator
        speed_text = f"Speed: {self.game_speed_multiplier:.1f}x"
        speed_surf = gui_instance.fonts['small'].render(speed_text, True, gui.COLORS['text_secondary'])
        gui_instance.screen.blit(speed_surf, (gui.WIDTH - 120, gui.HEIGHT - overlay_height + 75))
    
    def handle_battle_controls(self, gui_instance: gui.ChainReactionGUI) -> Tuple[bool, bool, bool]:
        """
        Handle real-time battle controls.
        Returns: (continue_battle, advance_to_next_game, restart_current_game)
        """
        continue_battle = True
        advance_to_next = False
        restart_game = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    continue_battle = False
                elif event.key == pygame.K_SPACE:
                    if not gui_instance.animating:  # Only allow pause when not animating
                        gui_instance.ai_vs_ai_running = not gui_instance.ai_vs_ai_running
                        print(f"Battle {'resumed' if gui_instance.ai_vs_ai_running else 'paused'}")
                elif event.key == pygame.K_UP:
                    self.game_speed_multiplier = min(5.0, self.game_speed_multiplier * 1.5)
                    gui_instance.ai_move_delay = max(0.05, 1.0 / self.game_speed_multiplier)
                    print(f"Speed: {self.game_speed_multiplier:.1f}x")
                elif event.key == pygame.K_DOWN:
                    self.game_speed_multiplier = max(0.2, self.game_speed_multiplier / 1.5)
                    gui_instance.ai_move_delay = max(0.05, 1.0 / self.game_speed_multiplier)
                    print(f"Speed: {self.game_speed_multiplier:.1f}x")
                elif event.key == pygame.K_RETURN:
                    advance_to_next = True
                elif event.key == pygame.K_a:
                    if not gui_instance.animating:  # Only allow toggle when not animating
                        gui_instance.auto_restart = not gui_instance.auto_restart
                        print(f"Auto-restart: {'ON' if gui_instance.auto_restart else 'OFF'}")
                elif event.key == pygame.K_r:
                    if not gui_instance.animating:  # Only allow restart when not animating
                        restart_game = True
        
        return continue_battle, advance_to_next, restart_game
    
    def run_single_battle_game(self, gui_instance: gui.ChainReactionGUI, 
                              game_id: int, total_games: int,
                              config1_name: str, config2_name: str,
                              wins1: int, wins2: int, draws: int) -> Tuple[Optional[int], int, float]:
        """Run a single game with real-time GUI display using existing GUI methods."""
        print(f"\n🎮 Starting Game {game_id}: {config1_name} vs {config2_name}")
        
        # Reset for new game using existing GUI methods
        gui_instance.state.reset()
        gui_instance.visual_state.reset()
        gui_instance.match_stats.current_game_moves = 0
        gui_instance.current_game_start_time = time.time()
        gui_instance.ai_vs_ai_running = True
        gui_instance.animating = False
        
        # Clear animations using existing methods
        gui_instance.animations.clear()
        gui_instance.explosion_animations.clear()
        gui_instance.orb_placement_animations.clear()
        gui_instance.animation_queue.clear()
        
        game_start_time = time.time()
        last_move_count = 0
        
        # NO ARTIFICIAL MOVE LIMIT - Game continues until there's a winner
        while True:
            # Handle events and controls
            continue_battle, advance_to_next, restart_game = self.handle_battle_controls(gui_instance)
            
            if not continue_battle:
                return None, gui_instance.match_stats.current_game_moves, time.time() - game_start_time
            
            if restart_game:
                print(f"🔄 Restarting Game {game_id}")
                # Reset everything and continue
                gui_instance.state.reset()
                gui_instance.visual_state.reset()
                gui_instance.match_stats.current_game_moves = 0
                gui_instance.current_game_start_time = time.time()
                gui_instance.ai_vs_ai_running = True
                gui_instance.animating = False
                gui_instance.animations.clear()
                gui_instance.explosion_animations.clear()
                gui_instance.orb_placement_animations.clear()
                gui_instance.animation_queue.clear()
                game_start_time = time.time()
                last_move_count = 0
                continue
            
            if advance_to_next and gui_instance.state.game_over:
                break
            
            # Check for natural game end
            if gui_instance.state.game_over:
                if self.auto_advance or gui_instance.auto_restart:
                    time.sleep(2)  # Brief pause to see result
                    break
                else:
                    # Wait for user to advance manually
                    pass
            
            # Update AI using existing GUI method
            if gui_instance.ai_vs_ai_running and not gui_instance.state.game_over:
                gui_instance.update_ai_vs_ai()
            
            # Update animations using existing GUI method
            gui_instance.update_animations()
            
            # Update legacy animations for compatibility
            now = pygame.time.get_ticks()
            gui_instance.animations[:] = [anim for anim in gui_instance.animations
                                         if now - anim.start_time < gui.EXPLOSION_DURATION_MS]
            
            # Render the game using existing GUI methods
            gui_instance.screen.fill(gui.COLORS['bg_primary'])
            gui_instance._draw_grid()
            gui_instance._draw_orbs()
            gui_instance._draw_explosions()
            
            # Draw enhanced battle overlay instead of regular UI
            self.draw_enhanced_battle_overlay(gui_instance, game_id, total_games, 
                                            config1_name, config2_name, wins1, wins2, draws)
            
            pygame.display.flip()
            gui_instance.clock.tick(gui.FPS)
            
            # Progress indicator (less frequent to avoid spam)
            current_moves = gui_instance.match_stats.current_game_moves
            if current_moves > last_move_count and current_moves % 50 == 0:
                print(f"  Move {current_moves}... (Game continues until winner)")
                last_move_count = current_moves
        
        game_duration = time.time() - game_start_time
        winner = gui_instance.state.get_winner() if gui_instance.state.game_over else None
        moves = gui_instance.match_stats.current_game_moves
        
        if winner:
            winner_name = config1_name if winner == 1 else config2_name
            print(f"🏁 Game {game_id} ended: Winner = {winner_name} (Player {winner}) in {moves} moves ({game_duration:.1f}s)")
        else:
            print(f"🏁 Game {game_id} ended: Draw in {moves} moves ({game_duration:.1f}s)")
        
        return winner, moves, game_duration
    
    def battle_configs_realtime(self, config1_name: str, config2_name: str, 
                               num_games: int = 10, show_each_game: bool = True) -> BattleResult:
        """Battle two AI configurations with real-time GUI display."""
        print(f"\n🎮 REAL-TIME BATTLE: {config1_name} vs {config2_name}")
        print(f"Running {num_games} games with live GUI display")
        print("🚫 NO ARTIFICIAL TIME LIMITS - Games continue until there's a winner!")
        if self.auto_advance:
            print("🔄 Auto-advance: ON (games will run automatically)")
        print("="*70)
        print("Controls during battle:")
        print("  SPACE: Pause/Resume")
        print("  ↑/↓: Increase/Decrease speed")
        print("  A: Toggle auto-restart")
        print("  R: Restart current game")
        print("  ENTER: Advance to next game (when current game ends)")
        print("  ESC: Stop battle")
        print("="*70)
        
        if config1_name not in self.available_configs:
            raise ValueError(f"Unknown config: {config1_name}")
        if config2_name not in self.available_configs:
            raise ValueError(f"Unknown config: {config2_name}")
        
        # Initialize results tracking
        player1_wins = 0
        player2_wins = 0
        draws = 0
        individual_results = []
        
        # Show configuration details
        self.show_config_comparison(config1_name, config2_name)
        
        input("\nPress ENTER to start the battle...")
        
        # Create a single GUI instance that we'll reuse
        gui_instance = self.setup_gui_for_battle(config1_name, config2_name, auto_restart=False)
        self.gui_instance = gui_instance
        
        try:
            for game_num in range(1, num_games + 1):
                if show_each_game:
                    # Run the game with real-time display
                    winner, moves, duration = self.run_single_battle_game(
                        gui_instance, game_num, num_games,
                        config1_name, config2_name, 
                        player1_wins, player2_wins, draws
                    )
                    
                    if winner is None:  # User cancelled
                        break
                    
                    individual_results.append((winner, moves, duration))
                    
                    # Update standings
                    if winner == 1:
                        player1_wins += 1
                    elif winner == 2:
                        player2_wins += 1
                    else:
                        draws += 1
                    
                    # Show progress
                    print(f"\nCurrent standings after {game_num} games:")
                    print(f"  {config1_name}: {player1_wins} wins")
                    print(f"  {config2_name}: {player2_wins} wins")
                    print(f"  Draws: {draws}")
                    
                    if not self.auto_advance and game_num < num_games:
                        choice = input(f"\nContinue to game {game_num + 1}? (y/n/auto): ").strip().lower()
                        if choice == 'n':
                            break
                        elif choice == 'auto':
                            self.auto_advance = True
        
        finally:
            # Clean up GUI instance
            if hasattr(gui_instance, 'screen') and gui_instance.screen:
                pygame.quit()
                pygame.init()  # Re-initialize for next battle if needed
        
        # Create final result
        total_completed = len(individual_results)
        result = BattleResult(
            config1_name=config1_name,
            config2_name=config2_name,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_games=total_completed,
            individual_results=individual_results
        )
        
        # Show final results
        self.show_final_results(result)
        
        return result
    
    def show_config_comparison(self, config1_name: str, config2_name: str):
        """Show detailed comparison of the two configurations."""
        print(f"\n📊 CONFIGURATION COMPARISON")
        print("="*60)
        
        config1 = self.available_configs[config1_name]()
        config2 = self.available_configs[config2_name]()
        
        print(f"{'Attribute':<20} {'Player 1 (' + config1_name + ')':<25} {'Player 2 (' + config2_name + ')':<25}")
        print("-" * 70)
        print(f"{'Depth':<20} {config1.depth:<25} {config2.depth:<25}")
        print(f"{'Timeout':<20} {config1.timeout:<25} {config2.timeout:<25}")
        print(f"{'Explosion Limit':<20} {config1.explosion_limit:<25} {config2.explosion_limit:<25}")
        
        print(f"\nHeuristic Weights:")
        all_heuristics = set(config1.weights.keys()) | set(config2.weights.keys())
        for heuristic in sorted(all_heuristics):
            enabled1 = config1.enabled_heuristics.get(heuristic, False)
            enabled2 = config2.enabled_heuristics.get(heuristic, False)
            weight1 = config1.weights.get(heuristic, 0) if enabled1 else "Disabled"
            weight2 = config2.weights.get(heuristic, 0) if enabled2 else "Disabled"
            heur_name = heuristic.replace('_', ' ').title()
            print(f"  {heur_name:<18} {str(weight1):<25} {str(weight2):<25}")
    
    def show_final_results(self, result: BattleResult):
        """Show comprehensive final results."""
        print(f"\n🏆 FINAL BATTLE RESULTS")
        print("="*70)
        print(f"Battle: {result.config1_name} vs {result.config2_name}")
        print(f"Games played: {result.total_games}")
        print()
        print(f"📈 OVERALL WINNER: ", end="")
        if result.player1_wins > result.player2_wins:
            print(f"{result.config1_name} ({result.player1_win_rate:.1%} win rate)")
        elif result.player2_wins > result.player1_wins:
            print(f"{result.config2_name} ({result.player2_win_rate:.1%} win rate)")
        else:
            print("TIE")
        
        print(f"\n📊 DETAILED STATISTICS:")
        print(f"  {result.config1_name} (Player 1): {result.player1_wins} wins ({result.player1_win_rate:.1%})")
        print(f"  {result.config2_name} (Player 2): {result.player2_wins} wins ({result.player2_win_rate:.1%})")
        print(f"  Draws: {result.draws} ({result.draws/result.total_games:.1%})")
        
        if result.individual_results:
            moves = [r[1] for r in result.individual_results]
            durations = [r[2] for r in result.individual_results]
            print(f"\n⚡ GAME STATISTICS:")
            print(f"  Average moves: {sum(moves)/len(moves):.1f}")
            print(f"  Move range: {min(moves)} - {max(moves)}")
            print(f"  Average duration: {sum(durations)/len(durations):.1f}s")
            print(f"  Total battle time: {sum(durations):.1f}s")
            print(f"  Longest game: {max(moves)} moves")
            
            print(f"\n🎮 INDIVIDUAL GAME RESULTS:")
            for i, (winner, moves, duration) in enumerate(result.individual_results[:10], 1):
                winner_name = result.config1_name if winner == 1 else result.config2_name if winner == 2 else "Draw"
                print(f"  Game {i}: {winner_name} in {moves} moves ({duration:.1f}s)")
            if len(result.individual_results) > 10:
                print(f"  ... and {len(result.individual_results) - 10} more games")

def main():
    """Main function with enhanced menu."""
    controller = RealtimeGUIBattleController()
    
    print("🎮 REAL-TIME GUI BATTLE CONTROLLER")
    print("Watch AI battles live with full GUI display!")
    print("🚫 NO ARTIFICIAL TIME LIMITS - Games run until completion!")
    print("="*60)
    
    available_configs = list(controller.available_configs.keys())
    print(f"Available configurations: {', '.join(available_configs)}")
    
    while True:
        print(f"\nChoose an option:")
        print("1. Watch live battle (2 configs)")
        print("2. Quick demo (aggressive vs balanced)")
        print("3. Configuration comparison")
        print("4. Settings")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (0-4): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == "0":
            break
        
        elif choice == "1":
            config1 = input(f"Enter first config ({'/'.join(available_configs)}): ").strip()
            config2 = input(f"Enter second config ({'/'.join(available_configs)}): ").strip()
            num_games = int(input("Enter number of games (default 5): ").strip() or "5")
            
            if config1 in available_configs and config2 in available_configs:
                controller.battle_configs_realtime(config1, config2, num_games)
            else:
                print("Invalid configuration names!")
        
        elif choice == "2":
            print("\n🔄 Demo: aggressive vs balanced")
            print("Perfect for seeing the difference in AI behavior!")
            print("🚫 Games will run until there's a clear winner (no time limits)")
            # Enable auto-advance for demo
            original_auto_advance = controller.auto_advance
            controller.auto_advance = True
            controller.battle_configs_realtime("aggressive", "balanced", num_games=3)
            # Restore original setting
            controller.auto_advance = original_auto_advance
        
        elif choice == "3":
            config1 = input(f"Enter first config ({'/'.join(available_configs)}): ").strip()
            config2 = input(f"Enter second config ({'/'.join(available_configs)}): ").strip()
            
            if config1 in available_configs and config2 in available_configs:
                controller.show_config_comparison(config1, config2)
            else:
                print("Invalid configuration names!")
        
        elif choice == "4":
            print("\n⚙️ Settings:")
            print(f"1. Game speed: {controller.game_speed_multiplier:.1f}x")
            print(f"2. Auto-advance: {'ON' if controller.auto_advance else 'OFF'}")
            print("3. Battle info")
            
            setting = input("Change setting (1-3): ").strip()
            if setting == "1":
                speed = float(input("Enter speed multiplier (0.2-5.0): ").strip() or "1.0")
                controller.game_speed_multiplier = max(0.2, min(5.0, speed))
            elif setting == "2":
                controller.auto_advance = not controller.auto_advance
                print(f"Auto-advance: {'ON' if controller.auto_advance else 'OFF'}")
            elif setting == "3":
                print("\n📋 Battle Information:")
                print("- Games run until there's a clear winner (no artificial time limits)")
                print("- Auto-restart can be toggled during battles with 'A' key")
                print("- Use 'R' to restart current game, 'SPACE' to pause/resume")
                print("- Speed can be adjusted in real-time with ↑/↓ keys")
                print("- All games are logged with move counts and durations")
                input("\nPress ENTER to continue...")
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()