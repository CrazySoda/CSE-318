"""
gui.py – Updated GUI with AI configuration menu
Supports TWO modes plus AI configuration:
1. PVP   (two humans on one screen)
2. PVAI  (Human vs configurable AI)
3. AI Config menu for selecting heuristics and tuning
"""

from __future__ import annotations
import sys, time, os, pygame
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import core
import ai

# ──────────────────────────────────────────────────────────────────────
# File-protocol helpers (used only in PVAI mode)
# ──────────────────────────────────────────────────────────────────────
FILE = "gamestate.txt"

def _write_state(header: str, state: core.GameState):
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(state.to_lines()))

def _read_until(header: str) -> core.GameState:
    while True:
        with open(FILE, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lines and lines[0].strip() == header:
            return core.GameState.from_file(lines[1:])
        time.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────
# GUI constants
# ──────────────────────────────────────────────────────────────────────
CELL_SIZE = 100
WIDTH, HEIGHT = 6 * CELL_SIZE, 9 * CELL_SIZE
FPS = 60
GRAY  = (200, 200, 200)
BLACK = (25, 25, 25)
WHITE = (250, 250, 250)
GREEN = (0, 200, 0)
LIGHT_GRAY = (150, 150, 150)
DARK_GRAY = (100, 100, 100)
EXPLOSION_MS = 250

MODE_PVP  = "PVP"
MODE_PVAI = "PVAI"

@dataclass
class ExplosionAnim:
    row: int
    col: int
    start_time: int
    radius: int = field(init=False, default=0)

class ChainReactionGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chain Reaction - Configurable AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)

        self.state = core.GameState(rows=9, cols=6)
        self.mode = None
        self.animations: List[ExplosionAnim] = []
        
        # AI Configuration
        self.ai_config = ai.AIConfig()
        self.config_menu_active = False
        self.selected_config_item = 0
        
        # Available AI presets
        self.ai_presets = {
            "Balanced": ai.create_balanced_config,
            "Aggressive": ai.create_aggressive_config,
            "Defensive": ai.create_defensive_config,
            "Material Only": ai.create_material_only_config
        }

    def _main_menu(self):
        """Main menu for mode selection."""
        choosing = True
        while choosing:
            self.screen.fill(BLACK)
            
            # Title
            title = self.font.render("Chain Reaction - AI Configuration", True, WHITE)
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//4))
            
            # Menu options
            options = [
                "1 - Two Players (PVP)",
                "2 - Play vs AI (PVAI)", 
                "3 - Configure AI",
                "ESC - Quit"
            ]
            
            for i, option in enumerate(options):
                color = WHITE if i < 3 else GRAY
                text = self.small_font.render(option, True, color)
                y_pos = HEIGHT//2 + i * 40
                self.screen.blit(text, (WIDTH//2 - text.get_width()//2, y_pos))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.mode = MODE_PVP
                        choosing = False
                    elif event.key == pygame.K_2:
                        self.mode = MODE_PVAI
                        choosing = False
                    elif event.key == pygame.K_3:
                        self._ai_config_menu()
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

        self._reset_board()

    def _ai_config_menu(self):
        """AI configuration menu."""
        configuring = True
        scroll_offset = 0
        
        while configuring:
            self.screen.fill(BLACK)
            
            # Title
            title = self.font.render("AI Configuration", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            y_pos = 50
            line_height = 25
            
            # Current configuration display
            config_lines = [
                "=== CURRENT CONFIGURATION ===",
                f"Search Depth: {self.ai_config.depth}",
                f"Transposition Table: {'ON' if self.ai_config.use_transposition_table else 'OFF'}",
                f"Move Ordering: {'ON' if self.ai_config.use_move_ordering else 'OFF'}",
                "",
                "=== HEURISTICS ===",
            ]
            
            for heuristic, enabled in self.ai_config.enabled_heuristics.items():
                weight = self.ai_config.weights[heuristic]
                status = f"{'✓' if enabled else '✗'}"
                config_lines.append(f"{status} {heuristic.replace('_', ' ').title()}: {weight:.1f}")
            
            config_lines.extend([
                "",
                "=== CONTROLS ===",
                "H - Toggle heuristics on/off",
                "W - Adjust weights",
                "D - Change search depth",
                "P - Load preset configuration",
                "R - Reset to defaults",
                "T - Test current configuration",
                "B - Back"
            ])
            
            # Display configuration
            for i, line in enumerate(config_lines):
                if line.startswith("==="):
                    color = GREEN
                elif line.startswith("✓"):
                    color = GREEN
                elif line.startswith("✗"):
                    color = GRAY
                else:
                    color = WHITE
                
                text = self.small_font.render(line, True, color)
                self.screen.blit(text, (10, y_pos + i * line_height))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self._toggle_heuristics_menu()
                    elif event.key == pygame.K_w:
                        self._adjust_weights_menu()
                    elif event.key == pygame.K_d:
                        self._change_depth_menu()
                    elif event.key == pygame.K_p:
                        self._preset_menu()
                    elif event.key == pygame.K_r:
                        self.ai_config = ai.AIConfig()  # Reset to defaults
                    elif event.key == pygame.K_t:
                        self._test_ai_config()
                    elif event.key == pygame.K_b:
                        configuring = False

    def _toggle_heuristics_menu(self):
        """Menu for toggling individual heuristics."""
        toggling = True
        selected = 0
        heuristic_names = list(self.ai_config.enabled_heuristics.keys())
        
        while toggling:
            self.screen.fill(BLACK)
            
            title = self.font.render("Toggle Heuristics (SPACE to toggle, ENTER to confirm)", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            y_pos = 60
            for i, heuristic in enumerate(heuristic_names):
                enabled = self.ai_config.enabled_heuristics[heuristic]
                status = "✓ ON " if enabled else "✗ OFF"
                color = GREEN if enabled else GRAY
                
                if i == selected:
                    # Highlight selected item
                    pygame.draw.rect(self.screen, DARK_GRAY, (5, y_pos + i * 30 - 2, WIDTH - 10, 26))
                
                text = self.small_font.render(f"{status} {heuristic.replace('_', ' ').title()}", True, color)
                self.screen.blit(text, (10, y_pos + i * 30))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(heuristic_names)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(heuristic_names)
                    elif event.key == pygame.K_SPACE:
                        heuristic = heuristic_names[selected]
                        current = self.ai_config.enabled_heuristics[heuristic]
                        self.ai_config.enabled_heuristics[heuristic] = not current
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        toggling = False

    def _adjust_weights_menu(self):
        """Menu for adjusting heuristic weights."""
        adjusting = True
        selected = 0
        heuristic_names = list(self.ai_config.weights.keys())
        
        while adjusting:
            self.screen.fill(BLACK)
            
            title = self.font.render("Adjust Weights (← → to change, ENTER to confirm)", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            instructions = self.small_font.render("Use LEFT/RIGHT arrows to adjust values by 0.5", True, GRAY)
            self.screen.blit(instructions, (10, 40))
            
            y_pos = 80
            for i, heuristic in enumerate(heuristic_names):
                weight = self.ai_config.weights[heuristic]
                
                if i == selected:
                    pygame.draw.rect(self.screen, DARK_GRAY, (5, y_pos + i * 30 - 2, WIDTH - 10, 26))
                
                text = self.small_font.render(f"{heuristic.replace('_', ' ').title()}: {weight:.1f}", True, WHITE)
                self.screen.blit(text, (10, y_pos + i * 30))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(heuristic_names)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(heuristic_names)
                    elif event.key == pygame.K_LEFT:
                        heuristic = heuristic_names[selected]
                        self.ai_config.weights[heuristic] = max(0.0, self.ai_config.weights[heuristic] - 0.5)
                    elif event.key == pygame.K_RIGHT:
                        heuristic = heuristic_names[selected]
                        self.ai_config.weights[heuristic] = min(10.0, self.ai_config.weights[heuristic] + 0.5)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        adjusting = False

    def _change_depth_menu(self):
        """Menu for changing search depth."""
        changing = True
        
        while changing:
            self.screen.fill(BLACK)
            
            title = self.font.render("Change Search Depth", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            current_text = self.font.render(f"Current Depth: {self.ai_config.depth}", True, WHITE)
            self.screen.blit(current_text, (10, 60))
            
            instructions = [
                "← → to adjust depth",
                "Higher depth = stronger but slower",
                f"Range: {ai.MIN_DEPTH} to {ai.MAX_DEPTH}",
                "ENTER to confirm"
            ]
            
            for i, instruction in enumerate(instructions):
                text = self.small_font.render(instruction, True, GRAY)
                self.screen.blit(text, (10, 100 + i * 25))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.ai_config.set_depth(self.ai_config.depth - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.ai_config.set_depth(self.ai_config.depth + 1)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        changing = False

    def _preset_menu(self):
        """Menu for loading preset configurations."""
        selecting = True
        selected = 0
        preset_names = list(self.ai_presets.keys())
        
        while selecting:
            self.screen.fill(BLACK)
            
            title = self.font.render("Load Preset Configuration", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            y_pos = 60
            for i, preset_name in enumerate(preset_names):
                if i == selected:
                    pygame.draw.rect(self.screen, DARK_GRAY, (5, y_pos + i * 40 - 2, WIDTH - 10, 36))
                
                text = self.font.render(preset_name, True, WHITE)
                self.screen.blit(text, (10, y_pos + i * 40))
                
                # Show description
                descriptions = {
                    "Balanced": "Default balanced strategy",
                    "Aggressive": "Focuses on immediate threats and chain reactions",
                    "Defensive": "Emphasizes territory control and material advantage",
                    "Material Only": "Uses only material advantage heuristic"
                }
                
                desc = self.small_font.render(descriptions.get(preset_name, ""), True, GRAY)
                self.screen.blit(desc, (10, y_pos + i * 40 + 20))
            
            instructions = self.small_font.render("UP/DOWN to select, ENTER to load, ESC to cancel", True, WHITE)
            self.screen.blit(instructions, (10, HEIGHT - 40))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(preset_names)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(preset_names)
                    elif event.key == pygame.K_RETURN:
                        preset_name = preset_names[selected]
                        self.ai_config = self.ai_presets[preset_name]()
                        selecting = False
                    elif event.key == pygame.K_ESCAPE:
                        selecting = False

    def _test_ai_config(self):
        """Test the current AI configuration."""
        testing = True
        
        # Create test state
        test_state = core.GameState(rows=9, cols=6)
        test_state.board[2][3].owner = 1
        test_state.board[2][3].count = 2
        test_state.board[4][5].owner = 2
        test_state.board[4][5].count = 1
        test_state.current_player = 1
        
        # Test AI
        test_agent = ai.MinimaxAgent(player=1, config=self.ai_config)
        
        import time
        start_time = time.time()
        move = test_agent.choose_move(test_state)
        end_time = time.time()
        
        stats = test_agent.get_search_statistics()
        
        while testing:
            self.screen.fill(BLACK)
            
            title = self.font.render("AI Configuration Test Results", True, WHITE)
            self.screen.blit(title, (10, 10))
            
            results = [
                f"Move chosen: {move}",
                f"Time taken: {end_time - start_time:.4f} seconds",
                f"Nodes explored: {stats['nodes_explored']}",
                f"Alpha-beta cutoffs: {stats['alpha_beta_cutoffs']}",
                f"Search depth: {stats['search_depth']}",
                f"Table hit rate: {stats.get('hit_rate_percent', 0):.1f}%",
                "",
                "Active heuristics:",
            ]
            
            results.extend([f"  • {h}" for h in stats['enabled_heuristics']])
            results.append("")
            results.append("Heuristic weights:")
            
            for heuristic, weight in stats['heuristic_weights'].items():
                results.append(f"  • {heuristic.replace('_', ' ').title()}: {weight:.1f}")
            
            results.extend(["", "Press any key to continue"])
            
            y_pos = 50
            for result in results:
                color = WHITE if not result.startswith("  ") else GRAY
                text = self.small_font.render(result, True, color)
                self.screen.blit(text, (10, y_pos))
                y_pos += 22
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    testing = False

    def _toggle_mode(self):
        """Toggle between PVP and PVAI modes."""
        self.mode = MODE_PVP if self.mode == MODE_PVAI else MODE_PVAI
        self._reset_board()

    def _reset_board(self):
        """Reset the game board."""
        self.state.reset()
        self.animations.clear()
        if self.mode == MODE_PVAI and not os.path.exists(FILE):
            _write_state("AI Move:", self.state)

    def _write_human_move(self):
        """Write human move to file for AI engine."""
        _write_state("Human Move:", self.state)

    def _wait_ai(self):
        """Wait for AI move from engine."""
        self.state = _read_until("AI Move:")

    def handle_click(self, mx: int, my: int):
        """Handle mouse clicks on the game board."""
        if self.state.game_over or self.animations:
            return
        
        r, c = my // CELL_SIZE, mx // CELL_SIZE

        try:
            exploded = self.state.apply_move(r, c)
        except ValueError:
            return

        now = pygame.time.get_ticks()
        for er, ec in exploded:
            self.animations.append(ExplosionAnim(er, ec, now))

        # If PVAI and Red (human) just moved, hand off to AI
        if self.mode == MODE_PVAI and self.state.current_player == 2 and not self.state.game_over:
            self._write_human_move()
            self._wait_ai()

    def _draw_grid(self):
        """Draw the game grid."""
        for r in range(1, 9):
            pygame.draw.line(self.screen, GRAY, (0, r*CELL_SIZE), (WIDTH, r*CELL_SIZE), 2)
        for c in range(1, 6):
            pygame.draw.line(self.screen, GRAY, (c*CELL_SIZE, 0), (c*CELL_SIZE, HEIGHT), 2)

    def _draw_orbs(self):
        """Draw orbs on the board."""
        offset, rad = CELL_SIZE*0.25, CELL_SIZE*0.18
        for r in range(9):
            for c in range(6):
                cell = self.state.board[r][c]
                if not cell.owner:
                    continue
                color = core.PLAYER_COLOR[cell.owner]
                cx, cy = c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2
                positions: List[Tuple[float, float]] = []
                if cell.count == 1:
                    positions.append((cx, cy))
                elif cell.count == 2:
                    positions.extend([(cx-offset, cy), (cx+offset, cy)])
                else:
                    positions.extend([(cx-offset, cy-offset), (cx+offset, cy-offset), (cx, cy+offset)])
                    if cell.count == 4:
                        positions.append((cx, cy))
                for px, py in positions[:cell.count]:
                    pygame.draw.circle(self.screen, color, (int(px), int(py)), int(rad))

    def _draw_ui(self):
        """Draw the user interface."""
        if self.state.game_over:
            win = self.state.get_winner()
            if win:
                txt = f"Player {win} ({'Red' if win==1 else 'Blue'}) wins!"
                color = core.PLAYER_COLOR[win]
            else:
                txt, color = "Game over!", WHITE
        else:
            if self.mode == MODE_PVP:
                turn = 'Red' if self.state.current_player == 1 else 'Blue'
                txt = f"PVP - {turn}'s turn"
            else:
                txt = "PVAI - Your turn (Red)"
            color = WHITE
        
        # Game status
        surf = self.font.render(txt, True, color)
        self.screen.blit(surf, (10, HEIGHT - 60))
        
        # Controls
        controls = "M-Mode  C-Config  ESC-Menu"
        control_surf = self.small_font.render(controls, True, GRAY)
        self.screen.blit(control_surf, (10, HEIGHT - 30))

    def run(self):
        """Main game loop."""
        self._main_menu()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self._toggle_mode()
                    elif event.key == pygame.K_c:
                        self._ai_config_menu()
                    elif event.key == pygame.K_ESCAPE:
                        self._main_menu()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            # Update animations
            now = pygame.time.get_ticks()
            self.animations[:] = [anim for anim in self.animations
                                 if now - anim.start_time < EXPLOSION_MS]

            # Draw everything
            self.screen.fill(BLACK)
            self._draw_grid()
            self._draw_orbs()
            self._draw_ui()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    ChainReactionGUI().run()