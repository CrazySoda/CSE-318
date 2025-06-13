from __future__ import annotations
import sys, time, os, pygame, threading, json, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import core
import ai

# ──────────────────────────────────────────────────────────────────────
# File-protocol helpers (used only in PVAI mode)
# ──────────────────────────────────────────────────────────────────────
FILE = "gamestate.txt"

def _write_state(header: str, state: core.GameState):
    """Write game state to file."""
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(state.to_lines()))

def _write_state_with_move(header: str, state: core.GameState, move: tuple[int, int] = None):
    """Write game state to file with optional move coordinates."""
    with open(FILE, "w", encoding="utf-8") as f:
        lines = [header]
        
        # Add move coordinates if provided
        if move is not None:
            lines.append(f"MOVE:{move[0]},{move[1]}")
        
        # Add game state
        lines.extend(state.to_lines())
        
        f.write("\n".join(lines))

def _read_until(header: str) -> core.GameState:
    """Read game state from file, waiting until header matches."""
    while True:
        try:
            with open(FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            time.sleep(0.05)
            continue
            
        if lines and lines[0].strip() == header:
            # Filter out the header and any MOVE: lines for state parsing
            state_lines = []
            for line in lines[1:]:  # Skip header
                if not line.startswith("MOVE:"):
                    state_lines.append(line)
            return core.GameState.from_file(state_lines)
        time.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────
# Enhanced GUI constants with modern color scheme
# ──────────────────────────────────────────────────────────────────────
CELL_SIZE = 85  # Reduced from 100 to make more compact
BOARD_WIDTH, BOARD_HEIGHT = 6 * CELL_SIZE, 9 * CELL_SIZE
UI_HEIGHT = 60  # Space for UI elements at bottom
WIDTH, HEIGHT = BOARD_WIDTH, BOARD_HEIGHT + UI_HEIGHT
FPS = 60

# Modern color palette
COLORS = {
    'bg_primary': (25, 28, 35),        # Dark blue-gray background
    'bg_secondary': (35, 40, 50),      # Lighter secondary background
    'bg_menu': (45, 52, 65),           # Menu background
    'accent': (100, 200, 255),         # Bright blue accent
    'accent_dark': (80, 160, 220),     # Darker blue
    'success': (80, 200, 120),         # Green
    'warning': (255, 180, 80),         # Orange
    'danger': (255, 100, 100),         # Red
    'text_primary': (240, 242, 245),   # Light text
    'text_secondary': (180, 185, 195), # Secondary text
    'text_muted': (120, 125, 135),     # Muted text
    'border': (70, 80, 95),            # Border color
    'highlight': (55, 65, 80),         # Highlight background
    'player1': (255, 100, 120),        # Player 1 red
    'player2': (100, 180, 255),        # Player 2 blue
    'explosion': (255, 255, 100),      # Explosion color
    'explosion_inner': (255, 200, 50), # Inner explosion
}

# Legacy color variables for compatibility
GRAY = COLORS['border']
BLACK = COLORS['bg_primary']
WHITE = COLORS['text_primary']
GREEN = COLORS['success']
LIGHT_GRAY = COLORS['text_secondary']
DARK_GRAY = COLORS['bg_secondary']
RED = COLORS['player1']
BLUE = COLORS['player2']
YELLOW = COLORS['warning']

# Animation timing constants
EXPLOSION_DURATION_MS = 400    # How long explosion effect lasts
ORB_PLACEMENT_DELAY_MS = 150   # Delay between orb placements
CHAIN_STEP_DELAY_MS = 300      # Delay between chain reaction steps
PARTICLE_COUNT = 8             # Number of explosion particles

MODE_PVP = "PVP"
MODE_PVAI = "PVAI"
MODE_AVAI = "AVAI"

class AnimationType(Enum):
    EXPLOSION = "explosion"
    ORB_PLACEMENT = "orb_placement"

@dataclass
class ExplosionParticle:
    x: float
    y: float
    vel_x: float
    vel_y: float
    life: float
    max_life: float
    color: Tuple[int, int, int]

@dataclass
class ExplosionAnim:
    row: int
    col: int
    start_time: int
    duration: int = EXPLOSION_DURATION_MS
    particles: List[ExplosionParticle] = field(default_factory=list)
    
    def __post_init__(self):
        # Create explosion particles
        cx = self.col * CELL_SIZE + CELL_SIZE // 2
        cy = self.row * CELL_SIZE + CELL_SIZE // 2
        
        for i in range(PARTICLE_COUNT):
            angle = (i / PARTICLE_COUNT) * 2 * math.pi
            speed = 2 + (i % 3)  # Varied speeds
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            
            life = self.duration / 1000.0  # Convert to seconds
            color = COLORS['explosion'] if i % 2 == 0 else COLORS['explosion_inner']
            
            particle = ExplosionParticle(
                x=float(cx), y=float(cy), 
                vel_x=vel_x, vel_y=vel_y,
                life=life, max_life=life, color=color
            )
            self.particles.append(particle)

@dataclass
class OrbPlacementAnim:
    row: int
    col: int
    player: int
    start_time: int
    duration: int = ORB_PLACEMENT_DELAY_MS
    scale: float = 0.0

@dataclass
class AnimationStep:
    step_type: AnimationType
    data: Any
    start_time: int
    completed: bool = False

@dataclass
class MatchStats:
    """Statistics for AI vs AI matches."""
    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    total_games: int = 0
    current_game_moves: int = 0
    avg_moves_per_game: float = 0.0
    avg_game_duration: float = 0.0
    last_winner: Optional[int] = None

class ChainReactionGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chain Reaction - Sequential Animation")
        self.clock = pygame.time.Clock()
        
        # Compact font system
        self.fonts = {
            'title': pygame.font.SysFont("Segoe UI", 22, bold=True),
            'header': pygame.font.SysFont("Segoe UI", 18, bold=True),
            'normal': pygame.font.SysFont("Segoe UI", 14),
            'small': pygame.font.SysFont("Segoe UI", 12),
            'tiny': pygame.font.SysFont("Segoe UI", 10),
        }

        self.state = core.GameState(rows=9, cols=6)
        self.visual_state = core.GameState(rows=9, cols=6)  # What's currently displayed
        self.mode = None
        
        # Animation systems - NEW
        self.explosion_animations: List[ExplosionAnim] = []
        self.orb_placement_animations: List[OrbPlacementAnim] = []
        self.animation_queue: List[AnimationStep] = []
        self.animating = False
        
        # Legacy animation list for compatibility
        self.animations: List[ExplosionAnim] = []
        
        # AI Configuration
        self.ai_config = ai.AIConfig()
        
        # AI vs AI specific
        self.ai_player1_config = ai.create_balanced_config()
        self.ai_player2_config = ai.create_aggressive_config()
        self.ai_agent1: Optional[ai.MinimaxAgent] = None
        self.ai_agent2: Optional[ai.MinimaxAgent] = None
        self.ai_vs_ai_running = False
        self.ai_move_delay = 1.0
        self.last_ai_move_time = 0
        self.auto_restart = True
        self.match_stats = MatchStats()
        self.current_game_start_time = 0
        
        # PVAI specific - external engine communication
        self.waiting_for_ai = False
        self.ai_check_interval = 0.1  # Check for AI response every 100ms
        self.last_ai_check_time = 0
        
        # Available AI presets
        self.ai_presets = {
            "Balanced": ai.create_balanced_config,
            "Aggressive": ai.create_aggressive_config,
            "Defensive": ai.create_defensive_config,
            "Material Only": ai.create_material_only_config
        }

    def draw_rounded_rect(self, surface, color, rect, radius=8):
        """Draw a rounded rectangle."""
        x, y, w, h = rect
        pygame.draw.rect(surface, color, (x + radius, y, w - 2*radius, h))
        pygame.draw.rect(surface, color, (x, y + radius, w, h - 2*radius))
        pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
        pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)
        pygame.draw.circle(surface, color, (x + radius, y + h - radius), radius)
        pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)

    def draw_button(self, surface, text, rect, font_key='normal', selected=False, enabled=True):
        """Draw a modern button with hover effects."""
        x, y, w, h = rect
        
        # Button colors
        if not enabled:
            bg_color = COLORS['bg_secondary']
            text_color = COLORS['text_muted']
            border_color = COLORS['border']
        elif selected:
            bg_color = COLORS['accent']
            text_color = COLORS['bg_primary']
            border_color = COLORS['accent_dark']
        else:
            bg_color = COLORS['bg_menu']
            text_color = COLORS['text_primary']
            border_color = COLORS['border']
        
        # Draw button background
        self.draw_rounded_rect(surface, bg_color, rect, 6)
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=6)
        
        # Draw text
        text_surf = self.fonts[font_key].render(text, True, text_color)
        text_x = x + (w - text_surf.get_width()) // 2
        text_y = y + (h - text_surf.get_height()) // 2
        surface.blit(text_surf, (text_x, text_y))
        
        return rect

    def draw_panel(self, surface, rect, title=None):
        """Draw a panel with optional title."""
        x, y, w, h = rect
        
        # Panel background
        self.draw_rounded_rect(surface, COLORS['bg_menu'], rect, 8)
        pygame.draw.rect(surface, COLORS['border'], rect, 1, border_radius=8)
        
        title_height = 0
        if title:
            # Title bar
            title_rect = (x, y, w, 25)
            self.draw_rounded_rect(surface, COLORS['accent'], (x, y, w, 25), 8)
            pygame.draw.rect(surface, COLORS['bg_menu'], (x, y + 15, w, 10))
            
            title_surf = self.fonts['header'].render(title, True, COLORS['bg_primary'])
            title_x = x + 10
            title_y = y + 3
            surface.blit(title_surf, (title_x, title_y))
            title_height = 25
        
        return (x + 10, y + title_height + 10, w - 20, h - title_height - 20)

    def _reset_all_mode_state(self):
        """Reset all mode-specific state variables."""
        # Reset AI vs AI state
        self.ai_vs_ai_running = False
        self.ai_agent1 = None
        self.ai_agent2 = None
        self.last_ai_move_time = 0
        self.match_stats = MatchStats()
        self.current_game_start_time = 0
        
        # Reset PVAI state
        self.waiting_for_ai = False
        self.last_ai_check_time = 0
        
        # Reset game state
        self.state.reset()
        self.visual_state.reset()
        
        # Reset animation state
        self.animations.clear()
        self.explosion_animations.clear()
        self.orb_placement_animations.clear()
        self.animation_queue.clear()
        self.animating = False
        
        # Clean up any existing game state file
        try:
            if os.path.exists(FILE):
                os.remove(FILE)
        except:
            pass

    def _main_menu(self):
        """Compact main menu with better visual design."""
        # Reset all state when entering main menu
        self._reset_all_mode_state()
        
        choosing = True
        selected = 0
        
        options = [
            ("1", "Two Players (PVP)", "Play against a friend"),
            ("2", "Human vs AI (PVAI)", "Challenge the computer"),
            ("3", "AI vs AI Battle (AVAI)", "Watch AIs compete"),
            ("4", "Configure AI", "Customize AI behavior"),
        ]
        
        while choosing:
            self.screen.fill(COLORS['bg_primary'])
            
            # Main panel
            panel_width = min(300, WIDTH - 40)
            panel_height = min(400, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "Chain Reaction AI")
            
            # Options
            button_height = 35
            button_spacing = 5
            start_y = content_rect[1] + 20
            
            for i, (key, title, desc) in enumerate(options):
                button_rect = (content_rect[0], start_y + i * (button_height + button_spacing), 
                             content_rect[2], button_height)
                
                is_selected = (i == selected)
                self.draw_button(self.screen, f"{key}. {title}", button_rect, 'normal', is_selected)
                
                # Description
                if desc:
                    desc_y = button_rect[1] + button_height + 2
                    desc_surf = self.fonts['small'].render(desc, True, COLORS['text_muted'])
                    self.screen.blit(desc_surf, (button_rect[0] + 10, desc_y))
            
            # Controls hint
            controls_y = content_rect[1] + content_rect[3] - 40
            controls_text = "Use 1-4 keys or ESC to quit"
            controls_surf = self.fonts['small'].render(controls_text, True, COLORS['text_secondary'])
            controls_x = content_rect[0] + (content_rect[2] - controls_surf.get_width()) // 2
            self.screen.blit(controls_surf, (controls_x, controls_y))
            
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
                        self.mode = MODE_AVAI
                        choosing = False
                    elif event.key == pygame.K_4:
                        self._ai_config_menu()
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(options)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        if selected == 0:
                            self.mode = MODE_PVP
                            choosing = False
                        elif selected == 1:
                            self.mode = MODE_PVAI
                            choosing = False
                        elif selected == 2:
                            self.mode = MODE_AVAI
                            choosing = False
                        elif selected == 3:
                            self._ai_config_menu()
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

        # Initialize the selected mode
        self._initialize_mode()

    def _initialize_mode(self):
        """Initialize the selected game mode."""
        print(f"Initializing mode: {self.mode}")
        
        if self.mode == MODE_PVP:
            # PVP mode - just reset the board
            self.state.reset()
            self.visual_state.reset()
            
        elif self.mode == MODE_PVAI:
            # PVAI mode - create initial game state file for external AI engine
            self.state.reset()
            self.visual_state.reset()
            _write_state("AI Move:", self.state)
            print("PVAI mode: Initial game state written for external AI engine")
            print("Make sure to run 'python engine.py' in another terminal!")
            
        elif self.mode == MODE_AVAI:
            # AVAI mode - configure and start AI vs AI
            self._ai_vs_ai_config_menu()
            
        print(f"Mode {self.mode} initialized successfully")

    def _reset_board(self):
        """Reset the game board - DEPRECATED, use _initialize_mode instead."""
        self._initialize_mode()

    def _preset_menu_for_player(self, player: int):
        """Compact preset selection menu."""
        selecting = True
        selected = 0
        preset_names = list(self.ai_presets.keys())
        
        descriptions = {
            "Balanced": "Well-rounded strategy",
            "Aggressive": "High-risk, high-reward",
            "Defensive": "Territory focused",
            "Material Only": "Orb counting only"
        }
        
        while selecting:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(280, WIDTH - 40)
            panel_height = min(350, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, f"Presets - Player {player}")
            
            button_height = 30
            button_spacing = 5
            start_y = content_rect[1] + 10
            
            for i, preset_name in enumerate(preset_names):
                button_rect = (content_rect[0], start_y + i * (button_height + button_spacing + 15), 
                             content_rect[2], button_height)
                
                is_selected = (i == selected)
                self.draw_button(self.screen, preset_name, button_rect, 'normal', is_selected)
                
                # Description
                desc_y = button_rect[1] + button_height + 2
                desc_surf = self.fonts['small'].render(descriptions.get(preset_name, ""), True, COLORS['text_muted'])
                self.screen.blit(desc_surf, (button_rect[0] + 5, desc_y))
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 25
            controls = "↑↓: Select | ENTER: Load | ESC: Cancel"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_secondary'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
            
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
                        config = self.ai_presets[preset_name]()
                        if player == 1:
                            self.ai_player1_config = config
                        else:
                            self.ai_player2_config = config
                        selecting = False
                    elif event.key == pygame.K_ESCAPE:
                        selecting = False

    def _change_depth_for_player(self, player: int):
        """Compact depth adjustment interface."""
        changing = True
        config = self.ai_player1_config if player == 1 else self.ai_player2_config
        
        while changing:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(250, WIDTH - 40)
            panel_height = min(200, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, f"Depth - Player {player}")
            
            # Current depth display
            depth_text = f"Current: {config.depth}"
            depth_surf = self.fonts['title'].render(depth_text, True, COLORS['text_primary'])
            depth_x = content_rect[0] + (content_rect[2] - depth_surf.get_width()) // 2
            self.screen.blit(depth_surf, (depth_x, content_rect[1] + 20))
            
            # Range info
            range_text = f"Range: {ai.MIN_DEPTH} - {ai.MAX_DEPTH}"
            range_surf = self.fonts['small'].render(range_text, True, COLORS['text_secondary'])
            range_x = content_rect[0] + (content_rect[2] - range_surf.get_width()) // 2
            self.screen.blit(range_surf, (range_x, content_rect[1] + 50))
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 40
            controls = "← → Adjust | ENTER: Confirm"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_muted'])
            controls_x = content_rect[0] + (content_rect[2] - controls_surf.get_width()) // 2
            self.screen.blit(controls_surf, (controls_x, controls_y))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        config.set_depth(config.depth - 1)
                    elif event.key == pygame.K_RIGHT:
                        config.set_depth(config.depth + 1)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        changing = False

    def _toggle_heuristics_for_player(self, player: int):
        """Compact heuristics toggle interface."""
        config = self.ai_player1_config if player == 1 else self.ai_player2_config
        
        toggling = True
        selected = 0
        heuristic_names = list(config.enabled_heuristics.keys())
        
        while toggling:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(350, WIDTH - 20)
            panel_height = min(400, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, f"Heuristics - Player {player}")
            
            # Heuristics list
            item_height = 20
            start_y = content_rect[1] + 10
            
            for i, heuristic in enumerate(heuristic_names):
                enabled = config.enabled_heuristics[heuristic]
                y_pos = start_y + i * item_height
                
                # Selection highlight
                if i == selected:
                    highlight_rect = (content_rect[0] - 5, y_pos - 2, content_rect[2] + 10, item_height)
                    self.draw_rounded_rect(self.screen, COLORS['highlight'], highlight_rect, 4)
                
                # Status indicator
                status_color = COLORS['success'] if enabled else COLORS['text_muted']
                status_text = "●" if enabled else "○"
                status_surf = self.fonts['normal'].render(status_text, True, status_color)
                self.screen.blit(status_surf, (content_rect[0], y_pos))
                
                # Heuristic name
                name_text = heuristic.replace('_', ' ').title()
                name_color = COLORS['text_primary'] if enabled else COLORS['text_muted']
                name_surf = self.fonts['normal'].render(name_text, True, name_color)
                self.screen.blit(name_surf, (content_rect[0] + 25, y_pos))
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 25
            controls = "↑↓: Select | SPACE: Toggle | ENTER: Done"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_secondary'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
            
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
                        current = config.enabled_heuristics[heuristic]
                        config.enabled_heuristics[heuristic] = not current
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        toggling = False

    def _speed_settings_menu(self):
        """Compact speed settings interface."""
        configuring = True
        
        while configuring:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(250, WIDTH - 40)
            panel_height = min(250, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "Speed Settings")
            
            # Move delay
            delay_text = f"Move Delay: {self.ai_move_delay:.1f}s"
            delay_surf = self.fonts['normal'].render(delay_text, True, COLORS['text_primary'])
            self.screen.blit(delay_surf, (content_rect[0], content_rect[1] + 20))
            
            # Auto restart
            restart_text = f"Auto Restart: {'ON' if self.auto_restart else 'OFF'}"
            restart_surf = self.fonts['normal'].render(restart_text, True, COLORS['text_primary'])
            self.screen.blit(restart_surf, (content_rect[0], content_rect[1] + 45))
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 40
            controls = ["← → Adjust delay", "A: Toggle restart", "ENTER: Done"]
            for i, control in enumerate(controls):
                control_surf = self.fonts['small'].render(control, True, COLORS['text_muted'])
                self.screen.blit(control_surf, (content_rect[0], controls_y + i * 15))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.ai_move_delay = max(0.1, self.ai_move_delay - 0.1)
                    elif event.key == pygame.K_RIGHT:
                        self.ai_move_delay = min(5.0, self.ai_move_delay + 0.1)
                    elif event.key == pygame.K_a:
                        self.auto_restart = not self.auto_restart
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        configuring = False

    def _test_ai_vs_ai(self):
        """Compact test results display."""
        testing = True
        
        # Create test agents
        agent1 = ai.MinimaxAgent(player=1, config=self.ai_player1_config)
        agent2 = ai.MinimaxAgent(player=2, config=self.ai_player2_config)
        
        # Test on simple scenario
        test_state = core.GameState(rows=9, cols=6)
        test_state.board[2][2].owner = 1
        test_state.board[2][2].count = 2
        test_state.board[6][4].owner = 2
        test_state.board[6][4].count = 1
        test_state.current_player = 1
        
        # Get moves from both agents
        start_time = time.time()
        move1 = agent1.choose_move(test_state.clone())
        time1 = time.time() - start_time
        
        start_time = time.time()
        move2 = agent2.choose_move(test_state.clone())
        time2 = time.time() - start_time
        
        stats1 = agent1.get_search_statistics()
        stats2 = agent2.get_search_statistics()
        
        while testing:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(350, WIDTH - 20)
            panel_height = min(300, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "Test Results")
            
            y_pos = content_rect[1] + 10
            line_height = 18
            
            # Player 1 results
            p1_results = [
                f"Player 1 (Red): {move1}",
                f"  Time: {time1:.3f}s | Nodes: {stats1['nodes_explored']}",
            ]
            
            for result in p1_results:
                color = COLORS['player1'] if result.startswith("Player") else COLORS['text_secondary']
                result_surf = self.fonts['small'].render(result, True, color)
                self.screen.blit(result_surf, (content_rect[0], y_pos))
                y_pos += line_height
            
            y_pos += 10
            
            # Player 2 results
            p2_results = [
                f"Player 2 (Blue): {move2}",
                f"  Time: {time2:.3f}s | Nodes: {stats2['nodes_explored']}",
            ]
            
            for result in p2_results:
                color = COLORS['player2'] if result.startswith("Player") else COLORS['text_secondary']
                result_surf = self.fonts['small'].render(result, True, color)
                self.screen.blit(result_surf, (content_rect[0], y_pos))
                y_pos += line_height
            
            # Continue prompt
            continue_y = content_rect[1] + content_rect[3] - 25
            continue_surf = self.fonts['small'].render("Press any key to continue", True, COLORS['text_muted'])
            continue_x = content_rect[0] + (content_rect[2] - continue_surf.get_width()) // 2
            self.screen.blit(continue_surf, (continue_x, continue_y))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    testing = False

    def _ai_config_menu(self):
        configuring = True
        
        while configuring:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(300, WIDTH - 40)
            panel_height = min(200, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Configuration")
            
            # Simple configuration display
            y_pos = content_rect[1] + 20
            
            config_text = f"Depth: {self.ai_config.depth}"
            config_surf = self.fonts['normal'].render(config_text, True, COLORS['text_primary'])
            self.screen.blit(config_surf, (content_rect[0], y_pos))
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 25
            controls = "B: Back"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_secondary'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b or event.key == pygame.K_ESCAPE:
                        configuring = False

    def _ai_vs_ai_config_menu(self):
        """Compact AI vs AI configuration menu."""
        configuring = True
        selected_player = 1
        selected_option = 0
        
        options = [
            ("P", "Load Preset"),
            ("D", "Change Depth"),
            ("H", "Toggle Heuristics"),
            ("S", "Speed Settings"),
            ("T", "Test Config"),
            ("R", "Start Match"),
        ]
        
        while configuring:
            self.screen.fill(COLORS['bg_primary'])
            
            # Main panel
            panel_width = min(400, WIDTH - 20)
            panel_height = min(500, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI vs AI Configuration")
            
            # Player selector
            player_y = content_rect[1]
            for i, (player_num, color, config) in enumerate([(1, COLORS['player1'], self.ai_player1_config), 
                                                            (2, COLORS['player2'], self.ai_player2_config)]):
                button_width = content_rect[2] // 2 - 5
                button_x = content_rect[0] + i * (button_width + 10)
                button_rect = (button_x, player_y, button_width, 25)
                
                is_selected = (selected_player == player_num)
                player_text = f"Player {player_num} ({'Red' if player_num == 1 else 'Blue'})"
                
                if is_selected:
                    pygame.draw.rect(self.screen, color, button_rect, border_radius=4)
                    text_color = COLORS['bg_primary']
                else:
                    pygame.draw.rect(self.screen, COLORS['bg_secondary'], button_rect, border_radius=4)
                    pygame.draw.rect(self.screen, color, button_rect, 2, border_radius=4)
                    text_color = color
                
                text_surf = self.fonts['normal'].render(player_text, True, text_color)
                text_x = button_x + (button_width - text_surf.get_width()) // 2
                text_y = player_y + 4
                self.screen.blit(text_surf, (text_x, text_y))
            
            # Current configuration display
            config_y = player_y + 35
            current_config = self.ai_player1_config if selected_player == 1 else self.ai_player2_config
            
            config_info = [
                f"Depth: {current_config.depth}",
                f"Enabled: {len([h for h, e in current_config.enabled_heuristics.items() if e])}/6 heuristics",
                f"Speed: {self.ai_move_delay:.1f}s delay",
                f"Auto-restart: {'ON' if self.auto_restart else 'OFF'}"
            ]
            
            for i, info in enumerate(config_info):
                info_surf = self.fonts['small'].render(info, True, COLORS['text_secondary'])
                self.screen.blit(info_surf, (content_rect[0], config_y + i * 18))
            
            # Options menu
            options_y = config_y + 80
            button_height = 25
            button_spacing = 3
            
            for i, (key, title) in enumerate(options):
                button_rect = (content_rect[0], options_y + i * (button_height + button_spacing), 
                             content_rect[2], button_height)
                
                is_selected = (i == selected_option)
                self.draw_button(self.screen, f"{key}. {title}", button_rect, 'small', is_selected)
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 30
            controls = "TAB: Switch Player | ENTER: Select | B: Back"
            controls_surf = self.fonts['tiny'].render(controls, True, COLORS['text_muted'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        selected_player = 2 if selected_player == 1 else 1
                    elif event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        key, _ = options[selected_option]
                        if key == "P":
                            self._preset_menu_for_player(selected_player)
                        elif key == "D":
                            self._change_depth_for_player(selected_player)
                        elif key == "H":
                            self._toggle_heuristics_for_player(selected_player)
                        elif key == "S":
                            self._speed_settings_menu()
                        elif key == "T":
                            self._test_ai_vs_ai()
                        elif key == "R":
                            configuring = False
                            self._start_ai_vs_ai()
                    elif event.key == pygame.K_p:
                        self._preset_menu_for_player(selected_player)
                    elif event.key == pygame.K_d:
                        self._change_depth_for_player(selected_player)
                    elif event.key == pygame.K_h:
                        self._toggle_heuristics_for_player(selected_player)
                    elif event.key == pygame.K_s:
                        self._speed_settings_menu()
                    elif event.key == pygame.K_t:
                        self._test_ai_vs_ai()
                    elif event.key == pygame.K_r:
                        configuring = False
                        self._start_ai_vs_ai()
                    elif event.key == pygame.K_b or event.key == pygame.K_ESCAPE:
                        # Return to main menu instead of just exiting config
                        self.mode = None
                        configuring = False

    def _start_ai_vs_ai(self):
        """Start AI vs AI match."""
        self.ai_agent1 = ai.MinimaxAgent(player=1, config=self.ai_player1_config)
        self.ai_agent2 = ai.MinimaxAgent(player=2, config=self.ai_player2_config)
        self.ai_vs_ai_running = True
        self.current_game_start_time = time.time()
        self.match_stats.current_game_moves = 0
        self.state.reset()
        self.visual_state.reset()
        print("🤖 AI vs AI match started!")

    def _write_human_move(self):
        """Write human move to file for AI engine."""
        _write_state("Human Move:", self.state)

    def _wait_ai(self):
        """Wait for AI move from engine."""
        self.state = _read_until("AI Move:")

    def create_animated_move(self, r: int, c: int) -> bool:
        """Create animated move sequence instead of instant move."""
        if self.animating:
            return False
        
        # Clone state to simulate the move
        temp_state = self.state.clone()
        try:
            explosions = temp_state.apply_move(r, c)
        except ValueError:
            return False
        
        # Create animation sequence
        now = pygame.time.get_ticks()
        
        # Step 1: Initial orb placement
        placement_data = (r, c, self.state.current_player)
        self.animation_queue.append(AnimationStep(
            AnimationType.ORB_PLACEMENT, 
            placement_data, 
            now
        ))
        
        # Step 2: Process explosions in sequence
        if explosions:
            explosion_delay = ORB_PLACEMENT_DELAY_MS
            
            for i, (er, ec) in enumerate(explosions):
                explosion_time = now + explosion_delay + (i * CHAIN_STEP_DELAY_MS)
                self.animation_queue.append(AnimationStep(
                    AnimationType.EXPLOSION,
                    (er, ec),
                    explosion_time
                ))
        
        # Apply the actual move to game state (but not visual state yet)
        self.state.apply_move(r, c)
        self.animating = True
        
        # Also add to legacy animations for compatibility
        for er, ec in explosions:
            self.animations.append(ExplosionAnim(er, ec, now))
        
        return True

    def update_animations(self):
        """Update all animation systems."""
        if not self.animating:
            return
        
        now = pygame.time.get_ticks()
        
        # Update explosion particles
        for explosion in self.explosion_animations[:]:
            dt = 0.016  # Approximate delta time for 60 FPS
            
            for particle in explosion.particles:
                particle.life -= dt
                particle.x += particle.vel_x
                particle.y += particle.vel_y
                particle.vel_y += 0.1  # Gravity effect
            
            # Remove expired explosions
            if now - explosion.start_time > explosion.duration:
                self.explosion_animations.remove(explosion)
        
        # Update orb placement animations
        for placement in self.orb_placement_animations[:]:
            elapsed = now - placement.start_time
            progress = min(1.0, elapsed / placement.duration)
            
            # Bouncy scale animation
            if progress < 0.7:
                placement.scale = progress / 0.7 * 1.2  # Scale up to 1.2
            else:
                bounce_progress = (progress - 0.7) / 0.3
                placement.scale = 1.2 - (bounce_progress * 0.2)  # Scale down to 1.0
            
            if progress >= 1.0:
                # Apply the orb to visual state
                self.visual_state.board[placement.row][placement.col].owner = placement.player
                self.visual_state.board[placement.row][placement.col].count += 1
                self.orb_placement_animations.remove(placement)
        
        # Process animation queue
        for step in self.animation_queue[:]:
            if step.completed:
                continue
                
            if now >= step.start_time:
                if step.step_type == AnimationType.ORB_PLACEMENT:
                    r, c, player = step.data
                    self.orb_placement_animations.append(OrbPlacementAnim(
                        r, c, player, now, ORB_PLACEMENT_DELAY_MS
                    ))
                    step.completed = True
                    
                elif step.step_type == AnimationType.EXPLOSION:
                    r, c = step.data
                    self.explosion_animations.append(ExplosionAnim(r, c, now))
                    
                    # Clear the exploded cell in visual state
                    self.visual_state.board[r][c].owner = None
                    self.visual_state.board[r][c].count = 0
                    
                    # Add orbs to adjacent cells
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 9 and 0 <= nc < 6:
                            # Schedule orb placement with slight delay
                            orb_delay = 100  # Small delay for each orb
                            orb_time = now + orb_delay
                            self.orb_placement_animations.append(OrbPlacementAnim(
                                nr, nc, self.state.current_player, orb_time, ORB_PLACEMENT_DELAY_MS
                            ))
                    
                    step.completed = True
        
        # Remove completed steps
        self.animation_queue = [step for step in self.animation_queue if not step.completed]
        
        # Check if all animations are complete
        if (not self.animation_queue and 
            not self.explosion_animations and 
            not self.orb_placement_animations):
            self.animating = False
            
            # Sync visual state with actual game state
            for r in range(9):
                for c in range(6):
                    self.visual_state.board[r][c].owner = self.state.board[r][c].owner
                    self.visual_state.board[r][c].count = self.state.board[r][c].count
            
            self.visual_state.current_player = self.state.current_player
            self.visual_state.game_over = self.state.game_over

    def handle_click(self, mx: int, my: int):
        """Handle mouse clicks on the game board."""
        if self.mode == MODE_AVAI or self.animating or self.waiting_for_ai:  # Block clicks during animation or AI thinking
            return
            
        if self.state.game_over:
            return
        
        # Only process clicks within the game board area
        if my >= BOARD_HEIGHT:
            return
        
        r, c = my // CELL_SIZE, mx // CELL_SIZE

        # Use animated move system
        if self.create_animated_move(r, c):
            if self.mode == MODE_PVAI and self.state.current_player == 2 and not self.state.game_over:
                # Wait for animation to complete, then handle AI move
                threading.Thread(target=self._handle_pvai_move_after_animation, daemon=True).start()

    def _handle_pvai_move_after_animation(self):
        """Handle AI move after human move animation completes (external engine)."""
        # Wait for animation to complete
        while self.animating:
            time.sleep(0.1)
        
        if not self.state.game_over and self.state.current_player == 2:
            # Write human move to file for external AI engine
            self._write_human_move()
            # Set flag to start checking for AI response
            self.waiting_for_ai = True
            self.last_ai_check_time = time.time()
            print("Waiting for AI engine response...")

    def _check_for_ai_response(self):
        """Non-blocking check for AI response from external engine."""
        if not self.waiting_for_ai:
            return
        
        current_time = time.time()
        if current_time - self.last_ai_check_time < self.ai_check_interval:
            return
        
        self.last_ai_check_time = current_time
        
        try:
            # Check if AI has responded
            if os.path.exists(FILE):
                with open(FILE, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                
                if lines and lines[0].strip() == "AI Move:":
                    # Look for explicit move coordinates first
                    ai_move = None
                    state_lines = []
                    
                    # Parse the file format
                    for i, line in enumerate(lines[1:], 1):  # Skip header
                        if line.startswith("MOVE:"):
                            try:
                                coords_str = line.split(":", 1)[1].strip()
                                coords = coords_str.split(",")
                                ai_move = (int(coords[0].strip()), int(coords[1].strip()))
                                print(f"Found explicit AI move: {ai_move}")
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing move coordinates: {e}")
                                ai_move = None
                        else:
                            # This should be game state data
                            state_lines.append(line)
                    
                    # If we have explicit move coordinates, use them
                    if ai_move is not None:
                        # Validate the move is legal
                        if self._is_legal_move(ai_move[0], ai_move[1]):
                            if self.create_animated_move(ai_move[0], ai_move[1]):
                                print(f"Applied AI move: {ai_move}")
                                self.waiting_for_ai = False
                                return
                            else:
                                print(f"Failed to apply AI move: {ai_move}")
                        else:
                            print(f"Illegal AI move received: {ai_move}")
                    
                    # Fallback to state comparison if no explicit move or move failed
                    if len(state_lines) >= 10:  # Should have metadata + 9 board rows
                        try:
                            new_state = core.GameState.from_file(state_lines)
                            ai_move = self._find_ai_move_improved(self.state, new_state)
                            
                            if ai_move:
                                if self._is_legal_move(ai_move[0], ai_move[1]):
                                    if self.create_animated_move(ai_move[0], ai_move[1]):
                                        print(f"Applied AI move from state comparison: {ai_move}")
                                        self.waiting_for_ai = False
                                        return
                                    else:
                                        print(f"Failed to apply AI move from state comparison: {ai_move}")
                                else:
                                    print(f"Illegal AI move from state comparison: {ai_move}")
                            else:
                                print("Could not determine AI move from state comparison")
                        except Exception as e:
                            print(f"Error parsing game state: {e}")
                    
                    # If we get here, something went wrong
                    print("Failed to process AI response - continuing to wait")
                    
        except Exception as e:
            print(f"Error checking AI response: {e}")
            # Continue waiting, maybe the file is being written

    def _is_legal_move(self, r: int, c: int) -> bool:
        """Check if a move is legal in the current state."""
        if self.state.game_over:
            return False
        if not (0 <= r < self.state.rows and 0 <= c < self.state.cols):
            return False
        
        cell = self.state.board[r][c]
        current_player = self.state.current_player
        
        # Can play on empty cells or own cells
        return cell.owner in (None, current_player)

    def _find_ai_move_improved(self, old_state: core.GameState, new_state: core.GameState) -> Optional[Tuple[int, int]]:
        """Improved method to find AI move by comparing states, handles explosions better."""
        # First, try the simple approach for non-explosive moves
        simple_move = self._find_ai_move_simple(old_state, new_state)
        if simple_move:
            return simple_move
        
        # For explosive moves, look for the most likely starting point
        # Strategy: find cells where the AI could have legally played
        legal_moves = old_state.generate_moves(2)  # AI is player 2
        candidates = []
        
        for r, c in legal_moves:
            # Simulate this move and see if it could lead to the new state
            test_state = old_state.clone()
            try:
                test_state.apply_move(r, c)
                # Check if this results in a state "consistent" with the new state
                score = self._calculate_state_similarity(test_state, new_state)
                candidates.append((score, (r, c)))
            except:
                continue
        
        # Return the move that results in the most similar state
        if candidates:
            candidates.sort(reverse=True)  # Sort by similarity score
            best_score, best_move = candidates[0]
            if best_score > 0.5:  # Only return if reasonably confident
                print(f"Best move candidate: {best_move} with similarity score: {best_score:.2f}")
                return best_move
        
        return None

    def _find_ai_move_simple(self, old_state: core.GameState, new_state: core.GameState) -> Optional[Tuple[int, int]]:
        """Simple move detection for non-explosive cases."""
        for r in range(old_state.rows):
            for c in range(old_state.cols):
                old_cell = old_state.board[r][c]
                new_cell = new_state.board[r][c]
                
                # Check if this cell had an orb added by player 2 (AI)
                if (old_cell.owner in (None, 2) and new_cell.owner == 2 and 
                    new_cell.count > old_cell.count):
                    return (r, c)
                
                # Check if this was an empty cell that got an orb from player 2
                if old_cell.owner is None and new_cell.owner == 2 and new_cell.count > 0:
                    return (r, c)
        
        return None

    def _calculate_state_similarity(self, state1: core.GameState, state2: core.GameState) -> float:
        """Calculate similarity score between two game states (0.0 to 1.0)."""
        total_cells = state1.rows * state1.cols
        matching_cells = 0
        
        for r in range(state1.rows):
            for c in range(state1.cols):
                cell1 = state1.board[r][c]
                cell2 = state2.board[r][c]
                
                # Check if cells match exactly
                if cell1.owner == cell2.owner and cell1.count == cell2.count:
                    matching_cells += 1
                # Partial credit for same owner but different count
                elif cell1.owner == cell2.owner and cell1.owner is not None:
                    matching_cells += 0.5
        
        # Also check if game state properties match
        state_match_bonus = 0
        if state1.current_player == state2.current_player:
            state_match_bonus += 0.1
        if state1.game_over == state2.game_over:
            state_match_bonus += 0.1
        
        return (matching_cells / total_cells) + state_match_bonus

    def _find_ai_move(self, old_state: core.GameState, new_state: core.GameState) -> Optional[Tuple[int, int]]:
        """Find the move made by comparing two game states (legacy method)."""
        return self._find_ai_move_improved(old_state, new_state)

    def _find_ai_move(self, old_state: core.GameState, new_state: core.GameState) -> Optional[Tuple[int, int]]:
        """Find the move made by comparing two game states."""
        # Look for the cell where count increased
        for r in range(old_state.rows):
            for c in range(old_state.cols):
                old_cell = old_state.board[r][c]
                new_cell = new_state.board[r][c]
                
                # Check if this cell had an orb added
                if (old_cell.owner in (None, 2) and new_cell.owner == 2 and 
                    new_cell.count > old_cell.count):
                    return (r, c)
                
                # Also check if this was an empty cell that got an orb
                if old_cell.owner is None and new_cell.owner == 2:
                    return (r, c)
        
        return None

    def update_ai_vs_ai(self):
        """Update AI vs AI game state - waits for animations to complete."""
        if not self.ai_vs_ai_running or self.state.game_over or self.animating:  # Block AI moves during animation
            return
        
        current_time = time.time()
        if current_time - self.last_ai_move_time < self.ai_move_delay:
            return
        
        current_agent = self.ai_agent1 if self.state.current_player == 1 else self.ai_agent2
        
        try:
            move = current_agent.choose_move(self.state)
            
            # Use animated move system
            if self.create_animated_move(move[0], move[1]):
                self.match_stats.current_game_moves += 1
                self.last_ai_move_time = current_time
                
                if self.state.game_over:
                    threading.Thread(target=self._handle_ai_game_end_after_animation, daemon=True).start()
            
        except Exception as e:
            print(f"AI Error: {e}")
            self.ai_vs_ai_running = False

    def _handle_ai_game_end_after_animation(self):
        """Handle AI game end after animation completes."""
        # Wait for animation to complete
        while self.animating:
            time.sleep(0.1)
        
        self._handle_ai_game_end()

    def _handle_ai_game_end(self):
        """Handle end of AI vs AI game."""
        winner = self.state.get_winner()
        game_duration = time.time() - self.current_game_start_time
        
        self.match_stats.total_games += 1
        self.match_stats.last_winner = winner
        
        if winner == 1:
            self.match_stats.player1_wins += 1
        elif winner == 2:
            self.match_stats.player2_wins += 1
        else:
            self.match_stats.draws += 1
        
        total_moves = (self.match_stats.avg_moves_per_game * (self.match_stats.total_games - 1) + 
                      self.match_stats.current_game_moves)
        self.match_stats.avg_moves_per_game = total_moves / self.match_stats.total_games
        
        total_duration = (self.match_stats.avg_game_duration * (self.match_stats.total_games - 1) + 
                         game_duration)
        self.match_stats.avg_game_duration = total_duration / self.match_stats.total_games
        
        print(f"🏁 Game {self.match_stats.total_games} ended: "
              f"Winner={'Player ' + str(winner) if winner else 'Draw'} "
              f"in {self.match_stats.current_game_moves} moves")
        
        if self.auto_restart:
            time.sleep(2)
            self.state.reset()
            self.visual_state.reset()
            self.current_game_start_time = time.time()
            self.match_stats.current_game_moves = 0
            self.ai_vs_ai_running = True
        else:
            self.ai_vs_ai_running = False

    def _draw_grid(self):
        """Draw the game grid with enhanced styling."""
        # Draw grid lines only in board area
        for r in range(1, 9):
            pygame.draw.line(self.screen, COLORS['border'], 
                           (0, r * CELL_SIZE), (BOARD_WIDTH, r * CELL_SIZE), 1)
        for c in range(1, 6):
            pygame.draw.line(self.screen, COLORS['border'], 
                           (c * CELL_SIZE, 0), (c * CELL_SIZE, BOARD_HEIGHT), 1)
        
        # Draw cell backgrounds
        for r in range(9):
            for c in range(6):
                cell_rect = (c * CELL_SIZE + 1, r * CELL_SIZE + 1, 
                           CELL_SIZE - 2, CELL_SIZE - 2)
                cell_color = COLORS['bg_secondary'] if (r + c) % 2 == 0 else COLORS['bg_primary']
                pygame.draw.rect(self.screen, cell_color, cell_rect)

    def _draw_orbs(self):
        """Draw orbs with enhanced visual effects and animations."""
        offset, base_rad = CELL_SIZE * 0.22, CELL_SIZE * 0.15
        
        # Draw stable orbs from visual state
        for r in range(9):
            for c in range(6):
                cell = self.visual_state.board[r][c]
                if not cell.owner:
                    continue
                
                color = COLORS['player1'] if cell.owner == 1 else COLORS['player2']
                shadow_color = (color[0] // 3, color[1] // 3, color[2] // 3)
                
                cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
                positions = []
                
                if cell.count == 1:
                    positions.append((cx, cy))
                elif cell.count == 2:
                    positions.extend([(cx - offset, cy), (cx + offset, cy)])
                else:
                    positions.extend([(cx - offset, cy - offset), (cx + offset, cy - offset), (cx, cy + offset)])
                    if cell.count == 4:
                        positions.append((cx, cy))
                
                for px, py in positions[:cell.count]:
                    rad = int(base_rad)
                    # Draw shadow
                    pygame.draw.circle(self.screen, shadow_color, (int(px + 2), int(py + 2)), rad)
                    # Draw orb
                    pygame.draw.circle(self.screen, color, (int(px), int(py)), rad)
                    # Draw highlight
                    highlight_color = (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40))
                    pygame.draw.circle(self.screen, highlight_color, (int(px - rad//3), int(py - rad//3)), rad//3)
        
        # Draw animated orb placements
        for placement in self.orb_placement_animations:
            color = COLORS['player1'] if placement.player == 1 else COLORS['player2']
            cx = placement.col * CELL_SIZE + CELL_SIZE // 2
            cy = placement.row * CELL_SIZE + CELL_SIZE // 2
            
            rad = int(base_rad * placement.scale)
            if rad > 0:
                # Animated orb with scaling effect
                pygame.draw.circle(self.screen, color, (cx, cy), rad)
                
                # Glow effect for new orbs
                for i in range(3):
                    glow_rad = rad + (i + 1) * 3
                    glow_intensity = max(0, 100 - i * 30)
                    if glow_rad > 0:
                        glow_surf = pygame.Surface((glow_rad * 2, glow_rad * 2), pygame.SRCALPHA)
                        glow_color = (*color, glow_intensity)
                        pygame.draw.circle(glow_surf, glow_color, (glow_rad, glow_rad), glow_rad)
                        self.screen.blit(glow_surf, (cx - glow_rad, cy - glow_rad), special_flags=pygame.BLEND_ALPHA_SDL2)

    def _draw_explosions(self):
        """Draw explosion effects."""
        for explosion in self.explosion_animations:
            for particle in explosion.particles:
                if particle.life > 0:
                    # Calculate particle properties based on remaining life
                    life_ratio = particle.life / particle.max_life
                    alpha = int(255 * life_ratio)
                    size = max(1, int(3 * life_ratio))
                    
                    # Create particle surface with alpha
                    particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    color_with_alpha = (*particle.color, alpha)
                    pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
                    
                    # Draw particle
                    self.screen.blit(particle_surf, (int(particle.x - size), int(particle.y - size)), 
                                   special_flags=pygame.BLEND_ALPHA_SDL2)

    def _draw_ui(self):
        """Draw compact UI."""
        if self.mode == MODE_AVAI:
            self._draw_ai_vs_ai_ui()
            return
        
        # Compact status bar - clearly below game board
        status_rect = (0, BOARD_HEIGHT, BOARD_WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['bg_menu'], status_rect)
        pygame.draw.line(self.screen, COLORS['border'], (0, BOARD_HEIGHT), (BOARD_WIDTH, BOARD_HEIGHT), 2)
        
        if self.state.game_over:
            win = self.state.get_winner()
            if win:
                txt = f"Player {win} wins!"
                color = COLORS['player1'] if win == 1 else COLORS['player2']
            else:
                txt, color = "Draw!", COLORS['text_primary']
        elif self.animating:
            txt, color = "Animating...", COLORS['warning']
        elif self.mode == MODE_PVAI and self.waiting_for_ai:
            txt, color = "AI thinking...", COLORS['warning']
        else:
            if self.mode == MODE_PVP:
                turn = 'Red' if self.state.current_player == 1 else 'Blue'
                txt = f"{turn}'s turn"
                color = COLORS['text_primary']
            elif self.mode == MODE_PVAI:
                if self.state.current_player == 1:
                    txt = "Your turn"
                else:
                    txt = "AI's turn"
                color = COLORS['text_primary']
            else:
                txt = "Your turn"
                color = COLORS['text_primary']
        
        status_surf = self.fonts['normal'].render(txt, True, color)
        self.screen.blit(status_surf, (10, BOARD_HEIGHT + 10))
        
        # Controls
        if self.mode == MODE_PVAI:
            controls = "ESC-Menu | C-Config | Make sure engine.py is running!"
        else:
            controls = "ESC-Menu | C-Config"
        controls_surf = self.fonts['small'].render(controls, True, COLORS['text_muted'])
        controls_x = BOARD_WIDTH - controls_surf.get_width() - 10
        self.screen.blit(controls_surf, (controls_x, BOARD_HEIGHT + 35))

    def _draw_ai_vs_ai_ui(self):
        """Draw compact AI vs AI UI."""
        # Status bar - clearly below game board
        status_rect = (0, BOARD_HEIGHT, BOARD_WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['bg_menu'], status_rect)
        pygame.draw.line(self.screen, COLORS['border'], (0, BOARD_HEIGHT), (BOARD_WIDTH, BOARD_HEIGHT), 2)
        
        # Game status
        if self.state.game_over:
            winner = self.state.get_winner()
            if winner:
                txt = f"Player {winner} wins!"
                color = COLORS['player1'] if winner == 1 else COLORS['player2']
            else:
                txt, color = "Draw!", COLORS['text_primary']
        elif self.animating:
            txt, color = "Animating...", COLORS['warning']
        elif self.ai_vs_ai_running:
            turn = 'Red' if self.state.current_player == 1 else 'Blue'
            txt = f"{turn}'s turn"
            color = COLORS['player1'] if self.state.current_player == 1 else COLORS['player2']
        else:
            txt = "Paused"
            color = COLORS['warning']
        
        status_surf = self.fonts['normal'].render(txt, True, color)
        self.screen.blit(status_surf, (10, BOARD_HEIGHT + 8))
        
        # Compact stats
        stats_text = f"G:{self.match_stats.total_games} R:{self.match_stats.player1_wins} B:{self.match_stats.player2_wins} M:{self.match_stats.current_game_moves}"
        stats_surf = self.fonts['small'].render(stats_text, True, COLORS['text_secondary'])
        self.screen.blit(stats_surf, (10, BOARD_HEIGHT + 32))
        
        # Controls
        controls = "SPACE-Pause | R-Restart | ESC-Menu"
        controls_surf = self.fonts['small'].render(controls, True, COLORS['text_muted'])
        controls_x = BOARD_WIDTH - controls_surf.get_width() - 10
        self.screen.blit(controls_surf, (controls_x, BOARD_HEIGHT + 32))

    def _write_human_move(self):
        """Write human move to file for AI engine."""
        _write_state("Human Move:", self.state)
        
    def _ai_config_menu(self):
        """Main AI configuration menu for PVAI mode."""
        configuring = True
        selected_option = 0
        
        options = [
            ("P", "Load Preset"),
            ("D", "Change Depth"),
            ("H", "Toggle Heuristics"),
            ("W", "Adjust Weights"),
            ("T", "Test Config"),
            ("A", "Apply Changes"),
        ]
        
        while configuring:
            self.screen.fill(COLORS['bg_primary'])
            
            panel_width = min(380, WIDTH - 20)
            panel_height = min(450, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Configuration")
            
            # Current configuration display
            config_y = content_rect[1] + 10
            
            config_info = [
                f"Depth: {self.ai_config.depth}",
                f"Timeout: {self.ai_config.timeout:.1f}s",
                f"Enabled: {len([h for h, e in self.ai_config.enabled_heuristics.items() if e])}/6 heuristics",
                f"Transposition Table: {'ON' if self.ai_config.use_transposition_table else 'OFF'}",
                f"Move Ordering: {'ON' if self.ai_config.use_move_ordering else 'OFF'}",
            ]
            
            for i, info in enumerate(config_info):
                info_surf = self.fonts['small'].render(info, True, COLORS['text_secondary'])
                self.screen.blit(info_surf, (content_rect[0], config_y + i * 18))
            
            # Options menu
            options_y = config_y + len(config_info) * 18 + 20
            button_height = 25
            button_spacing = 3
            
            for i, (key, title) in enumerate(options):
                button_rect = (content_rect[0], options_y + i * (button_height + button_spacing), 
                             content_rect[2], button_height)
                
                is_selected = (i == selected_option)
                self.draw_button(self.screen, f"{key}. {title}", button_rect, 'small', is_selected)
            
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 30
            controls = "↑↓: Select | ENTER: Choose | B: Back"
            controls_surf = self.fonts['tiny'].render(controls, True, COLORS['text_muted'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        key, _ = options[selected_option]
                        if key == "P":
                            self._preset_menu_for_ai_config()
                        elif key == "D":
                            self._change_depth_for_ai_config()
                        elif key == "H":
                            self._toggle_heuristics_for_ai_config()
                        elif key == "W":
                            self._adjust_weights_for_ai_config()
                        elif key == "T":
                            self._test_ai_config()
                        elif key == "A":
                            self._apply_ai_config_changes()
                    elif event.key == pygame.K_p:
                        self._preset_menu_for_ai_config()
                    elif event.key == pygame.K_d:
                        self._change_depth_for_ai_config()
                    elif event.key == pygame.K_h:
                        self._toggle_heuristics_for_ai_config()
                    elif event.key == pygame.K_w:
                        self._adjust_weights_for_ai_config()
                    elif event.key == pygame.K_t:
                        self._test_ai_config()
                    elif event.key == pygame.K_a:
                        self._apply_ai_config_changes()
                    elif event.key == pygame.K_b or event.key == pygame.K_ESCAPE:
                        configuring = False

    def _preset_menu_for_ai_config(self):
        """Preset selection menu for AI config."""
        selecting = True
        selected = 0
        preset_names = list(self.ai_presets.keys())
            
        descriptions = {
            "Balanced": "Well-rounded strategy",
            "Aggressive": "High-risk, high-reward",
            "Defensive": "Territory focused",
            "Material Only": "Orb counting only"
        }
            
        while selecting:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(300, WIDTH - 40)
            panel_height = min(380, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Presets")
                
            button_height = 30
            button_spacing = 5
            start_y = content_rect[1] + 10
            
            for i, preset_name in enumerate(preset_names):
                button_rect = (content_rect[0], start_y + i * (button_height + button_spacing + 20), 
                            content_rect[2], button_height)
                    
                is_selected = (i == selected)
                self.draw_button(self.screen, preset_name, button_rect, 'normal', is_selected)
                    
                # Description
                desc_y = button_rect[1] + button_height + 2
                desc_surf = self.fonts['small'].render(descriptions.get(preset_name, ""), True, COLORS['text_muted'])
                self.screen.blit(desc_surf, (button_rect[0] + 5, desc_y))
                
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 25
            controls = "↑↓: Select | ENTER: Load | ESC: Cancel"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_secondary'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
                
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

    def _change_depth_for_ai_config(self):
        """Depth adjustment for AI config."""
        changing = True
            
        while changing:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(280, WIDTH - 40)
            panel_height = min(220, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Search Depth")
                
            # Current depth display
            depth_text = f"Current: {self.ai_config.depth}"
            depth_surf = self.fonts['title'].render(depth_text, True, COLORS['text_primary'])
            depth_x = content_rect[0] + (content_rect[2] - depth_surf.get_width()) // 2
            self.screen.blit(depth_surf, (depth_x, content_rect[1] + 20))
                
            # Range info
            range_text = f"Range: {ai.MIN_DEPTH} - {ai.MAX_DEPTH}"
            range_surf = self.fonts['small'].render(range_text, True, COLORS['text_secondary'])
            range_x = content_rect[0] + (content_rect[2] - range_surf.get_width()) // 2
            self.screen.blit(range_surf, (range_x, content_rect[1] + 50))
                
            # Description
            desc_text = "Higher = smarter but slower"
            desc_surf = self.fonts['small'].render(desc_text, True, COLORS['text_muted'])
            desc_x = content_rect[0] + (content_rect[2] - desc_surf.get_width()) // 2
            self.screen.blit(desc_surf, (desc_x, content_rect[1] + 70))
                
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 40
            controls = "← → Adjust | ENTER: Confirm"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_muted'])
            controls_x = content_rect[0] + (content_rect[2] - controls_surf.get_width()) // 2
            self.screen.blit(controls_surf, (controls_x, controls_y))
                
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

    def _toggle_heuristics_for_ai_config(self):
        """Heuristics toggle for AI config."""
        toggling = True
        selected = 0
        heuristic_names = list(self.ai_config.enabled_heuristics.keys())
            
        heuristic_descriptions = {
            'material': 'Count and value orbs',
            'territorial': 'Control territory',
            'critical_mass': 'Explosion threats',
            'mobility': 'Move options',
            'chain_potential': 'Chain reactions',
            'positional': 'Board position'
        }
            
        while toggling:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(400, WIDTH - 20)
            panel_height = min(420, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Heuristics")
                
            # Heuristics list
            item_height = 35
            start_y = content_rect[1] + 10
                
            for i, heuristic in enumerate(heuristic_names):
                enabled = self.ai_config.enabled_heuristics[heuristic]
                y_pos = start_y + i * item_height
                    
                    # Selection highlight
                if i == selected:
                    highlight_rect = (content_rect[0] - 5, y_pos - 2, content_rect[2] + 10, item_height - 5)
                    self.draw_rounded_rect(self.screen, COLORS['highlight'], highlight_rect, 4)
                    
                # Status indicator
                status_color = COLORS['success'] if enabled else COLORS['text_muted']
                status_text = "●" if enabled else "○"
                status_surf = self.fonts['normal'].render(status_text, True, status_color)
                self.screen.blit(status_surf, (content_rect[0], y_pos))
                    
                # Heuristic name and weight
                name_text = heuristic.replace('_', ' ').title()
                weight = self.ai_config.weights.get(heuristic, 0)
                if enabled:
                    display_text = f"{name_text} (Weight: {weight:.1f})"
                    name_color = COLORS['text_primary']
                else:
                    display_text = f"{name_text} (Disabled)"
                    name_color = COLORS['text_muted']
                    
                name_surf = self.fonts['normal'].render(display_text, True, name_color)
                self.screen.blit(name_surf, (content_rect[0] + 25, y_pos))
                    
                # Description
                desc_text = heuristic_descriptions.get(heuristic, "")
                desc_surf = self.fonts['small'].render(desc_text, True, COLORS['text_muted'])
                self.screen.blit(desc_surf, (content_rect[0] + 25, y_pos + 15))
                
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 25
            controls = "↑↓: Select | SPACE: Toggle | ENTER: Done"
            controls_surf = self.fonts['small'].render(controls, True, COLORS['text_secondary'])
            self.screen.blit(controls_surf, (content_rect[0], controls_y))
                
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

    def _adjust_weights_for_ai_config(self):
        """Adjust weights for AI config."""
        adjusting = True
        selected = 0
        heuristic_names = [h for h in self.ai_config.enabled_heuristics.keys() 
                        if self.ai_config.enabled_heuristics[h]]
            
        if not heuristic_names:
            # Show message if no heuristics are enabled
            self._show_message("No Enabled Heuristics", "Enable some heuristics first!")
            return
            
        while adjusting:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(380, WIDTH - 20)
            panel_height = min(400, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "Adjust Weights")
                
            # Weights list
            item_height = 30
            start_y = content_rect[1] + 10
                
            for i, heuristic in enumerate(heuristic_names):
                weight = self.ai_config.weights[heuristic]
                y_pos = start_y + i * item_height
                    
                # Selection highlight
                if i == selected:
                    highlight_rect = (content_rect[0] - 5, y_pos - 2, content_rect[2] + 10, item_height - 5)
                    self.draw_rounded_rect(self.screen, COLORS['highlight'], highlight_rect, 4)
                    
                # Heuristic name and weight
                name_text = heuristic.replace('_', ' ').title()
                display_text = f"{name_text}: {weight:.1f}"
                    
                name_surf = self.fonts['normal'].render(display_text, True, COLORS['text_primary'])
                self.screen.blit(name_surf, (content_rect[0], y_pos))
                    
                # Weight bar
                bar_width = 100
                bar_height = 8
                bar_x = content_rect[0] + content_rect[2] - bar_width - 10
                bar_y = y_pos + 8
                    
                # Background bar
                bar_bg_rect = (bar_x, bar_y, bar_width, bar_height)
                pygame.draw.rect(self.screen, COLORS['bg_secondary'], bar_bg_rect, border_radius=4)
                    
                # Weight bar (normalize to 0-10 range for display)
                weight_ratio = min(1.0, weight / 10.0)
                weight_width = int(bar_width * weight_ratio)
                if weight_width > 0:
                    weight_rect = (bar_x, bar_y, weight_width, bar_height)
                    color = COLORS['success'] if weight <= 5 else COLORS['warning'] if weight <= 8 else COLORS['danger']
                    pygame.draw.rect(self.screen, color, weight_rect, border_radius=4)
                
            # Controls
            controls_y = content_rect[1] + content_rect[3] - 40
            controls = ["↑↓: Select | ← → Adjust Weight", "ENTER: Done | ESC: Cancel"]
            for i, control in enumerate(controls):
                control_surf = self.fonts['small'].render(control, True, COLORS['text_muted'])
                self.screen.blit(control_surf, (content_rect[0], controls_y + i * 15))
                
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
                        current_weight = self.ai_config.weights[heuristic]
                        new_weight = max(0.1, current_weight - 0.1)
                        self.ai_config.set_weight(heuristic, new_weight)
                    elif event.key == pygame.K_RIGHT:
                        heuristic = heuristic_names[selected]
                        current_weight = self.ai_config.weights[heuristic]
                        new_weight = min(10.0, current_weight + 0.1)
                        self.ai_config.set_weight(heuristic, new_weight)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        adjusting = False

    def _test_ai_config(self):
        """Test current AI configuration."""
        testing = True
            
        # Create test agent with current config
        test_agent = ai.MinimaxAgent(player=2, config=self.ai_config)
            
        # Create test scenario
        test_state = core.GameState(rows=9, cols=6)
        test_state.board[2][2].owner = 1
        test_state.board[2][2].count = 2
        test_state.board[6][4].owner = 2
        test_state.board[6][4].count = 1
        test_state.current_player = 2
            
        # Get AI move and stats
        start_time = time.time()
        try:
            ai_move = test_agent.choose_move(test_state.clone())
            test_time = time.time() - start_time
            stats = test_agent.get_search_statistics()
            success = True
        except Exception as e:
            test_time = time.time() - start_time
            error_msg = str(e)
            success = False
            
        while testing:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(380, WIDTH - 20)
            panel_height = min(350, HEIGHT - 20)
            panel_rect = (10, 10, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, "AI Test Results")
                
            y_pos = content_rect[1] + 10
            line_height = 20
                
            if success:
                results = [
                    f"✓ Test completed successfully",
                    f"AI Move: {ai_move}",
                    f"Time: {test_time:.3f}s",
                    f"Nodes Explored: {stats['nodes_explored']}",
                    f"Search Depth: {stats['search_depth']}",
                    f"Alpha-Beta Cutoffs: {stats['alpha_beta_cutoffs']}",
                    f"Table Hit Rate: {stats['hit_rate_percent']:.1f}%",
                ]
                    
                for i, result in enumerate(results):
                    color = COLORS['success'] if result.startswith("✓") else COLORS['text_primary']
                    result_surf = self.fonts['small'].render(result, True, color)
                    self.screen.blit(result_surf, (content_rect[0], y_pos + i * line_height))
            else:
                error_results = [
                    f"✗ Test failed",
                    f"Time: {test_time:.3f}s",
                    f"Error: {error_msg[:50]}{'...' if len(error_msg) > 50 else ''}",
                ]
                    
                for i, result in enumerate(error_results):
                    color = COLORS['danger'] if result.startswith("✗") else COLORS['text_primary']
                    result_surf = self.fonts['small'].render(result, True, color)
                    self.screen.blit(result_surf, (content_rect[0], y_pos + i * line_height))
                
            # Continue prompt
            continue_y = content_rect[1] + content_rect[3] - 25
            continue_surf = self.fonts['small'].render("Press any key to continue", True, COLORS['text_muted'])
            continue_x = content_rect[0] + (content_rect[2] - continue_surf.get_width()) // 2
            self.screen.blit(continue_surf, (continue_x, continue_y))
                
            pygame.display.flip()
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    testing = False

    def _apply_ai_config_changes(self):
        """Apply and save AI configuration changes."""
        self._show_message("Configuration Applied", "AI settings have been updated!")

    def _show_message(self, title: str, message: str):
        """Show a simple message dialog."""
        showing = True
            
        while showing:
            self.screen.fill(COLORS['bg_primary'])
                
            panel_width = min(320, WIDTH - 40)
            panel_height = min(180, HEIGHT - 40)
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2
            panel_rect = (panel_x, panel_y, panel_width, panel_height)
            content_rect = self.draw_panel(self.screen, panel_rect, title)
                
            # Message
            message_surf = self.fonts['normal'].render(message, True, COLORS['text_primary'])
            message_x = content_rect[0] + (content_rect[2] - message_surf.get_width()) // 2
            message_y = content_rect[1] + 30
            self.screen.blit(message_surf, (message_x, message_y))
                
            # OK button
            ok_y = content_rect[1] + content_rect[3] - 40
            ok_surf = self.fonts['small'].render("Press any key to continue", True, COLORS['text_muted'])
            ok_x = content_rect[0] + (content_rect[2] - ok_surf.get_width()) // 2
            self.screen.blit(ok_surf, (ok_x, ok_y))
                
            pygame.display.flip()
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    showing = False

    def run(self):
        """Main game loop with proper mode handling."""
        while True:
            # If no mode is selected, show main menu
            if self.mode is None:
                self._main_menu()
                continue
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        if not self.animating:  # Don't allow config during animation
                            if self.mode == MODE_AVAI:
                                self._ai_vs_ai_config_menu()
                            else:
                                self._ai_config_menu()
                    elif event.key == pygame.K_SPACE and self.mode == MODE_AVAI:
                        self.ai_vs_ai_running = not self.ai_vs_ai_running
                        if self.ai_vs_ai_running:
                            self.last_ai_move_time = time.time()
                    elif event.key == pygame.K_r and self.mode == MODE_AVAI:
                        if not self.animating:  # Don't allow restart during animation
                            self.state.reset()
                            self.visual_state.reset()
                            self.animations.clear()
                            self.explosion_animations.clear()
                            self.orb_placement_animations.clear()
                            self.animation_queue.clear()
                            self.animating = False
                            self.match_stats = MatchStats()
                            self.current_game_start_time = time.time()
                            self.ai_vs_ai_running = True
                    elif event.key == pygame.K_ESCAPE:
                        if not self.animating:  # Don't allow menu during animation
                            self.mode = None  # This will trigger main menu on next loop
                            continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            # Mode-specific updates
            if self.mode == MODE_AVAI:
                self.update_ai_vs_ai()
            elif self.mode == MODE_PVAI:
                self._check_for_ai_response()

            # Update animations
            self.update_animations()

            # Legacy animation update for compatibility
            now = pygame.time.get_ticks()
            self.animations[:] = [anim for anim in self.animations
                                 if now - anim.start_time < EXPLOSION_DURATION_MS]

            # Rendering
            self.screen.fill(COLORS['bg_primary'])
            self._draw_grid()
            self._draw_orbs()
            self._draw_explosions()
            self._draw_ui()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    ChainReactionGUI().run()