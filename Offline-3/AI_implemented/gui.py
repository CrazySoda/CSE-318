"""
Enhanced Compact GUI with sequential animations
Features: Modern colors, compact menus, step-by-step chain reactions, AI waits for animations
"""

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

    def _main_menu(self):
        """Compact main menu with better visual design."""
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

        self._reset_board()

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
                        configuring = False

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

    def _start_ai_vs_ai(self):
        """Start AI vs AI match."""
        self.ai_agent1 = ai.MinimaxAgent(player=1, config=self.ai_player1_config)
        self.ai_agent2 = ai.MinimaxAgent(player=2, config=self.ai_player2_config)
        self.ai_vs_ai_running = True
        self.current_game_start_time = time.time()
        self.match_stats.current_game_moves = 0
        print("🤖 AI vs AI match started!")

    def _ai_config_menu(self):
        """Main AI configuration menu - kept compact."""
        # This is a simplified version - you can expand it similar to the AI vs AI menu
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

    def _reset_board(self):
        """Reset the game board."""
        self.state.reset()
        self.visual_state.reset()
        self.animations.clear()
        self.explosion_animations.clear()
        self.orb_placement_animations.clear()
        self.animation_queue.clear()
        self.animating = False
        self.ai_vs_ai_running = False
        
        if self.mode == MODE_PVAI and not os.path.exists(FILE):
            _write_state("AI Move:", self.state)
        elif self.mode == MODE_AVAI:
            self._ai_vs_ai_config_menu()

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
        if self.mode == MODE_AVAI or self.animating:  # Block clicks during animation
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
                threading.Thread(target=self._handle_ai_move_after_animation, daemon=True).start()

    def _handle_ai_move_after_animation(self):
        """Handle AI move after human move animation completes."""
        # Wait for animation to complete
        while self.animating:
            time.sleep(0.1)
        
        if not self.state.game_over:
            self._write_human_move()
            self._wait_ai()

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
        else:
            if self.mode == MODE_PVP:
                turn = 'Red' if self.state.current_player == 1 else 'Blue'
                txt = f"{turn}'s turn"
            else:
                txt = "Your turn"
            color = COLORS['text_primary']
        
        status_surf = self.fonts['normal'].render(txt, True, color)
        self.screen.blit(status_surf, (10, BOARD_HEIGHT + 10))
        
        # Controls
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

    def run(self):
        """Main game loop."""
        self._main_menu()
        
        while True:
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
                            self._main_menu()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            if self.mode == MODE_AVAI:
                self.update_ai_vs_ai()

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