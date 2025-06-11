import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 600
ROWS, COLS = 6, 9
CELL_SIZE = WIDTH // COLS
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
COLORS = [RED, BLUE, GREEN, YELLOW]

# Function to get critical mass for any cell
def get_critical_mass(row, col):
    """Return the critical mass for a given cell based on its position."""
    # Corner cells: 2 neighbors
    if (row == 0 or row == ROWS-1) and (col == 0 or col == COLS-1):
        return 2
    # Edge cells: 3 neighbors
    elif row == 0 or row == ROWS-1 or col == 0 or col == COLS-1:
        return 3
    # Inner cells: 4 neighbors
    else:
        return 4

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chain Reaction Game")

# Font for displaying text
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# Game state
board = [[{'color': None, 'count': 0} for _ in range(COLS)] for _ in range(ROWS)]
turn = 0  # 0 for Player 1 (Red), 1 for Player 2 (Blue)
game_over = False
winner = None
moves_made = 0  # Track total moves made by both players

def draw_board():
    """Draw the game board."""
    screen.fill(WHITE)
    
    for row in range(ROWS):
        for col in range(COLS):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            
            # Draw cell background
            pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)
            
            cell = board[row][col]
            if cell['color'] is not None and cell['count'] > 0:
                # Draw atoms based on count
                center_x, center_y = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                radius = CELL_SIZE // 8
                
                if cell['count'] == 1:
                    pygame.draw.circle(screen, cell['color'], (center_x, center_y), radius)
                elif cell['count'] == 2:
                    pygame.draw.circle(screen, cell['color'], (center_x - radius//2, center_y), radius)
                    pygame.draw.circle(screen, cell['color'], (center_x + radius//2, center_y), radius)
                elif cell['count'] == 3:
                    pygame.draw.circle(screen, cell['color'], (center_x, center_y - radius//2), radius)
                    pygame.draw.circle(screen, cell['color'], (center_x - radius//2, center_y + radius//2), radius)
                    pygame.draw.circle(screen, cell['color'], (center_x + radius//2, center_y + radius//2), radius)
                
                # Show count number
                count_text = small_font.render(str(cell['count']), True, BLACK)
                screen.blit(count_text, (x + 5, y + 5))
    
    # Draw current player indicator
    if not game_over:
        player_text = font.render(f"Player {turn + 1}'s Turn", True, COLORS[turn])
        screen.blit(player_text, (10, HEIGHT - 40))
    else:
        winner_text = font.render(f"Player {winner} Wins!", True, COLORS[winner-1])
        screen.blit(winner_text, (10, HEIGHT - 40))
    
    pygame.display.flip()

def get_cell(pos):
    """Return the row and column of the cell at the given position."""
    x, y = pos
    return y // CELL_SIZE, x // CELL_SIZE

def is_valid_move(row, col):
    """Check if the move is valid."""
    if not (0 <= row < ROWS and 0 <= col < COLS):
        return False
    cell = board[row][col]
    # Can only place on empty cells or cells you own
    return cell['color'] is None or cell['color'] == COLORS[turn]

def get_neighbors(row, col):
    """Return the neighboring cells."""
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            neighbors.append((nr, nc))
    return neighbors

def explode_cell(row, col):
    """Handle the explosion of a single cell."""
    cell = board[row][col]
    color = cell['color']
    
    # Reset the exploding cell
    board[row][col] = {'color': None, 'count': 0}
    
    # Distribute atoms to neighbors
    neighbors = get_neighbors(row, col)
    for nr, nc in neighbors:
        neighbor = board[nr][nc]
        neighbor['color'] = color
        neighbor['count'] += 1

def check_explosions():
    """Check for and handle all explosions in sequence."""
    explosion_occurred = True
    
    while explosion_occurred:
        explosion_occurred = False
        cells_to_explode = []
        
        # Find all cells that should explode
        for row in range(ROWS):
            for col in range(COLS):
                cell = board[row][col]
                if cell['count'] >= get_critical_mass(row, col):
                    cells_to_explode.append((row, col))
        
        # Explode all critical cells
        for row, col in cells_to_explode:
            explode_cell(row, col)
            explosion_occurred = True
        
        # Small delay for visual effect
        if explosion_occurred:
            draw_board()
            pygame.time.wait(200)

def check_for_winner():
    """Check if there's a winner."""
    global game_over, winner
    
    # Don't check for winner until both players have made at least one move
    if moves_made < 2:
        return
    
    # Count total atoms for each player
    player_atoms = {}
    
    for row in range(ROWS):
        for col in range(COLS):
            cell = board[row][col]
            if cell['color'] is not None and cell['count'] > 0:
                color = cell['color']
                if color not in player_atoms:
                    player_atoms[color] = 0
                player_atoms[color] += cell['count']
    
    # If only one player has atoms remaining, they win
    if len(player_atoms) == 1:
        winning_color = list(player_atoms.keys())[0]
        winner = COLORS.index(winning_color) + 1
        game_over = True
        print(f"Player {winner} wins!")

def make_move(row, col):
    """Make a move at the specified position."""
    global turn, moves_made
    
    if game_over:
        return
    
    cell = board[row][col]
    cell['color'] = COLORS[turn]
    cell['count'] += 1
    moves_made += 1
    
    # Check for explosions
    check_explosions()
    
    # Check for winner
    check_for_winner()
    
    # Switch turns
    if not game_over:
        turn = 1 - turn

def restart_game():
    """Restart the game by resetting all variables."""
    global board, turn, game_over, winner, moves_made
    board = [[{'color': None, 'count': 0} for _ in range(COLS)] for _ in range(ROWS)]
    turn = 0
    game_over = False
    winner = None
    moves_made = 0

def main():
    """Main game loop."""
    clock = pygame.time.Clock()
    
    while True:
        draw_board()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if event.button == 1:  # Left click
                    row, col = get_cell(event.pos)
                    if is_valid_move(row, col):
                        make_move(row, col)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'R' to restart
                    restart_game()
        
        clock.tick(FPS)

if __name__ == "__main__":
    main()