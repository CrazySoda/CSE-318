import tkinter as tk
import copy

ROWS, COLS = 9, 6
CELL_SIZE = 60

CRITICAL_MASS = [[
    2 if (r in [0, ROWS-1] and c in [0, COLS-1]) else
    3 if (r in [0, ROWS-1] or c in [0, COLS-1]) else 4
    for c in range(COLS)] for r in range(ROWS)]

class Game:
    def __init__(self):
        self.board = [[(0, None) for _ in range(COLS)] for _ in range(ROWS)]
        self.turn = 'R'
    
    def clone(self):
        new_game = Game()
        new_game.board = copy.deepcopy(self.board)
        new_game.turn = self.turn
        return new_game
    
    def inside(self, r, c):
        return 0 <= r < ROWS and 0 <= c < COLS

    def explode(self, r, c):
        count, owner = self.board[r][c]
        if count < CRITICAL_MASS[r][c]:
            return
        self.board[r][c] = (count - CRITICAL_MASS[r][c], None)
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if self.inside(nr, nc):
                n_count, _ = self.board[nr][nc]
                self.board[nr][nc] = (n_count + 1, owner)
                self.explode(nr, nc)

    def apply_move(self, r, c):
        count, owner = self.board[r][c]
        if owner not in [None, self.turn]:
            return False
        self.board[r][c] = (count + 1, self.turn)
        self.explode(r, c)
        self.turn = 'B' if self.turn == 'R' else 'R'
        return True

    def legal_moves(self, color):
        return [(r, c) for r in range(ROWS) for c in range(COLS)
                if self.board[r][c][1] in [None, color]]

    def winner(self):
        players = set(owner for row in self.board for _, owner in row if owner)
        return players.pop() if len(players) == 1 else None

    def score(self, color):
        return sum(1 for row in self.board for _, owner in row if owner == color)

# Basic Heuristic
def heuristic(game, color):
    return game.score(color) - game.score('B' if color == 'R' else 'R')

def minimax(game, depth, alpha, beta, maximizing, color):
    winner = game.winner()
    if winner == color:
        return float('inf'), None
    elif winner is not None:
        return float('-inf'), None
    if depth == 0:
        return heuristic(game, color), None
    
    best = None
    if maximizing:
        max_eval = float('-inf')
        for move in game.legal_moves(color):
            new_game = game.clone()
            new_game.apply_move(*move)
            eval, _ = minimax(new_game, depth-1, alpha, beta, False, color)
            if eval > max_eval:
                max_eval, best = eval, move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best
    else:
        min_eval = float('inf')
        opp_color = 'B' if color == 'R' else 'R'
        for move in game.legal_moves(opp_color):
            new_game = game.clone()
            new_game.apply_move(*move)
            eval, _ = minimax(new_game, depth-1, alpha, beta, True, color)
            if eval < min_eval:
                min_eval, best = eval, move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best

# UI
class ChainReactionUI:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)
        self.game = Game()
        self.ai_color = 'B'
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c*CELL_SIZE, r*CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")
                count, owner = self.game.board[r][c]
                if owner:
                    color = "red" if owner == 'R' else "blue"
                    for i in range(count):
                        cx = x1 + (i+1)*CELL_SIZE/(count+1)
                        cy = y1 + CELL_SIZE/2
                        self.canvas.create_oval(cx-8, cy-8, cx+8, cy+8, fill=color)
        self.root.update()

        if self.game.turn == self.ai_color and not self.game.winner():
            self.root.after(500, self.ai_move)

    def click(self, event):
        if self.game.turn != 'R':
            return
        r, c = event.y // CELL_SIZE, event.x // CELL_SIZE
        if self.game.apply_move(r, c):
            self.draw()
            if self.game.winner():
                self.show_winner()

    def ai_move(self):
        _, move = minimax(self.game, 3, float('-inf'), float('inf'), True, self.ai_color)
        if move:
            self.game.apply_move(*move)
            self.draw()
            if self.game.winner():
                self.show_winner()

    def show_winner(self):
        win = self.game.winner()
        self.canvas.create_text(COLS*CELL_SIZE//2, ROWS*CELL_SIZE//2,
                                text=f"{'Red' if win == 'R' else 'Blue'} Wins!",
                                font=("Arial", 32), fill="green")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chain Reaction (CSE 318)")
    app = ChainReactionUI(root)
    root.mainloop()
