import heapq
import math
from copy import deepcopy

class PuzzleState:
    def __init__(self, board, g, parent, heuristic_func, goal):
        self.board = board
        self.g = g
        self.parent = parent
        self.heuristic_func = heuristic_func
        self.goal = goal
        self.h = heuristic_func(board, goal)
        self.f = self.g + self.h

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return self.board == other.board

    def get_blank_pos(self):
        for i, row in enumerate(self.board):
            for j, val in enumerate(row):
                if val == 0:
                    return (i, j)

    def get_neighbors(self):
        x, y = self.get_blank_pos()
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.board) and 0 <= ny < len(self.board[0]):
                new_board = deepcopy(self.board)
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(new_board)
        return neighbors

# ------------------- Heuristics ------------------- #
def hamming(board, goal):
    return sum(
        board[i][j] != goal[i][j] and board[i][j] != 0
        for i in range(len(board))
        for j in range(len(board[0]))
    )

def manhattan(board, goal):
    dist = 0
    k = len(board)
    pos = {goal[i][j]: (i, j) for i in range(k) for j in range(k)}
    for i in range(k):
        for j in range(k):
            val = board[i][j]
            if val != 0:
                gi, gj = pos[val]
                dist += abs(i - gi) + abs(j - gj)
    return dist

def euclidean(board, goal):
    dist = 0
    k = len(board)
    pos = {goal[i][j]: (i, j) for i in range(k) for j in range(k)}
    for i in range(k):
        for j in range(k):
            val = board[i][j]
            if val != 0:
                gi, gj = pos[val]
                dist += math.sqrt((i - gi)**2 + (j - gj)**2)
    return dist

def linear_conflict(board, goal):
    man = manhattan(board, goal)
    k = len(board)
    conflict = 0
    for row in range(k):
        max_col = -1
        for col in range(k):
            val = board[row][col]
            if val == 0: continue
            goal_row, goal_col = divmod(val - 1, k)
            if goal_row == row:
                if goal_col < max_col:
                    conflict += 1
                else:
                    max_col = goal_col
    for col in range(k):
        max_row = -1
        for row in range(k):
            val = board[row][col]
            if val == 0: continue
            goal_row, goal_col = divmod(val - 1, k)
            if goal_col == col:
                if goal_row < max_row:
                    conflict += 1
                else:
                    max_row = goal_row
    return man + 2 * conflict

# ------------------- Solvability ------------------- #
def count_inversions(flat_board):
    inv = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] and flat_board[j] and flat_board[i] > flat_board[j]:
                inv += 1
    return inv

def find_blank_row_from_bottom(board):
    k = len(board)
    for i in range(k - 1, -1, -1):
        for j in range(k):
            if board[i][j] == 0:
                return k - i

def is_solvable(board):
    flat = [num for row in board for num in row]
    k = int(len(flat)**0.5)
    inv = count_inversions(flat)
    if k % 2 == 1:
        return inv % 2 == 0
    else:
        row_from_bottom = find_blank_row_from_bottom(board)
        return (inv + row_from_bottom) % 2 == 0

# ------------------- A* Solver ------------------- #
def a_star(start_board, goal_board, heuristic_func):
    open_list = []
    closed_set = set()
    nodes_expanded = 0
    nodes_explored = 0  

    start = PuzzleState(start_board, 0, None, heuristic_func, goal_board)
    heapq.heappush(open_list, start)
    nodes_explored += 1 

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if current.board == goal_board:
            path = []
            while current:
                path.append(current.board)
                current = current.parent
            path.reverse()
            return path, nodes_expanded, nodes_explored

        closed_set.add(tuple(map(tuple, current.board)))

        for neighbor in current.get_neighbors():
            if tuple(map(tuple, neighbor)) in closed_set:
                continue
            neighbor_state = PuzzleState(neighbor, current.g + 1, current, heuristic_func, goal_board)
            heapq.heappush(open_list, neighbor_state)
            nodes_explored += 1 

    return None, nodes_expanded, nodes_explored


# ------------------- Input/Run ------------------- #
def parse_input():
    size = int(input())
    board = [list(map(int, input().split())) for _ in range(size)]
    return size, board

def get_goal_board(size):
    nums = list(range(1, size * size)) + [0]
    return [nums[i*size:(i+1)*size] for i in range(size)]

def main():
    size, board = parse_input()
    goal = get_goal_board(size)

    heuristic_map = {
        "manhattan": manhattan,
        "hamming": hamming,
        "euclidean": euclidean,
        "linear": linear_conflict
    }

    heuristic_name = input("Choose heuristic (manhattan/hamming/euclidean/linear): ").strip()
    heuristic = heuristic_map.get(heuristic_name, manhattan)

    if not is_solvable(board):
        print("Unsolvable puzzle")
        return

    path, nodes_expanded, nodes_explored = a_star(board, goal, heuristic)
    print(f"Minimum number of moves = {len(path) - 1}")
    for p in path:
        for row in p:
            print(" ".join(map(str, row)))
        print()
    print("Nodes expanded:", nodes_expanded)
    print("Nodes explored:", nodes_explored)

if __name__ == "__main__":
    main()
