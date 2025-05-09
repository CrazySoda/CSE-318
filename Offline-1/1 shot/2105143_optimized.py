

import heapq, math, sys
# ──────────────────────────── Helpers ──────────────────────────── #
def flatten(board):
    """[[…]] ⟶ tuple[…]"""
    return tuple(num for row in board for num in row)

def unflatten(t, k):
    return [list(t[i*k:(i+1)*k]) for i in range(k)]

def goal_positions(goal_flat, k):
    """value ↦ (row,col) for fast look‑ups in heuristics."""
    return {v: divmod(i, k) for i, v in enumerate(goal_flat)}

# Perfect hash (Cantor / factoradic rank) ------------------------- #
from math import factorial
def precompute_factorials(n):
    fac = [1]
    for i in range(1, n+1):
        fac.append(fac[-1]*i)       # iterative avoids math.factorial calls
    return fac

def cantor_rank(perm, factorials):
    """Unique integer for the permutation `perm` (tuple)."""
    rank = 0
    n = len(perm)
    for i in range(n):
        smaller = sum(1 for j in range(i+1, n) if perm[j] < perm[i])
        rank += smaller * factorials[n-i-1]
    return rank

# ───────────────────────── Heuristics ──────────────────────────── #
def manhattan(board, pos_tbl):
    k = int(len(board)**0.5)
    return sum(abs(r-pos_tbl[v][0]) + abs(c-pos_tbl[v][1])
               for i, v in enumerate(board) if v
               for r, c in [divmod(i, k)])

def euclidean(board, pos_tbl):
    k = int(len(board)**0.5)
    return sum(math.hypot(r-pos_tbl[v][0], c-pos_tbl[v][1])
               for i, v in enumerate(board) if v
               for r, c in [divmod(i, k)])

def hamming(board, pos_tbl):
    k = int(len(board)**0.5)
    return sum(1 for i, v in enumerate(board) if v and divmod(i,k)!=pos_tbl[v])

def linear_conflict(board, pos_tbl):
    k   = int(len(board)**0.5)
    man = manhattan(board, pos_tbl)
    conflict = 0
    # rows
    for r in range(k):
        max_c = -1
        for c in range(k):
            v = board[r*k + c]
            if v and pos_tbl[v][0] == r:
                gc = pos_tbl[v][1]
                if gc < max_c:
                    conflict += 1
                else:
                    max_c = gc
    # cols
    for c in range(k):
        max_r = -1
        for r in range(k):
            v = board[r*k + c]
            if v and pos_tbl[v][1] == c:
                gr = pos_tbl[v][0]
                if gr < max_r:
                    conflict += 1
                else:
                    max_r = gr
    return man + 2*conflict

# ───────────────────────── Puzzle state ───────────────────────── #
class PuzzleState:
    __slots__ = ("board","g","h","f","parent")
    MOVES = (-1, 1, 0, 0,   0, 0, -1, 1)  # dx,dy encoded

    def __init__(self, board, g, parent, h_func, pos_tbl):
        self.board  = board
        self.g      = g
        self.h      = h_func(board, pos_tbl)
        self.f      = self.g + self.h
        self.parent = parent

    def __lt__(self, other):
        return (self.f, self.h) < (other.f, other.h)

    def neighbors(self, k):
        z = self.board.index(0)
        zx, zy = divmod(z, k)
        for dx, dy in zip(self.MOVES[:4], self.MOVES[4:]):
            nx, ny = zx+dx, zy+dy
            if 0 <= nx < k and 0 <= ny < k:
                nz = nx*k + ny
                nb = list(self.board)
                nb[z], nb[nz] = nb[nz], nb[z]
                yield tuple(nb)

# ─────────────────────── Solvability check ─────────────────────── #
def is_solvable(flat):
    inv = sum(1
        for i in range(len(flat))
        for j in range(i+1, len(flat))
        if flat[i] and flat[j] and flat[i] > flat[j])
    k = int(len(flat)**0.5)
    if k & 1:
        return inv & 1 == 0
    row_from_bottom = k - flat.index(0)//k
    return (inv + row_from_bottom) & 1 == 0

# ─────────────────────────── A* search ─────────────────────────── #
def a_star(start_board, goal_board, h_func):
    k            = len(start_board)
    start_flat   = flatten(start_board)
    goal_flat    = flatten(goal_board)
    pos_tbl      = goal_positions(goal_flat, k)

    facts        = precompute_factorials(k*k)      # for Cantor ranking
    closed_set   = set()                           # stores ints, not tuples
    best_g       = {start_flat: 0}

    open_pq      = []
    heapq.heappush(open_pq, PuzzleState(start_flat, 0, None, h_func, pos_tbl))
    explored = expanded = 1

    while open_pq:
        cur = heapq.heappop(open_pq)
        expanded += 1

        if cur.board == goal_flat:
            path = []
            while cur:
                path.append(unflatten(cur.board, k))
                cur = cur.parent
            return path[::-1], expanded, explored

        closed_set.add(cantor_rank(cur.board, facts))

        for nb in cur.neighbors(k):
            if cantor_rank(nb, facts) in closed_set:
                continue
            g_nb = cur.g + 1
            if g_nb < best_g.get(nb, 1e9):
                best_g[nb] = g_nb
                heapq.heappush(open_pq,
                    PuzzleState(nb, g_nb, cur, h_func, pos_tbl))
                explored += 1
    return None, expanded, explored  # should never hit for solvable input

# ──────────────────────────── I/O glue ─────────────────────────── #
def read_board():
    k = int(sys.stdin.readline())
    board = [list(map(int, sys.stdin.readline().split())) for _ in range(k)]
    return k, board

def goal_board(k):
    nums = list(range(1, k*k)) + [0]
    return [nums[i*k:(i+1)*k] for i in range(k)]

if __name__ == "__main__":
    k, start = read_board()
    goal     = goal_board(k)

    heuristics = {
        "manhattan": manhattan,
        "hamming"  : hamming,
        "euclidean": euclidean,
        "linear"   : linear_conflict
    }
    choice = input(
        "Choose heuristic (manhattan/hamming/euclidean/linear): "
    ).strip().lower()
    h_func = heuristics.get(choice, manhattan)

    if not is_solvable(flatten(start)):
        print("Unsolvable puzzle"); sys.exit()

    path, expanded, explored = a_star(start, goal, h_func)
    print("Minimum moves =", len(path)-1)
    for p in path:
        print("\n".join(" ".join(map(str,row)) for row in p), "\n")
    print("Nodes expanded:", expanded)
    print("Nodes explored:", explored)
