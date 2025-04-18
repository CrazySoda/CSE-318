import heapq
from copy import deepcopy

class PuzzleState:
    def __init__(self,board,g,parent,heuristic_func,goal):
        self.board = board
        self.g = g
        self.parent = parent
        self.heuristic_func = heuristic_func
        self.goal = goal
        self.h = heuristic_func(board,goal)
        self.f = self.g + self.h
        
    def __lt__(self,other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return self.board == other.board
    
    def get_blank_pos(self):
        for i, row in enumerate(self.board):
            for j,val in enumerate(row):
                if val == 0:
                    return(i,j)
                
    def get_neighbours(self):
        x,y = self.get_blank_pos()
        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        neighbours = []
        for dx,dy in moves:
            nx,ny = x+dx , y+dy
            if 0 <= nx < len(self.board) and 0 <= ny  < len(self.board[0]):
                new_board = deepcopy(self.board)
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbours.append(new_board)
        return neighbours
    

#A* ALgorithm

def a_star(start_board, goal_board,heuristic_func):
    open_list = []
    closed_set = set()
    nodes_expanded = 0
    nodes_explored = 0
    
    start = PuzzleState(start_board, 0 , None, heuristic_func, goal_board)
    heapq.heappush(open_list, start)
    nodes_explored += 1
    
    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1
        
        #if the current being explored is the goal board
        if current.board == goal_board:
            path = []
            while current:
                path.append(current.board)
                current = current.parent
            path.reverse()
            return path, nodes_expanded, nodes_explored
        
        #convert the list into a tuple of tuples
        closed_set.add(tuple(map(tuple, current.board)))
        
        for neighbour in current.get_neighbours():
            if tuple(map(tuple,neighbour)) in closed_set:
                continue
            neighbour_state = PuzzleState(neighbour, current.g + 1, current , heuristic_func, goal_board)
            heapq.heappush(open_list,neighbour_state)
            nodes_explored +=1
            
    return None, nodes_expanded, nodes_explored