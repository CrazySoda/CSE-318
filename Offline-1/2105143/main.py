from board_utils import is_solvable, get_goal_board
from heuristics import manhattan,hamming,euclidean,linear_conflict
from solver import a_star

def parse_input():
    size = int(input())
    board = [list(map(int , input().split()))for _ in range(size)]
    return size , board

def main():
    size, board = parse_input()
    goal = get_goal_board(size) 
    
    
    if not is_solvable(board):
        print("Unsolvable puzzle")
        return
    
    heuristic_map = {
        "manhattan": manhattan,
        "hamming":hamming,
        "euclidean":euclidean,
        "linear_conflict":linear_conflict
    }
    
    heuristic_name = input("Choose heuristic(manhattan/hamming/euclidean/linear_conflict)").strip()
    
    if heuristic_name not in heuristic_map:
        print("Invalid heuristic mode. Please try again")
        return
    
    heuristic = heuristic_map[heuristic_name]
    
    path, nodes_expanded, nodes_explored = a_star(board, goal, heuristic)
    print(f"Minimum number of moves = {len(path)- 1}")
    for p in path:
        for row in p:
            print(" ".join(map(str,row)))
        print()
    print("Nodes expanded:", nodes_expanded)
    print("Nodes explored:", nodes_explored)
    
    
if __name__ == "__main__": 
    main()