import math

def hamming(board, goal):
    return sum(
        board[i][j] != goal[i][j] and board[i][j]!= 0
        for i in range (len(board))
        for j in range (len(board[0]))
    )
    
def manhattan(board, goal):
    dist = 0
    k = len(board)
    pos = {goal[i][j]:(i,j) for i in range(k) for j in range(k)}#make a dictionary to assign a x,y value for a goal board
    for i in range(k):
        for j in range(k):
            val = board[i][j]
            if val!=0:
                gi,gj = pos[val]
                dist += abs(i - gi) + abs(j-gj)
    return dist

def euclidean(board,goal):
    dist = 0
    k = len(board)
    pos = {goal[i][j]:(i,j) for i in range(k) for j in range(k)}
    for i in range(k):
        for j in range(k):
            val = board[i][j]
            if val != 0:
                gi,gj = pos[val]
                dist += math.sqrt((i-gi)**2 + (j-gj)**2)
    return dist

def linear_conflict(board,goal):
    man = manhattan(board,goal)
    k = len(board)
    conflict = 0
    for row in range(k):
        max_col = -1
        for col in range(k):
            val = board[row][col]
            if val == 0: continue
            goal_row,goal_col = divmod(val-1,k)#flattens the board and divides it by size
            if goal_row == row:
                if goal_col < max_col:
                    conflict +=1
                else:
                    max_col = goal_col
    
    for col in range(k):
        max_row = -1
        for row in range(k):
            val = board[row][col]
            if val == 0 : continue 
            goal_row, goal_col = divmod(val-1,k)
            if goal_col == col:
                if goal_row <max_row:
                    conflict+= 1
                else: max_row = goal_row
    
    return man +  2 * conflict
