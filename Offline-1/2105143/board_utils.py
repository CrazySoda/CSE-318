def count_inversion(flat_board):
    inv = 0
    for i in range(len(flat_board)):
        for j in range(i+1, len(flat_board)):
            if flat_board[i] and flat_board[j] and flat_board[i]> flat_board[j]:
                inv +=1
                
    return inv

def find_blank_row_from_bottom(board):
    k = len(board) #number of rows
    for i in range(k-1,-1,-1):
        for j in range(k):
            if board[i][j] == 0:
                return k-i
            
def is_solvable(board):
    flat = [num for row in board for num in row]
    k = int (len(flat)**0.5)#as square board
    inv = count_inversion(flat)
    if k%2 == 1:
        return inv %2 == 0
    else:
        row_from_bottom = find_blank_row_from_bottom(board)
        return (inv + row_from_bottom) % 2 == 0
    
def get_goal_board(size):
    nums = list(range(1,size*size)) + [0]
    return [nums[i*size : (i+1)*size] for i in range(size)]
               