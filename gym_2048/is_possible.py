
from functools import lru_cache


@lru_cache(maxsize=4096)
def is_action_possible_cache(board, action):
  assert type(action) == int, f'Action must be an integer, not {type(action)}, we want to be hashable, not tensor'

  height = 4
  width = 4

  if action == 0:
    for i in range(height):
      row = board[i]
      #check if this row can be slided, if it can, return True
      zero_found=False
      for j in range(width):
        if row[j] == 0:
          zero_found=True
        elif zero_found:
          return True # There was a non zero after a zero so the row can be slided
        
        if j < width - 1 and row[j] == row[j+1] and row[j] != 0:
          return True # 2 non-zero blocks can be merged
  if action == 1:
    # slide up
    for i in range(width):
      #check columns
      #col = board[:,i]
      zero_found=False
      for j in range(height):
        if board[j][i] == 0:
          zero_found=True
        elif zero_found:
          return True # There was a non zero after a zero so the col can be slided up
        
        if j < width - 1 and board[j][i] == board[j+1][i] and board[j][i] != 0:
          return True # 2 non-zero blocks can be merged
  if action == 2:
    # slide right
    for i in range(height):
      row = board[i]
      #check if this row can be slided, if it can, return True
      zero_found=False
      for j in range(width):
        j = width - 1 - j
        if row[j] == 0:
          zero_found=True
        elif zero_found:
          return True # There was a non zero after a zero so the row can be slided
        
        if j < width - 1 and row[j] == row[j+1] and row[j] != 0:
          return True # 2 non-zero blocks can be merged
  if action == 3:
    # slide down
    for i in range(width):
      #check columns
      #col = board[:,i]
      zero_found=False
      for j in range(height):
        j = height - 1 - j
        if board[j][i] == 0:
          zero_found=True
        elif zero_found:
          return True # There was a non zero after a zero so the col can be slided up
        
        if j < width - 1 and board[j][i] == board[j+1][i] and board[j][i] != 0:
          return True # 2 non-zero blocks can be merged
    
  return False
