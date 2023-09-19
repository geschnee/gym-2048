import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding

import random

from gym_2048.is_possible import is_action_possible_cache


class Base2048Env(gym.Env):
  metadata = {
      'render_modes': ['human', 'dict'],
  }

  ##
  # NOTE: Don't modify these numbers as
  # they define the number of
  # anti-clockwise rotations before
  # applying the left action on a grid
  #
  LEFT = 0
  UP = 1
  RIGHT = 2
  DOWN = 3

  ACTION_STRING = {
      LEFT: 'left',
      UP: 'up',
      RIGHT: 'right',
      DOWN: 'down',
  }

  REWARD_SCHEME_CLASSIC = "classic"
  REWARD_SCHEME_MAX_MERGE = "max_merge"
  REWARD_SCHEME_TRIPLET = "triplet"
  REWARD_SCHEME_ENCOURAGE_EMPTY = "encourage_empty"
  REWARD_SCHEME_MERGE_COUNTS_ENCOURAGE_EMPTY = "merge_counts_encourage_empty"
  REWARD_SCHEME_MERGE_COUNTS = "merge_counts"

  @staticmethod
  def action_names():
   return { 0: "LEFT", 1: "UP", 2: "RIGHT", 3:"DOWN"}

  @staticmethod
  def get_reward_schemes():
    return [Base2048Env.REWARD_SCHEME_CLASSIC, Base2048Env.REWARD_SCHEME_ENCOURAGE_EMPTY,Base2048Env.REWARD_SCHEME_MERGE_COUNTS_ENCOURAGE_EMPTY, Base2048Env.REWARD_SCHEME_MERGE_COUNTS, Base2048Env.REWARD_SCHEME_TRIPLET, Base2048Env.REWARD_SCHEME_MAX_MERGE]

  def __init__(self, width=4, height=4, reward_scheme=REWARD_SCHEME_CLASSIC, only_2s=False, punish_illegal_move=True, full_info=False):
    self.width = width
    self.height = height

    assert reward_scheme in self.get_reward_schemes()
    self.reward_scheme = reward_scheme
    self.only_2s = only_2s
    self.punish_illegal_move = punish_illegal_move
    self.full_info = full_info
    

    self.observation_space = spaces.Box(low=0,
                                        high=2**14,
                                        shape=(self.width, self.height),
                                        dtype=np.int64)
    self.action_space = spaces.Discrete(4)

    # Internal Variables
    self.board = None
    self.np_random = None

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def rot_board_no_numpy(self, board, k):
    k=k % 4
    if k == 0:
      return board
    if k == 1:
      newboard = np.zeros((self.width, self.height), dtype=np.int64)
      for i in range(self.height):
        for j in range(self.width):
          newboard[i][j] = board[j][self.width - i -1]
      return newboard
    if k == 2:
      newboard = np.zeros((self.width, self.height), dtype=np.int64)
      for i in range(self.height):
        for j in range(self.width):
          newboard[i][j] = board[self.height - i - 1][self.width - j - 1] 
      return newboard
    if k == 3:
      newboard = np.zeros((self.width, self.height), dtype=np.int64)
      for i in range(self.height):
        for j in range(self.width):
          newboard[i][j] = board[4 - j - 1][i]
      return newboard

  def old_move(self, action: int):
    raise NotImplementedError("This method is not used anymore, it is only kept for reference")

    # Align board action with left action
    rotated_obs = self.rot_board_no_numpy(self.board, k = action)
    reward, updated_obs, changed = self._slide_l_and_merge(rotated_obs)
    board = self.board
    if changed:
      #update the board only if a change resulted from the action
      board = self.rot_board_no_numpy(updated_obs, k = 4 - action)
    return reward, board, changed
  
  def new_move(self, action: int):
    # Align board action with left action
    if action == 0:
      reward, updated_obs, changed = self._slide_left_and_merge(self.board)
    elif action == 1:
      reward, updated_obs, changed = self._slide_up_and_merge(self.board)
    elif action == 2:
      reward, updated_obs, changed = self._slide_right_and_merge(self.board)
    elif action == 3:
      reward, updated_obs, changed = self._slide_down_and_merge(self.board)
    else: 
      raise ValueError("action should be 0, 1, 2 or 3")

    return reward, updated_obs, changed

  def step(self, action: int):
    """Perform step, return observation, reward, terminated, false, info."""
    
    reward, board, changed = self.new_move(action)

    self.board = board

    definitely_not_terminated = False
    if changed:
      #update the board only if a change resulted from the action

      definitely_not_terminated = self._place_random_single_tile(self.board)

    if definitely_not_terminated:
      terminated = False
    else:
      terminated = self.is_done()

    # board.copy() is returned because of an error/incompatibility with Salina https://github.com/facebookresearch/salina
    # since we do not use salina anymore, we try without board copy

    if terminated or self.full_info:
      mx = self.board_max(self.board)
      info_dict = {"max_block" : mx, "end_value": self.board_sum(self.board), "is_success": mx >= 2048}
    else:
      # TODO is it okay like this for sb3?
      info_dict = {}


    return self.board, reward, terminated, info_dict
    # TODO change the returned tuple to match the new gym step API
    # https://www.gymlibrary.dev/content/api/#stepping
    # it should then return this:
    # return self.board.copy(), reward, terminated, False, {"max_block" : np.max(self.board), "end_value": np.sum(self.board), "is_success": np.max(self.board) >= 2048}
    # stable-baselines3 is not ready for this change yet

  def board_max(self, board):
    # quicker than np.max(board)
    mx = 0
    for i in range(self.height):
      for j in range(self.width):
        if board[i][j] > mx:
          mx = board[i][j]
    return mx
  

  def board_sum(self, board):
    # quicker than np.sum(board)
    sum = 0
    for i in range(self.height):
      for j in range(self.width):
        sum += board[i][j]
    return sum

  def is_done(self):
    board_hashable = tuple(map(tuple, self.board))
    for action in [0, 1, 2, 3]:
      possible = is_action_possible_cache(board_hashable, int(action))
      if possible:
        return False

    return True
  
  def _merge_possible(self, board):
    raise NotImplementedError("This method is not used anymore, it is only kept for reference")

    # this function just checks if two adjacent tiles have the same value (and thus can be merged)
    # this method is only intended for the check of is_done when the entire board is full

    # this method is much faster than the _slide_left_and_merge method, which was used to check this before

    #   1143205    1.438    0.000   27.043    0.000 env.py:139(is_done)
    #   374688    3.988    0.000   18.043    0.000 env.py:176(_slide_left_and_merge)
    #   374688    2.163    0.000    2.292    0.000 env.py:159(_merge_possible)

    changed = False
    for row in board:
      
      assert len(row) == self.width
      for i in range(self.width - 1):
        if row[i] == 0:
          raise ValueError("row should not contain 0, we already checked if there is a zero anywhere in the board and there is none, see is_done")

        if row[i] == row[i + 1]:
          changed = True
          return changed
        
    return changed


  def reset(self, seed=None, **kwargs):
    """Place 2 tiles on empty board."""
    #super().reset(seed=seed)

    self.board = np.zeros((self.width, self.height), dtype=np.int64)
    self._place_random_tiles(self.board, count=2)

    if 'return_info' in kwargs and kwargs['return_info']:
      # return_info parameter is included and true
      return self.board, {"max_block" : self.board_max(self.board), "end_value": self.board_sum(self.board)}

    return self.board
  
  def is_action_possible(self, action: int):
    # we do not need the rotation, profiling shows a large proportion of time is spent in rot_board_no_numpy

    board_hashable = tuple(map(tuple, self.board))

    # how to view cache status:
    # print(is_action_possible_cache.cache_info())

    return is_action_possible_cache(board_hashable, int(action))

  def possible_actions(self):
    possible_actions = []

    for i in range(4):
      if self.is_action_possible(i):
        possible_actions.append(i)
    
    return possible_actions

  def return_board(self):
    return self.board.copy()

  def render(self, mode='human'):
    if mode == 'human':
      for row in self.board.tolist():
        print(' \t'.join(map(str, row)))
    if mode == 'dict':
      board = self.board
      dictionary = dict()
      dictionary[
          "line0"
      ] = f"{board[0,0]} {board[0,1]} {board[0,2]} {board[0,3]}"
      dictionary[
          "line1"
      ] = f"{board[1,0]} {board[1,1]} {board[1,2]} {board[1,3]}"
      dictionary[
          "line2"
      ] = f"{board[2,0]} {board[2,1]} {board[2,2]} {board[2,3]}"
      dictionary[
          "line3"
      ] = f"{board[3,0]} {board[3,1]} {board[3,2]} {board[3,3]}"
      return dictionary

  def set_board(self, board):
    self.board = board

  def _sample_tiles(self, count=1):
    """Sample tile 2 or 4."""

    if self.only_2s:
      return [2] * count

    choices = [2, 4]
    probs = [0.9, 0.1]

    tiles = self.np_random.choice(choices,
                                  size=count,
                                  p=probs)
    return tiles.tolist()
  
  def _sample_tiles_no_numpy(self, count = 1):
    
    if self.only_2s:
      return [2] * count
    if count == 1:
      if random.random() < 0.9:
        return [2]
      else:
        return [4]
    tiles = []
    for i in range(count):
      if random.random() < 0.9:
        tiles.append(2)
      else:
        tiles.append(4)
    return tiles

  
  def _place_random_tiles(self, board, count=1):
    

    # get eligible locations
    zero_locs = []
    for i in range(self.height):
      for j in range(self.width):
        if board[i][j] == 0:
          zero_locs.append([i,j])
    assert len(zero_locs) > 0, "Board is full."

    # sample locations
    zero_indices = []
    while len(zero_indices) < count:
      index = random.randint(0, len(zero_locs) - 1)
      if index not in zero_indices:
        zero_indices.append(index)

    # place tiles
    for index in zero_indices:
      if self.only_2s:
        tile = 2
      else:
        if random.random() < 0.9:
          tile = 2
        else:
          tile = 4
      self.board[zero_locs[index][0]][zero_locs[index][1]] = tile

  def _place_random_single_tile(self, board):
    # quicker than the adaptable version with the lists ...

    # get eligible locations
    zero_locs = []
    for i in range(self.height):
      for j in range(self.width):
        if board[i][j] == 0:
          zero_locs.append([i,j])
    assert len(zero_locs) > 0, "Board is full."

    # sample locations
    
    index = random.randint(0, len(zero_locs) - 1)
   
    if self.only_2s:
      tile = 2
    else:
      if random.random() < 0.9:
        tile = 2
      else:
        tile = 4
    self.board[zero_locs[index][0]][zero_locs[index][1]] = tile 

    # definitely not terminated if 2 or more empty tiles before placing
    return len(zero_locs) >= 2
  
  def pad(self, result_row):
    for _ in range(self.width - len(result_row)):
      result_row.append(0)
    return result_row

  
  def row_unequal(self, row, rowc):
    for i in range(self.width):
      if row[i] != rowc[i]:
        return True
    return False
  
  def get_reward(self, merges, result_board, changed):
    if self.reward_scheme == self.REWARD_SCHEME_MAX_MERGE:
      score = max(merges) if len(merges)>0 else 0
    elif self.reward_scheme == self.REWARD_SCHEME_MERGE_COUNTS or self.reward_scheme == self.REWARD_SCHEME_MERGE_COUNTS_ENCOURAGE_EMPTY:
      score = len(merges)
    elif self.reward_scheme == self.REWARD_SCHEME_TRIPLET:
      score = sum(merges)
      if score > 0:
        score = 1
      if score < 0:
        score = -1
    else:
      #default score
      score = sum(merges)

    if self.reward_scheme == self.REWARD_SCHEME_ENCOURAGE_EMPTY or self.reward_scheme == self.REWARD_SCHEME_MERGE_COUNTS_ENCOURAGE_EMPTY:
      score += result_board.size - np.count_nonzero(result_board)

    if self.punish_illegal_move and changed == False:
      score = -1
      #moves without any changes
    return score

  def _slide_left_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result_board = np.zeros((self.width, self.height), dtype=np.int64)

    merges = []
    changed = False
    for i, row in enumerate(board):
      # zeile

      pos_traverse , pos_new = 0, 0

      while pos_traverse < self.width:
        offset = 1
        while pos_traverse < self.width and row[pos_traverse] == 0:
          pos_traverse += 1
        while pos_traverse + offset < self.width and row[pos_traverse + offset] == 0:
          offset += 1

        if pos_traverse + offset < self.width and row[pos_traverse] == row[pos_traverse + offset] and row[pos_traverse] != 0:
          # merge
          result_board[i][pos_new] = row[pos_traverse] * 2
          merges.append(row[pos_traverse] * 2)
          pos_traverse += offset + 1
          changed = True
        elif pos_traverse < self.width and row[pos_traverse] != 0:
          result_board[i][pos_new] = row[pos_traverse]
          if pos_traverse != pos_new:
            changed = True
          pos_traverse += offset
        pos_new += 1


    score = self.get_reward(merges, result_board, changed)

    return score, result_board, changed
  
  def _slide_up_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result_board = np.zeros((self.width, self.height), dtype=np.int64)

    merges = []
    changed = False
    for i in range(self.width):
      # column
      column = board[:,i]

      pos_traverse , pos_new = 0, 0

      while pos_traverse < self.height:
        offset = 1
        while pos_traverse < self.height and column[pos_traverse] == 0:
          pos_traverse += 1
        while pos_traverse + offset < self.height and column[pos_traverse + offset] == 0:
          offset += 1

        if pos_traverse + offset < self.height and column[pos_traverse] == column[pos_traverse + offset] and column[pos_traverse] != 0:
          # merge
          result_board[pos_new][i] = column[pos_traverse] * 2
          merges.append(column[pos_traverse] * 2)
          pos_traverse += offset + 1
          changed = True
        elif pos_traverse < self.height and column[pos_traverse] != 0:
          result_board[pos_new][i] = column[pos_traverse]
          if pos_traverse != pos_new:
            changed = True
          pos_traverse += offset
        pos_new += 1


    score = self.get_reward(merges, result_board, changed)

    return score, result_board, changed
  
  def _slide_right_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result_board = np.zeros((self.width, self.height), dtype=np.int64)

    merges = []
    changed = False
    for i, row in enumerate(board):
      # zeile

      pos_traverse , pos_new = self.width -1, self.width -1

      while pos_traverse >= 0:
        offset = -1
        while pos_traverse >= 0 and row[pos_traverse] == 0:
          pos_traverse -= 1
        while pos_traverse + offset >= 0 and row[pos_traverse + offset] == 0:
          offset -= 1

        if pos_traverse + offset >= 0 and row[pos_traverse] == row[pos_traverse + offset] and row[pos_traverse] != 0:
          # merge
          result_board[i][pos_new] = row[pos_traverse] * 2
          merges.append(row[pos_traverse] * 2)
          pos_traverse += offset - 1
          changed = True
        elif pos_traverse >= 0 and row[pos_traverse] != 0:
          result_board[i][pos_new] = row[pos_traverse]
          if pos_traverse != pos_new:
            changed = True
          pos_traverse += offset
          
        pos_new -= 1


    score = self.get_reward(merges, result_board, changed)

    return score, result_board, changed

  def _slide_down_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result_board = np.zeros((self.width, self.height), dtype=np.int64)

    merges = []
    changed = False
    for i in range(self.width):
      # spalte
      column = board[:,i]

      pos_traverse , pos_new = self.height -1, self.height -1

      while pos_traverse >= 0:
        offset = -1
        while pos_traverse >= 0 and column[pos_traverse] == 0:
          pos_traverse -= 1
        while pos_traverse + offset >= 0 and column[pos_traverse + offset] == 0:
          offset -= 1

        if pos_traverse + offset >= 0 and column[pos_traverse] == column[pos_traverse + offset] and column[pos_traverse] != 0:
          # merge
          result_board[pos_new][i] = column[pos_traverse] * 2
          merges.append(column[pos_traverse] * 2)
          pos_traverse += offset - 1
          changed = True
        elif pos_traverse >= 0 and column[pos_traverse] != 0:
          result_board[pos_new][i] = column[pos_traverse]
          if pos_traverse != pos_new:
            changed = True
          pos_traverse += offset
        pos_new -= 1


    score = self.get_reward(merges, result_board, changed)

    return score, result_board, changed

  def _try_merge(self, row):
    merges = []
    result_row = []

    i = 1
    while i < len(row):
      if row[i] == row[i - 1]:
        merges.append(row[i] + row[i - 1])
        result_row.append(row[i] + row[i - 1])
        i += 2
      else:
        result_row.append(row[i - 1])
        i += 1

    if i == len(row):
      result_row.append(row[i - 1])

    return merges, result_row
  

  def flip_board_no_numpy(self, board):
    # flip board horizontally
    new_board = []
    for i in range(self.height):
      new_board.append(board[self.height - 1 - i])
      
    return np.array(new_board, dtype=np.int64)




  def permutate_board(self, board, action_values):
    # rotate board in all 4 directions (3 directions since one is already given)
    # also flip board across x and y axis for more data
    # in total 8 mutations of the board are returned

    rotation_action_mappings = {
      0: [0, 1, 2, 3],
      1: [1, 2, 3, 0],
      2: [2, 3, 0, 1],
      3: [3, 0, 1, 2],
    }

    flipped_rotation_action_mappings = {
      4: [0, 3, 2, 1],
      5: [3, 2, 1, 0],
      6: [2, 1, 0, 3],
      7: [1, 0, 3, 2],
    }

    # return all the data and the mapping of old actions to new actions
    mutations = [] # first mutation is the original board
    for i in range(4):
      rot_b = self.rot_board_no_numpy(board, i)
      new_action_values = [action_values[j] for j in rotation_action_mappings[i]]
      mutations.append((rot_b, new_action_values))

    flipped_board = self.flip_board_no_numpy(board)
    for i in range(4):
      rot_b = self.rot_board_no_numpy(flipped_board, i)
      new_action_values = [action_values[j] for j in flipped_rotation_action_mappings[i + 4]]
      mutations.append((rot_b, new_action_values))

    return mutations

  def permutate_board_single_action(self, board, action):
    # rotate board in all 4 directions (3 directions since one is already given)
    # also flip board across x and y axis for more data

    rotation_action_mappings = {
      0: [0, 1, 2, 3],
      1: [3, 0, 1, 2],
      2: [2, 3, 0, 1],
      3: [1, 2, 3, 0],
    }

    flipped_rotation_action_mappings = {
      4: [0, 3, 2, 1],
      5: [3, 2, 1, 0],
      6: [2, 1, 0, 3],
      7: [1, 0, 3, 2],
    }

    # return all the data and the mapping of old actions to new actions
    mutations = [] # first mutation is the original board
    for i in range(4):
      rot_b = self.rot_board_no_numpy(board, i)
      new_action = rotation_action_mappings[i][action]
      mutations.append((rot_b, new_action))

    flipped_board = self.flip_board_no_numpy(board)
    for i in range(4):
      rot_b = self.rot_board_no_numpy(flipped_board, i)
      new_action = flipped_rotation_action_mappings[i + 4][action]
      mutations.append((rot_b, new_action))

    return mutations