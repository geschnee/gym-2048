import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding

import random

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

  def __init__(self, width=4, height=4, reward_scheme=REWARD_SCHEME_CLASSIC, only_2s=False, punish_illegal_move=True):
    self.width = width
    self.height = height

    assert reward_scheme in self.get_reward_schemes()
    self.reward_scheme = reward_scheme
    self.only_2s = only_2s
    self.punish_illegal_move = punish_illegal_move
    

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

  def rot_old(self, board, k):
    return np.rot90(board, k=k)
  
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

  def step(self, action: int):
    """Perform step, return observation, reward, terminated, false, info."""

    # Align board action with left action
    rotated_obs = self.rot_board_no_numpy(self.board, k = action)
    reward, updated_obs, changed = self._slide_left_and_merge(rotated_obs)

    if changed:
      #update the board only if a change resulted from the action

      self.board = self.rot_board_no_numpy(updated_obs, k = 4 - action)
      # Place one random tile on empty location
      self._place_random_tiles(self.board, count=1)


    terminated = self.is_done()
    # board.copy() is returned because of an error/incompatibility with Salina https://github.com/facebookresearch/salina
    return self.board.copy(), reward, terminated, {"max_block" : np.max(self.board), "end_value": np.sum(self.board), "is_success": np.max(self.board) >= 2048}
    # TODO change the returned tuple to match the new gym step API
    # https://www.gymlibrary.dev/content/api/#stepping
    # it should then return this:
    # return self.board.copy(), reward, terminated, False, {"max_block" : np.max(self.board), "end_value": np.sum(self.board), "is_success": np.max(self.board) >= 2048}
    # stable-baselines3 is not ready for this change yet

  def is_done(self):
    
    if not self.board.all():
      return False
    
    copy_board = self.board.copy()

    for action in [0, 1, 2, 3]:
      rotated_obs = self.rot_board_no_numpy(copy_board, k = action)
      
      merge_possible = self._merge_possible(rotated_obs)
      if merge_possible:
        # something can be merged --> we are not done
        return False

    return True
  
  def _merge_possible(self, board):
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
      return self.board, {"max_block" : np.max(self.board), "end_value": np.sum(self.board)}

    return self.board
  
  def is_action_possible(self, action: int):
    rotated_obs = self.rot_board_no_numpy(self.board, k = action)
    for i in range(self.height):
      row = rotated_obs[i]
      #check if this row can be slided, if it can, return True
      zero_found=False
      for j in range(self.width):
        if row[j] == 0:
          zero_found=True
        elif zero_found:
          return True # There was a non zero after a zero so the row can be slided
        
        if j < self.width - 1 and row[j] == row[j+1] and row[j] != 0:
          return True # 2 non-zero blocks can be merged

    return False

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


  def _sample_tile_locations(self, board, count=1):
    """Sample grid locations with no tile."""

    zero_locs = np.argwhere(board == 0)
    zero_indices = self.np_random.choice(
        len(zero_locs), size=count)

    zero_pos = zero_locs[zero_indices]

    t = np.transpose(zero_pos)
    
    return t

  def _sample_tile_locations_no_numpy(self, board, count=1):
    """Sample grid locations with no tile."""

    zero_locs = []
    for i in range(self.height):
      for j in range(self.width):
        if board[i][j] == 0:
          zero_locs.append([i,j])
    
    zero_indices = []
    while len(zero_indices) < count:
      index = random.randint(0, len(zero_locs) - 1)
      if index not in zero_indices:
        zero_indices.append(index)

    t = [[],[]]
    for index in zero_indices:
      t[0].append(zero_locs[index][0])
      t[1].append(zero_locs[index][1])

    return t

  def _place_random_tiles(self, board, count=1):
    if not board.all():
      tiles = self._sample_tiles_no_numpy(count)
      tile_locs = self._sample_tile_locations_no_numpy(board, count)

      board[(tile_locs[0], tile_locs[1])] = tiles
      #tile_locs[0] is the x indices and tile_locs[1] is the y indices

    else:
      raise Exception("Board is full.")
  
  def pad(self, result_row):
    for _ in range(self.width - len(result_row)):
      result_row.append(0)
    return result_row
  
  def new_extract_nonzero(self, row):
    result = []
    for i in row:
      if i != 0:
        result.append(i)
    return result
  

  def row_unequal(self, row, rowc):
    for i in range(self.width):
      if row[i] != rowc[i]:
        return True
    return False
    
  def _slide_left_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result = []

    merges = []
    changed = False
    for row in board:
      rowc = self.new_extract_nonzero(row)
      merges_, result_row = self._try_merge(rowc)
      merges.extend(merges_)
      nrow = self.pad(result_row)
      result.append(nrow)
      if not changed and (len(merges) > 0 or self.row_unequal(row, nrow)):
        changed = True

    result_board = np.array(result, dtype=np.int64)

    

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

    if self.punish_illegal_move and np.array_equal(board, result_board):
      score = -1
      #moves without any changes

    print("score: ", score)
    print(f'score type: {type(score)}')
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