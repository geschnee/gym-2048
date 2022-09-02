import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding


class Base2048Env(gym.Env):
  metadata = {
      'render.modes': ['human'],
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

  def __init__(self, width=4, height=4, reward_scheme="classic", only_2s=False):
    self.width = width
    self.height = height
    self.reward_scheme = reward_scheme
    self.only_2s = only_2s
    

    self.observation_space = spaces.Box(low=2,
                                        high=2**32,
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

  def step(self, action: int):
    """Rotate board aligned with left action"""
    #print(f'action {action} on Env\n{self.board} ')

    
    # Align board action with left action
    rotated_obs = np.rot90(self.board, k=action)
    reward, updated_obs = self._slide_left_and_merge(rotated_obs)

    if not np.array_equal(rotated_obs, updated_obs):
      #update the board only if a change resulted from the action

      self.board = np.rot90(updated_obs, k=4 - action)
      # Place one random tile on empty location
      self._place_random_tiles(self.board, count=1) 
      # since count is always 1 and we did an action, there should never be an issue

    done = self.is_done()



    # board.copy() is returned because of an error/incompatibility with Salina https://github.com/facebookresearch/salina
    return self.board.copy(), reward, done, {"max_block" : np.max(self.board), "end_value": np.sum(self.board)}

  def is_done(self):
    copy_board = self.board.copy()

    if not copy_board.all():
      return False

    for action in [0, 1, 2, 3]:
      rotated_obs = np.rot90(copy_board, k=action)
      _, updated_obs = self._slide_left_and_merge(rotated_obs)
      if not updated_obs.all():
        return False

    return True


  def reset(self):
    """Place 2 tiles on empty board."""

    self.board = np.zeros((self.width, self.height), dtype=np.int64)
    self._place_random_tiles(self.board, count=2)

    return self.board

  def is_action_possible(self, action: int):
    
    rotated_obs = np.rot90(self.board, k=action)
    _, merged_rotated_board = self._slide_left_and_merge(rotated_obs)

    if np.array_equal(rotated_obs, merged_rotated_board):
      return False


    return True

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

  def _sample_tile_locations(self, board, count=1):
    """Sample grid locations with no tile."""

    zero_locs = np.argwhere(board == 0)
    zero_indices = self.np_random.choice(
        len(zero_locs), size=count)

    zero_pos = zero_locs[zero_indices]

    t = np.transpose(zero_pos)
    
    return t

  def _place_random_tiles(self, board, count=1):
    if not board.all():
      tiles = self._sample_tiles(count)
      tile_locs = self._sample_tile_locations(board, count)

      board[(tile_locs[0], tile_locs[1])] = tiles
      #tile_locs[0] is the x indices and tile_locs[1] is the y indices

    else:
      raise Exception("Board is full.")

  def _slide_left_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result = []

    score = 0
    for row in board:
      row = np.extract(row > 0, row)
      score_, result_row = self._try_merge(row)
      score += score_
      row = np.pad(np.array(result_row), (0, self.width - len(result_row)),
                   'constant', constant_values=(0,))
      result.append(row)

    result_board = np.array(result, dtype=np.int64)


    if self.reward_scheme == "encourage_empty" or self.reward_scheme == "merge_counts_encourage_empty":
      score += result_board.size - np.count_nonzero(result_board)

    if np.array_equal(board, result_board):
      score = -1
      #moves without any changes

    return score, result_board

  
  def _try_merge(self, row):
    score = 0
    result_row = []

    i = 1
    while i < len(row):
      if row[i] == row[i - 1]:
        if self.reward_scheme == "merge_counts":
          score += 1
        else:
          score += row[i] + row[i - 1]
        result_row.append(row[i] + row[i - 1])
        i += 2
      else:
        result_row.append(row[i - 1])
        i += 1

    if i == len(row):
      result_row.append(row[i - 1])

    return score, result_row

  @staticmethod
  def action_names():
   return { 0: "LEFT", 1: "UP", 2: "RIGHT", 3:"DOWN"}

  @staticmethod
  def get_reward_schemes():
    return ['encourage_empty', 'classic','merge_counts_encourage_empty']