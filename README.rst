Gym 2048
=========


.. image:: https://badge.fury.io/py/gym-2048-extended.svg
    :target: https://pypi.org/project/gym-2048-extended/

This package implements the classic grid game 2048
for OpenAI gym environment. 

Summarizing my changes to the repo `Activated Geek Gym 2048 <https://github.com/activatedgeek/gym-2048>`_:

- changed the requirements.txt to be more flexible
- added the method ``is_action_possible``
- added different reward schemes
- added the render mode ``dict``
- added ``game.py`` a PyGame that uses the GymEnvironment from this repo

Install
-------

Pip
^^^

.. code:: shell

    pip install gym-2048-extended

From cloned repository
^^^^^^^^^^^^^^^^^^^^^^

`GitHub Page <https://github.com/geschnee/gym-2048>`_

.. code:: shell

    python setup.py install

C:\Users\User\AppData\Local\Programs\Python\Python310\Lib\site-packages

Environment(s)
--------------

The package currently contains two environments

- ``Tiny2048-v0``: A ``2 x 2`` grid game.
- ``2048-v0``: The standard ``4 x 4`` grid game.

I only checked the ``4 x 4`` grid game, the other one might not work.


Attributes
^^^^^^^^^^^

- **Observation**: All observations are ``n x n`` numpy arrays
  representing the grid. The array is ``0`` for empty locations
  and numbered ``2, 4, 8, ...`` wherever the tiles are placed.

- **Actions**: There are four actions defined by integers.
    - ``LEFT = 0``
    - ``UP = 1``
    - ``RIGHT = 2``
    - ``DOWN = 3``

- **Reward**: Reward is the total score obtained by merging any
  potential tiles for a given action. Score obtained by merging
  two tiles is simply the sum of values of those two tiles.

Rendering
^^^^^^^^^^

Currently 2 rendering modes are implemented

- basic print rendering (``mode='human'``)
- dict rendering (``mode='dict'``) returns a dictionary with the board state

Usage
------

PyGame Interactive Demo
^^^^^^^^^^^^^^^^^^^^^^^

``game.py`` provides a PyGame implementation of the game.
Use the arrow keys to play, ``q`` and ``n`` can be used to quit the game or restart it.

The game serves as a demo, the different reward schemes and step function can be explored.

.. code:: shell

    python gym_2048/game.py

.. image:: pygame.png
   :width: 600


Basic Demo
^^^^^^^^^^

Here is a sample rollout of the game which follows the same API as
OpenAI ``gym.Env``.

.. code:: python

    import gym_2048
    import gym


    if __name__ == '__main__':
      env = gym.make('2048-extended-v2')
      env.seed(42)

      env.reset()
      env.render()

      done = False
      moves = 0
      while not done:
        action = env.np_random.choice(range(4), 1).item()
        next_state, reward, done, _, info = env.step(action)
        moves += 1

        print('Next Action: "{}"\n\nReward: {}'.format(
          gym_2048.Base2048Env.ACTION_STRING[action], reward))
        env.render()

      print('\nTotal Moves: {}'.format(moves))


**NOTE**: Top level ``import gym_2048`` is needed to ensure registration with
``Gym``.
