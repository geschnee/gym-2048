

import env
import numpy as np

import pytest

# python -m pytest test.py

# with Output:
# python -m pytest test.py -s

def test_is_done():
    print('\ntest_is_done')
    e = env.Base2048Env()

    e.render()

    print(f'e.board {e.board}')

    done_board = get_impossible_board()
    print(f'done_board.all() {done_board.all()}')


    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r

        e.board = done_board

        e.render()

        print(f'is_done? {e.is_done()}')
        assert e.is_done(), f'Failed {r}'
        assert np.array_equal(e.board, get_impossible_board()), f'Failed {r}'



def test_is_any_action_possible_done_board():
    print('\ntest_is_any_action_possible_done_board')
    e = env.Base2048Env()
    e.board = get_impossible_board()

    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r

        for i in range(4):
            print(f'Action {i} possible {e.is_action_possible(i)}')
            assert not e.is_action_possible(i), f'Failed {r}'
        
        assert np.array_equal(e.board, get_impossible_board()), f'Failed {r}'


def test_everything_impossible_except_left():
    print(f'\ntest_everything_impossible_except_left')
    e = env.Base2048Env()

    boards = [np.array([[0, 4, 8, 16], [0, 8, 4, 32], [0, 4, 8, 16], [0, 8, 4, 2]]), 
        np.array([[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16]])]

    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r
        for b in boards:
            e.board = b
            e.render()

            assert not e.is_done()

            for i in [1,2,3]:
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')
                assert not e.is_action_possible(i), f'Failed {r}'

            assert e.is_action_possible(0), f'Failed {r}'
            print(f'Action {env.Base2048Env.action_names()[0]} possible {e.is_action_possible(0)}')
            
            assert np.array_equal(e.board, b), f'Failed {r}'



def test_everything_possible_except_left():
    print('\ntest_everything_possible_except_left')
    e = env.Base2048Env()

    boards = [np.array([[8, 32, 16, 2], [2, 16, 4, 2], [16, 2, 0, 0], [8, 4, 2, 0]])]
    
    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r


        for b in boards:
            e.board = np.copy(b)
            e.render()

            assert not e.is_done()

            for i in [1,2,3]:
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')
                assert e.is_action_possible(i), f'Failed {r}'

            assert not e.is_action_possible(0), f'Failed {r}'
            print(f'Action {env.Base2048Env.action_names()[0]} possible {e.is_action_possible(0)}')

            assert np.array_equal(e.board, b), f'Failed {r}'

def test_everything_left_right_possible():
    print('\ntest_everything_possible_except_left')
    e = env.Base2048Env()

    boards = [np.array([[0, 2, 4, 0], [0, 4, 2, 0], [0, 2, 4, 0], [0, 4, 2, 0]])]
    
    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r


        for b in boards:
            e.board = np.copy(b)
            e.render()

            assert not e.is_done()

            for i in [0,2]:
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')
                assert e.is_action_possible(i), f'Failed {r}'

            for i in [1,3]:
                assert not e.is_action_possible(i), f'Failed {r}'
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')

            assert np.array_equal(e.board, b), f'Failed {r}'

def test_everything_left_up_possible():
    print('\ntest_everything_possible_except_left')
    e = env.Base2048Env()

    boards = [np.array([[0, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 8], [8, 4, 2, 16]])]
    
    for r in reward_schemes():
        print(f'testing Reward scheme {r}')
        e.reward_scheme = r


        for b in boards:
            e.board = np.copy(b)
            e.render()

            assert not e.is_done()

            for i in [0,1]:
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')
                assert e.is_action_possible(i), f'Failed {r}'

            for i in [2,3]:
                assert not e.is_action_possible(i), f'Failed {r}'
                print(f'Action {env.Base2048Env.action_names()[i]} possible {e.is_action_possible(i)}')

            assert np.array_equal(e.board, b), f'Failed {r}'

@pytest.mark.skip(reason="show how to skip test")
def test_fail():
    print('\ntest_fail')
    assert False


def test_when_impossible_action_then_raise_error():
    print(f'\ntest_when_impossible_action_then_raise_error')

    e = env.Base2048Env()
    boards = [np.array([[4, 2, 4, 0], [2, 4, 2, 0], [4, 2, 4, 0], [2, 4, 2, 0]])]
    #the action left should not cause any changes

    for b in boards:
        e.board = np.copy(b)
        assert not e.is_action_possible(0)
        
        e.render()
        _, reward, _, _ = e.step(0)
        e.render()
        assert np.array_equal(e.board, b), f'There was an unexpected change'
        assert reward == -1, f'Rewards for illegal actions should be -1'


def test_possible_action_change():
    e = env.Base2048Env()

    boards = [np.array([[4, 2, 0, 2], [2, 4, 2, 0], [4, 2, 4, 0], [2, 4, 2, 0]])]
    #the action left should cause changes

    for b in boards:
        e.board = np.copy(b)
        assert e.is_action_possible(0)

        e.step(0)
        assert not np.array_equal(e.board, b), f'There was no change'


#helper functions

def get_impossible_board():
    return np.array([[2, 4, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 2]])

def reward_schemes():
    return env.Base2048Env.get_reward_schemes()



def test_numpy_rot90():
    # this test checks if np.rot90(board, k=i) modifies the board
    # it should not change the board

    compare_board = np.array([[2, 4, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 0]])
    for i in range(4):
        board = np.array([[2, 4, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 0]])
        rotated = np.rot90(board, k=i)
        assert np.array_equal(board, compare_board)
        if i != 0:
            assert not np.array_equal(rotated, compare_board)

def test_is_action_possible_no_change():
    # this test checks if the is_action_possible function changes the board
    e = env.Base2048Env()

    boards = [np.array([[4, 2, 0, 2], [2, 4, 2, 0], [4, 2, 4, 0], [2, 4, 2, 0]])]

    for b in boards:
        e.board = np.copy(b)
        assert e.is_action_possible(0)
        assert np.array_equal(e.board, b), f'There was an unexpected change'


