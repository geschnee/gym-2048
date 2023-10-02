
from gym_2048.env import Base2048Env
import env


from gym_2048.is_possible import is_action_possible_cache

def test_cache_hit():
    print('\ntest_cache_hit')
    e = env.Base2048Env()
    p = e.is_action_possible(0)
    hits = is_action_possible_cache.cache_info().hits
    assert hits == 0
    assert is_action_possible_cache.cache_info().misses == 1
    print(f'first check')
    print(f'cache info {is_action_possible_cache.cache_info()}')

    
    p2 = e.is_action_possible(0)
    hits_after = is_action_possible_cache.cache_info().hits
    assert hits_after == 1

    print(f'hit {hits} and hit after {hits_after}')

    print(f'cache info {is_action_possible_cache.cache_info()}')

    e2 = Base2048Env()
    e2.board = e.board
    p2 = e2.is_action_possible(0)
    print(f'cache info {is_action_possible_cache.cache_info()}')
    assert is_action_possible_cache.cache_info().hits == 2


    assert hits_after == hits + 1
    assert p == p2

test_cache_hit()









