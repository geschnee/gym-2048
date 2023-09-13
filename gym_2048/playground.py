
from gym_2048.env import Base2048Env#, is_action_possible_cache
import env


from gym_2048.is_possible import is_action_possible_cache


# TODO debug
def test_cache_hit():
    print('\ntest_cache_hit')
    e = env.Base2048Env()
    p = e.is_action_possible(0)
    hits = is_action_possible_cache.cache_info().hits
    print(f'first check')

    print(f'cache info {is_action_possible_cache.cache_info()}')
    p2 = e.is_action_possible(0)
    hits_after = is_action_possible_cache.cache_info().hits

    print(f'hit {hits} and hit after {hits_after}')

    print(f'cache info {is_action_possible_cache.cache_info()}')

    print(f'{is_action_possible_cache}')


    e2 = Base2048Env()
    e2.board = e.board
    p2 = e2.is_action_possible(0)
    print(f'cache info {is_action_possible_cache.cache_info()}')


    assert hits_after == hits + 1, f'Cache hit failed'
    assert p == p2, f'Cache hit failed'


#import cProfile
#cProfile.run("test_cache_hit()", sort="tottime")
test_cache_hit()









