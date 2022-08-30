import pygame as pg
import sys
import env

def render_game(screen, engine):

    for i in range(4):
        for j in range(4):
            i = 3-i
            j = 3-j
            value = engine.board[j,i]

            color = colours[value]
            
            img = big_font.render(f'{value}', True, BLACK)

            rect_obj = pg.draw.rect(screen, color, (render_x + rect_width*i, render_y +rect_width*j, rect_width, rect_width), 1)
            text_rect = img.get_rect(center=rect_obj.center)
            
            pg.draw.rect(screen, color, rect_obj)
            screen.blit(img, text_rect)

def render_text(msg, y):

    img = big_font.render(msg, True, BLACK)

    screen.blit(img, (text_x, y))


pg.init()
screen = pg.display.set_mode([800, 600])
clock = pg.time.Clock()

big_font = pg.font.SysFont(None, 48)
small_font = pg.font.SysFont(None, 24)

action_mapping = env.Base2048Env.action_names()

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

colours = {0: (238, 228, 218),
           2: (235, 215, 195), 
           4: (224, 193, 162),
           8: (224, 182, 139),
           16: (230, 174, 119),
           32: (222, 160, 98),
           64: (217, 147, 78),
           128: (214, 133, 51),
           256: (209, 120, 31),
           512: (173, 96, 19),
           1024: (138, 74, 11),
           2048: (110, 60, 11),}

#initialize game
rewardsum=0
scoring_metric="classic"
#['encourage_empty', 'classic','merge_counts_encourage_empty']

last_action="New Game"
board_value = 0
max_block = 0

render_x = 20
render_y = 100
rect_width=100

text_x = render_x+4*rect_width + 50


engine = env.Base2048Env(reward_scheme=scoring_metric)

go = True
action = None
newgame= False
reward = ""
board_value = ""
max_block = ""
turn = 0
done = False
while go:

    if pg.key.get_pressed()[pg.K_q]:
        sys.exit()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            #quit using the cross in top right corner
            sys.exit()

        if event.type == pg.KEYUP:
            button = event.key

            if button==pg.K_LEFT:
                action = 0

            if button == pg.K_UP:
                action = 1

            if button == pg.K_RIGHT:
                action = 2

            if button == pg.K_DOWN :
                action = 3

            if button == pg.K_n:
                newgame= True

    if action != None:
        board, reward, done, info = engine.step(action)
        
        last_action = action
        action = None
        board_value = info["end_value"]
        max_block = info["max_block"]

        rewardsum+=reward
        turn +=1

        engine.render()

    if newgame:
        #new game
        engine.reset()
        rewardsum=0
        scoring_metric="classic"
        last_action="New Game"
        newgame= False
        reward = ""
        turn = 0

    #render
    screen.fill((235, 97, 52))

    img = big_font.render('2048 GymEnvironment', True, BLACK)

    screen.blit(img, (10, 10))

    render_game(screen, engine)
    render_text(f'rewardsum: {rewardsum}', 100)
    render_text(f'reward: {reward}', 150)
    render_text(f'b value: {board_value}', 200)
    render_text(f'max block: {max_block}', 250)
    if last_action == "New Game":
        render_text(f'new game', 300)
    else:
        render_text(f'last action: {action_mapping[last_action]}', 300)


    render_text(f'metric: {scoring_metric}', 350)
    render_text(f'turn {turn}', 400)

    if done:
        render_text(f'game over', 450)

    pg.display.update()
    clock.tick(60)

# time it took to program this file:
# 2h 7m