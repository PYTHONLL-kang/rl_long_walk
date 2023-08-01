import pygame as pg
import math
import dqn_agent as dqn_agent
import time

pg.init()

size = [800, 450]
screen = pg.display.set_mode(size)

done = False
clock = pg.time.Clock()
game_font = pg.font.Font(None, 40)

head_image = pg.image.load("./self_made\\assets\\head.png")

background_image = pg.image.load("./self_made\\assets\\background.png")

l_leg = pg.image.load("./self_made\\assets\\l_leg.png")
l_leg = pg.transform.scale(l_leg, (800, 450))
r_leg = pg.image.load("./self_made\\assets\\r_leg.png")
r_leg = pg.transform.scale(r_leg, (800, 450))

backgroun_image = pg.transform.scale(background_image, (800, 450))

white = (255, 255, 255)
R = 150
head_pos = [400, 325]


class Head:
    def __init__(self):
        self.dist = 0
        self.angle = -(math.pi / 2)
        self.body_image = pg.image.load("./self_made\\assets\\body_.png")
        self.gravity = 0.01
        self.X = head_pos[0] + math.cos(self.angle) * R
        self.Y = head_pos[1] + math.sin(self.angle) * R

    def act_move(self, dir, elaps_time):
        if dir < 0:
            acc = - (elaps_time / 100)
        else:
            acc = elaps_time / 100
        self.angle = self.angle + dir + acc

    def pas_move(self, elaps_time):
        self.gravity = self.gravity + (elaps_time / 100000)
        if round(self.angle, 2) < -1.6:
            self.angle = self.angle - self.gravity
        else:
            self.angle = self.angle + self.gravity

        if round(self.angle, 2) <= -3.2:
            self.angle = -(math.pi / 2)
            return True

        if round(self.angle, 2) >= 0:
            self.angle = -(math.pi / 2)
            return True

    def draw_head(self):
        self.X = head_pos[0] + math.cos(self.angle) * R
        self.Y = head_pos[1] + math.sin(self.angle) * R

        screen.blit(head_image, [self.X, self.Y])

    def draw_body(self):
        ang =  math.degrees(self.angle)
        angle = -ang - 60

        if angle < 0:
            angle = angle - 15
        
        # print("ang: {0:.2f}, angle = {1:.2f}, self.ang: {2:.2f}".format(ang, angle, self.angle))
        rotate_body = pg.transform.rotate(self.body_image, angle)

        new_rect = rotate_body.get_rect(
            center=self.body_image.get_rect(center=(450, 350)).center
        )

        screen.blit(rotate_body, new_rect)

    def return_pos(self):
        return self.X, self.Y


head = Head()
agent = dqn_agent.Agent()
start_ticks = pg.time.get_ticks()
elapsed_time = 0
counter = 0
c = 0

while not done:
    playing = True

    clock.tick(20)
    screen.fill(white)
    screen.blit(background_image, (0, 0))

    cur_x, cur_y = head.return_pos()
    state = [cur_y, elapsed_time]

    action = agent.act(state, counter)

    key_event = pg.key.get_pressed()

    if key_event[pg.K_LEFT] or action[0] == 0:
        head.act_move(-0.1, elapsed_time)

    if key_event[pg.K_RIGHT] or action[0] == 1:
        head.act_move(0.1, elapsed_time)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            agent.save_net()
            done = True

    end_ticks = pg.time.get_ticks()
    elapsed_time = (end_ticks - start_ticks) / 1000
    timer = game_font.render(str(int(elapsed_time)), True, (0, 0, 0))

    next_x, next_y = head.return_pos()
    next_state = [next_y, elapsed_time]

    if c < 3:
        screen.blit(l_leg, (50, 50))
        c = c + 1
    elif c < 6:
        screen.blit(r_leg, (50, 50))
        c = c + 1
    else:
        c = 0

    stop = head.pas_move(elapsed_time)
    head.draw_body()
    head.draw_head()

    if stop:
        reward = -10
        playing = False
    else:
        reward = size[1] - next_y

    agent.memorize(state, action, reward, next_state)
    agent.learn()

    screen.blit(timer, (10, 10))
    pg.display.flip()

    if playing == False:
        head = Head()
        start_ticks = pg.time.get_ticks()
        print("==========")
        print("==========")
        print("==========")
        print("game over, timer: {0:.2f}".format(elapsed_time))
        print("==========")
        print("==========")
        print("==========")
        reward = 0
        counter = counter + 1

pg.quit()
