import cnn_dqn_agent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time as t
import cv2
import os
from collections import deque
import random as rand
import pyautogui
import numpy as np
import torch

dir = 'web_made\data'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

print("dir set")

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

url_path = "https://vidkidz.tistory.com/2825"
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(1)

driver.get(url_path) # url로 이동
driver.implicitly_wait(1) # wait time

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="canvas"]'))
    )

finally:
    pass

for i in range(4):
    t.sleep(1)
    element.send_keys(Keys.SPACE)

print("space")

def screen_shot(current_time):
    img_path = 'web_made\data\\' + str(current_time) + '.png'
    element.screenshot(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (84, 84))
    history.append(img)
    
    return history

agent = cnn_dqn_agent.Agent()
done = False
counter = 0 # Step counter

EPS_START = 1 # 학습 시작 시 무작위 행동할 확률
EPS_END = 0.001 # 학습 종료 시 무작위 행동할 확률

eps_timesteps = (0.1) * \
        float(1000)

history = deque(maxlen=4)

timer = t.time()

print("game ready")
state = screen_shot(timer)
episode_rewards = [0.0]

for st in range(1000):
    fraction = min(1.0, float(st) / eps_timesteps)
    eps_threshold = EPS_START + fraction * \
        (EPS_END - EPS_START)
    sample = rand.random()

    if(sample > eps_threshold):
        # Exploit
        action = agent.act(state)
    else:
        # Explore
        action = rand.randint(0, 1)

    if action == 0:
        pyautogui.keyDown('left')
        t.sleep(0.0001)
        pyautogui.keyUp('left')
    if action == 1:
        pyautogui.keyDown('right')
        t.sleep(0.0001)
        pyautogui.keyUp('right')

    current_time = t.time() - timer
    next_state = screen_shot(current_time)
    reward = current_time
    agent.memory.add(state, action, reward, next_state, float(done))

    if state == next_state:
        done = True

    if done:
        timer = t.time()
        element.send_keys(Keys.SPACE)
        t.sleep(0.1)
        element.send_keys(Keys.SPACE)
        done = False
        episode_rewards.append(0.0)

    state = next_state

    if st > 200: # learning start
        agent.optimise_td_loss()

    if st % 200 == 0: # target network update
        agent.update_target_network

    episode_rewards[-1] += reward
    if done and len(episode_rewards) % 20: # print step result
        mean_hundred_epi = round(np.mean(episode_rewards[-101:-1], 1))
        print("********************************************************")
        print("steps: {}".format(st))
        print("episodes: {}".format(len(episode_rewards)))
        print("mean 100 episode reward: {}".format(mean_hundred_epi))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************")
        torch.save(agent.policy_network.state_dict(), f'checkpoint.pth')
        np.savetxt('rewards_per_episode.csv', episode_rewards,
                    delimiter=',', fmt='%1.3f')

        dir = 'web_made\data'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
            