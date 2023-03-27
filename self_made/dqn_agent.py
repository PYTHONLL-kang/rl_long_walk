import tensorflow as tf
import torch
import numpy as np
import math
from collections import deque
from random import *
import random as rand

class Agent:
    def __init__(self):
        self.model = torch.nn.Sequential(
                                        torch.nn.Linear(2, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256, 128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(128, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, 3)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)

        self.EPS_START = 0.9 #학습 시작 시 무작위 행동할 확률
        self.EPS_END = 0.001 #학습 종료 시 무작위 행동할 확률
        self.EPS_DECAY = 200 #학습이 반복되며 무작위로 행동할 확률 감소 값

        self.GAMMA = 0.8 #감마는 할인계수, 에이전트가 현재를 미래보다 더 가치있게 여기는 것

        self.batch_size = 64

    def memorize(self, state, action, reward, next_state):
        state = torch.FloatTensor([state])
        self.memory.append((
                            state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])
        ))
        print("reward: {0:.2f}".format(reward))

    def act(self, state, counter):
        state = torch.FloatTensor(state)
        eps_threshold = self.EPS_END + ((self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY))
        self.steps_done = counter

        if rand.random() > eps_threshold: #최대 보상
            print("eps threshold: {0:.2f}, ==greed==".format(eps_threshold))
            return self.model(state).data.max(0)[1].view(1, 1)
        
        else: #무작위
            print("eps threshold: {0:.2f}, ==random==".format(eps_threshold))
            return torch.LongTensor([[rand.randrange(3)]])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = rand.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states) # 64, 2
        actions = torch.cat(actions) # 64, 1
        rewards = torch.cat(rewards) # 64
        next_states = torch.cat(next_states) # 64, 2

        # print("state_shape: {0}, actions_shape: {1}, rewards_shape: {2}, next_states: {3}".format(states.shape, actions.shape, rewards.shape, next_states.shape))

        current_q = self.model(states).gather(1, actions) # 64, 1. 그때 그때 한 행동들의 가치

        max_next_q = self.model(next_states).detach().max(1)[0] # 64 # 다음 상황에서 가장 클 행동의 가치
        expected_q = rewards + (self.GAMMA * max_next_q) # 64
        # print("mnq shape: {0}, eq shape: {1}".format(max_next_q.shape, expected_q.shape))

        loss = torch.nn.functional.mse_loss(current_q.squeeze(), expected_q)
        print("loss: {0}".format(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return 0

    def save_net(self):
        torch.save(self.model, "./self_made/model.pt")
        torch.save(self.model.state_dict(), "./self_made/model_state_dict.pt")
        