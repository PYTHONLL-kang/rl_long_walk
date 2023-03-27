import torch
import numpy as np
import math
from collections import deque
from random import *
import random as rand
import tool

class Agent:
    def __init__(self):
        self.steps_done = 0
        self.memory = tool.Replay_memory()

        self.policy = tool.Net()
        self.target = tool.Net()

        self.update_target_network()
        self.target.eval()

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=0.00001)

        self.GAMMA = 0.8 #감마는 할인계수, 에이전트가 현재를 미래보다 더 가치있게 여기는 것
        self.batch_size = 64

    def optimise_td_loss(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()

        with torch.no_grad():
            _, max_next_action = self.policy(next_states).max(1)
            max_next_q_values = self.target(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
        target_q_values = rewards + (1 - dones) * self.GAMMA * max_next_q_values

        input_q_values = self.policy(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = torch.nn.functional.smooth_l1_loss(input_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        self.target.load_state_dict(self.policy.state_dict())

    def act(self, state):
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy(state)
            _, action = q_values.max(1)
            return action.item()