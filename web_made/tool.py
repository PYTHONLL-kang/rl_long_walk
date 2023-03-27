import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
                                    )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class Replay_memory:
    def __init__(self):
        self.storage = []
        self.max_len = 1000
        self.next_idx = 0
    
    def _len_(self):
        return len(self.storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.max_len

    def encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self.storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size=64):
        indices = np.random.randint(0, len(self.storage) - 1, size=batch_size)
        return self.encode_sample(indices)