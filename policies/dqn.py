import random
from collections import deque
import modules

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayMemory(object):
    def __init__(self, size) -> None:
        self.memory = deque([], maxlen=size)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(modules.Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Model(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x) -> torch.Tensor: # (batch_size, n_observations)
        x = F.relu(self.layer1(x)) # (batch_size, 128)
        x = F.relu(self.layer2(x)) # (batch_size, 128)
        return self.layer3(x) # (batch_size, n_actions)