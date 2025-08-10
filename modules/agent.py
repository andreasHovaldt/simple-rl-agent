import math
import random

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import modules
import policies



class DQNAgent():
    def __init__(
        self,
        env: modules.Env,
        lr = 3e-4,
        batch_size = 128,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.01,
        eps_decay = 2500,
        tau = 0.005,
        replay_memory_size = 10_000,
    ) -> None:
        
        self.env = env
        self.LR = lr
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        # Get number of available actions
        n_actions = env.action_space.n
        
        # Get number of state observations
        state, _ = env.reset()
        n_observations = len(state)
        self.policy_model = policies.DQN.Model(n_observations=n_observations, n_actions=n_actions).to(device=self.device)
        self.target_model = policies.DQN.Model(n_observations=n_observations, n_actions=n_actions).to(device=self.device)
        
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr, amsgrad=True)
        self.memory = policies.DQN.ReplayMemory(size=replay_memory_size)
        
        self.steps_done = 0
        
    
    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Essentially take the argmax of the action dim
                return self.policy_model(state).max(dim=1).indices.view(1,1)
        
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
    
    
    def _optimize_model(self):
        # Ensure replay memory has enough recorded state-action pairs
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        # Unpack batch of memory samples
        transitions = self.memory.sample(batch_size=self.BATCH_SIZE)
        
        # transitions = [
        #     Transition(s_0, a_0, ns_0, r_0),
        #     Transition(s_1, a_1, ns_1, r_1),
        #     ...
        #     Transition(s_n, a_n, ns_n, r_n),
        # ]
        
        # zip(*transitions)
        # (
        #     (s_0,  s_1,  ..., s_n),
        #     (a_0,  a_1,  ..., a_n),
        #     (ns_0, ns_1, ..., ns_n),
        #     (r_0,  r_1,  ..., r_n),
        # )
        
        batch = modules.Transition(*zip(*transitions)) # same as: states, actions, next_states, rewards = zip(*transitions)
        # batch.action     == (s_0,  s_1,  ..., s_n)
        # batch.action     == (a_0,  a_1,  ..., a_n)
        # batch.next_state == (ns_0, ns_1, ..., ns_n)
        # batch.reward     == (r_0,  r_1,  ..., r_n)
        
        # Convert to tensor
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Get mask of non-final states -> states where the game ends, thus next_state is none 
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model.forward(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_state).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        # Compute Huber loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()
    
    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    
    def eval(self, *args, **kwargs):
        raise NotImplementedError
    
    
    def save(self, save_path: Path):
        if save_path.suffix != ".pt":
            save_path = save_path.with_suffix(".pt")
        torch.save(self.policy_model.state_dict(), save_path)
        return True
    
    
    def load(self, load_path: Path):
        if load_path.suffix != ".pt":
            load_path = load_path.with_suffix(".pt")
        self.policy_model.load_state_dict(torch.load(load_path, map_location=self.device))
        return True