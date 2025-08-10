import numpy as np
import math
import random
import os
# from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from pathlib import Path

from gridworld_env import GridWorldEnv
from dqn import DQN, ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F













class GridWorldAgent():
    
    def __init__(
        self,
        env: GridWorldEnv,
        lr = 3e-4,
        batch_size = 128,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.01,
        eps_decay = 2500,
        tau = 0.005,
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
        n_actions = env.ACTIONS.__len__()
        
        # Get number of state observations
        state, _ = env.reset()
        n_observations = len(state)
        
        self.policy_model = DQN(n_observations=n_observations, n_actions=n_actions).to(device=self.device)
        self.target_model = DQN(n_observations=n_observations, n_actions=n_actions).to(device=self.device)
        
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10_000)
        
        self.steps_done = 0
        self.episode_durations = []
        
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Essentially take the argmax of the action dim
                return self.policy_model(state).max(dim=1).indices.view(1,1)
        
        else:
            return torch.tensor([[self.env.sample_action_space()]], device=self.device, dtype=torch.long)
    
    
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
        
        batch = Transition(*zip(*transitions)) # same as: states, actions, next_states, rewards = zip(*transitions)
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
    
    
    def train(self, num_episodes = 500, plot = True):
        
        print(f"Started training...\n  Device: {self.device}\n  Episodes: {num_episodes}")
        
        # Ensure the models are set to train mode
        self.policy_model.train()
        self.target_model.train()
        
        for i_episode in tqdm(range(num_episodes)):
            
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Iterate through timesteps
            for t in count():
                action = self.select_action(state=state)
                # action_compat = int(action.view(-1)[0].item())
                observation, reward, terminated, truncated, _ = self.env.step(action=int(action.view(-1)[0].item()))
                reward = torch.tensor([reward], device=self.device)
                
                # Save new state after action has been performed
                if terminated: next_state = None
                else: next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                
                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_model_state_dict = self.target_model.state_dict()
                policy_model_state_dict = self.policy_model.state_dict()
                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key]*self.TAU + target_model_state_dict[key]*(1-self.TAU)
                self.target_model.load_state_dict(target_model_state_dict)
                
                # Check if the env has terminated or truncated
                if terminated or truncated:
                    self.episode_durations.append(t + 1)
                    if plot: self.plot_durations()
                    break
        
        print("Training Complete!")
        self.plot_durations(show_result=True)
    
    
    def eval(self, num_episodes: int, render=False, render_scale=50, render_fps=2):
        
        # Put model in eval mode
        self.policy_model.eval()
        scores = []
        
        for _ in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            score = 0.0
            done = False
            if render: self.env.render(scale=render_scale, fps=render_fps)
            
            with torch.no_grad():
                while not done:
                    q: torch.Tensor = self.policy_model(state)
                    action = int(q.argmax(dim=1).item()) # Choose action with biggest q-value
                    observation, reward, terminated, truncated, _ = self.env.step(action=action)
                    score += reward
                    done = terminated or truncated
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    if render: self.env.render(scale=render_scale, fps=render_fps)
            
            scores.append(score)
        
        print(f"Average score over {num_episodes} episodes: {sum(scores)/len(scores):.4f}")
        if render: self.env.close()
    
    
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
    
    
    def plot_durations(self, show_result=False, moving_average_window_size=50):
        import matplotlib
        import matplotlib.pyplot as plt
        from IPython import display
        import torch

        is_ipython = 'inline' in matplotlib.get_backend()
        plt.ion()

        plt.figure(1)
        durations_t = torch.as_tensor(self.episode_durations, dtype=torch.float32)
        n = durations_t.numel()

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')

        plt.xlabel('Episode')
        plt.ylabel('Duration')

        if n > 0:
            # raw durations
            plt.plot(durations_t.numpy(), label='Episode duration')

            # variable-window moving average: window = min(i, ma_window)
            cumsum = torch.cat([torch.zeros(1), durations_t]).cumsum(0)   # len n+1
            idx = torch.arange(1, n + 1, dtype=torch.long)                # 1..n
            win = torch.minimum(idx, torch.tensor(moving_average_window_size, dtype=torch.long))
            start = idx - win                                             # indices to subtract
            means = (cumsum[idx] - cumsum[start]) / win                   # length n

            plt.plot(means.numpy(), label=f'Avg (≤ {moving_average_window_size})')
            plt.legend(loc='best')

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

        if show_result:
            plt.ioff()
            plt.show()
    


def main():
    
    obstacle_positions = [
        (0, 5), (0, 10),
        (1, 4), (1, 6), (1, 8),
        (2, 1), (2, 8),
        (3, 1), (3, 4), (3, 5), (3, 7), (3, 8), (3, 10),
        (4, 0), (4, 1), (4, 2), (4, 6),
        (5, 5), (5, 9),
        (6, 1), (6, 4), (6, 8),
        (7, 3), (7, 5),
        (8, 1), (8, 3), (8, 4),
        (9, 0), (9, 1), (9, 7),
        (10, 0), (10, 7), (10, 8),
    ]
    
    env = GridWorldEnv(
        world_size=(11,11),
        start_position=(0,0),
        goal_position=(10,10),
        obstacle_positions=obstacle_positions,#[(3,3), (2,3), (1,3)],
        max_steps=150,
    )
    # env.render(fps=0.25)
    
    agent = GridWorldAgent(env=env)
    
    num_episodes = 500
    model_path = Path(f"models/gridworld_agent_{num_episodes}")
    model_path.parent.mkdir(exist_ok=True)
    
    # agent.train(num_episodes=num_episodes, plot=True)
    # agent.save(model_path)
    # agent.eval(num_episodes=5, render=True)
    
    agent.load(model_path)
    agent.eval(num_episodes=3, render=True, render_fps=10)
    


if __name__ == "__main__":
    main()