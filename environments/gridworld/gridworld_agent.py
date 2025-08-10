from itertools import count
from tqdm import tqdm
from pathlib import Path
import torch

import modules



class GridWorldAgent(modules.DQNAgent):
    
    def __init__(self, env: modules.Env, lr=0.0003, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=2500, tau=0.005, replay_memory_size=10000) -> None:
        super().__init__(env, lr, batch_size, gamma, eps_start, eps_end, eps_decay, tau, replay_memory_size)
        self.episode_durations = []
    
    
    def train(self, num_episodes = 500, plot = True):
        
        print(f"Started training...\n  Device: {self.device}\n  Episodes: {num_episodes}")
        
        # Ensure the models are set to train mode
        self.policy_model.train()
        self.target_model.train()
        
        for _ in tqdm(range(num_episodes)):
            
            # Initialize the environment and get its state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Iterate through timesteps
            for t in count():
                action = self._select_action(state=state)
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
    