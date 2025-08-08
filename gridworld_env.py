import numpy as np
import cv2
from copy import deepcopy


class GridWorldEnv():
    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }
    
    WORLD_CONSTANTS = {
        "obstacle": -1,
        "empty": 0,
        "goal": 1, 
        "player": 2,
    }

    # from gymnasium import spaces
    # observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    # action_space = spaces.Discrete(4)
    
    
    def __init__(
            self, 
            world_size: tuple[int, int], 
            start_position: tuple[int, int], 
            goal_position: tuple[int, int], 
            obstacle_positions: list[tuple[int, int],],
            max_steps: int = 50,
        ) -> None:
        
        self.world_size = world_size
        
        self.start_position = (start_position[0] + 1, start_position[1] + 1)
        assert 0 <= start_position[0] < world_size[0] and 0 <= start_position[1] < world_size[1], "Start out of bounds"
        assert (start_position[0], start_position[1]) not in obstacle_positions, "Start on obstacle"
        assert (start_position[0], start_position[1]) != goal_position, "Start equals goal"
        
        self.goal_position = (goal_position[0] + 1, goal_position[1] + 1)
        self.obstacle_positions = obstacle_positions
        self.max_steps = max_steps
        self.world = self._world_builder(size=world_size, goal=goal_position, obstacles=obstacle_positions)
        self.reset()
        
        
    def reset(self):
        self.player_position = self.start_position
        self.terminated = self.truncated = False
        self.steps = 0
        return self.observation(), {}
    
        
    def step(self, action):
        
        # Check if game has ended 
        if self.terminated or self.truncated:
            return self.observation(), 0.0, self.terminated, self.truncated, {}
        
        # Check if action is allowed
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action encountered: {action}")
        
        # Get new position
        movement = self.ACTIONS[action]
        new_position = (self.player_position[0] + movement[0], self.player_position[1] + movement[1])
        
        # Update player position if movement is valid
        hit_wall = True
        if self.world[new_position] != self.WORLD_CONSTANTS["obstacle"]:
            self.player_position = new_position
            hit_wall = False
        
        # Check termination
        if self.player_position == self.goal_position:
            self.terminated = True
            reward = 1.0
            
        # Penalize for walking into wall
        elif hit_wall:
            reward = -0.05
            
        # Small negative reward to discourage redundant movements
        else:
            reward = -0.01
        
        
        self.steps += 1
        self.truncated = self.steps >= self.max_steps
        
        
        return self.observation(), reward, self.terminated, self.truncated, {}
        
    
    def observation(self):
        
        # Get positions
        p_y, p_x = self.player_position
        g_y, g_x = self.goal_position

        # Adjacent tiles
        up    = self.world[p_y - 1, p_x]
        down  = self.world[p_y + 1, p_x]
        left  = self.world[p_y, p_x - 1]
        right = self.world[p_y, p_x + 1]

        # Manhattan distance to goal
        # dist = abs(g_y - p_y) + abs(g_x - p_x)
        
        # Direction to goal
        dy, dx = g_y - p_y, g_x - p_x
        angle = np.arctan2(dy, dx)
    
        # Define observation
        obs = np.array([up, down, left, right, np.cos(angle), np.sin(angle)], dtype=np.float32)
        return obs
    
    
    def render(self, scale=50):
        
        # Map from world constants to RGB colors
        C = self.WORLD_CONSTANTS
        color_map = {
            C["obstacle"]: [0,0,0],
            C["empty"]:    [255,255,255],
            C["goal"]:     [0,255,0],
            C["player"]:   [255,255,0],
        }
        
        # Make a copy so we don't overwrite the real world
        player_world = deepcopy(self.world)
        player_world[self.player_position] = C["player"]
        
        # Convert symbolic world to RGB image
        h, w = player_world.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        for value, color in color_map.items():
            rgb_image[player_world == value] = color

        # Optional: Scale up the image so it's easier to see
        rgb_image = cv2.resize(rgb_image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Show image in window
        cv2.imshow("World", rgb_image)
        cv2.waitKey(1)
        
          
    def render_simple(self):
        C = self.WORLD_CONSTANTS
        symbols = {
            C["player"]: "o", 
            C["goal"]: "X", 
            C["empty"]: " ", 
            C["obstacle"]: "#"
        }
        player_world = deepcopy(self.world)
        player_world[self.player_position] = C["player"]
        
        for row in player_world:
            print(' '.join(symbols[val] for val in row))
        print()
    

    def close(self):
        cv2.destroyAllWindows()


    def _world_builder(self, size, goal, obstacles, dtype=np.int8) -> np.ndarray:
        C = self.WORLD_CONSTANTS
        
        # Ensure world params are valid
        assert 0 <= goal[0] < size[0] and 0 <= goal[1] < size[1], "The goal position must be placed within the world bounds!"
        assert all(0 <= r < size[0] and 0 <= c < size[1] for r, c in obstacles), "All obstacles must be placed within the world bounds!"
        assert not any(goal == obstacle for obstacle in obstacles), "The goal can't be placed on an obstacle!"
        
        # Init world
        world = np.zeros(shape=size, dtype=dtype)
        
        # Add obstacles
        for obstacle in obstacles:
            world[obstacle] = C["obstacle"]
        
        # Add goal
        world[goal] = C["goal"]
        
        # Add boundary
        world = np.pad(world, pad_width=1, mode='constant', constant_values=C["obstacle"])
        
        return world




def main() -> None:
    
    env = GridWorldEnv(
        world_size=(5,5),
        start_position=(0,0),
        goal_position=(4,4),
        obstacle_positions=[(3,3), (2,3), (1,3)],
    )
    
    episodes = 5
    for episode in range(episodes):
        observation, _ = env.reset() # Reset/init env and obtain initial state observation
        terminated = truncated = False
        score = 0
        env.render()
        
        while not terminated and not truncated:
            
            action = np.random.randint(0, 4) # Randomly sample an action within the boundaries of the action space
            observation, reward, terminated, truncated, _ = env.step(action=action) # Step the env forward one time step, performing the defined action
            env.render()
            score += reward # Add received reward to total score
        
        print(f"Episode:{episode} | Score:{score:.4f}")
    
    env.close()


if __name__ == "__main__":
    main()