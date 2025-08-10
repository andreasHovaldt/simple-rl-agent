from pathlib import Path
from environments.gridworld import GridWorldEnv, GridWorldAgent



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
        obstacle_positions=obstacle_positions,
        max_steps=150,
    )
    # env.render(fps=0.25)
    
    agent = GridWorldAgent(env=env)
    
    num_episodes = 750
    model_path = Path(f"models/gridworld_agent_{num_episodes}")
    model_path.parent.mkdir(exist_ok=True)
    
    agent.train(num_episodes=num_episodes, plot=True)
    agent.save(model_path)
    agent.eval(num_episodes=3, render=True, render_fps=10)
    
    # agent.load(model_path)
    # agent.eval(num_episodes=3, render=True, render_fps=10)
    


if __name__ == "__main__":
    main()