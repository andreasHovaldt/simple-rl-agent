from tictactoe_env import TicTacToeEnv

env = TicTacToeEnv()
obs = env.reset()
env.render()

done = False
while not done:
    action = int(input("Enter move (0–8): "))
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
