import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 = X, -1 = O
        self.done = False
        return self._get_obs()

    def step(self, action):
        row, col = divmod(action, 3)

        if self.board[row, col] != 0 or self.done:
            raise ValueError("Invalid action")

        self.board[row, col] = self.current_player

        winner = self._check_winner()
        self.done = winner is not None or np.all(self.board != 0)

        reward = 0
        if winner == self.current_player:
            reward = 1
        elif self.done:
            reward = 0.5  # draw

        obs = self._get_obs()
        info = {'winner': winner}
        self.current_player *= -1
        return obs, reward, self.done, info

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[val] for val in row))
        print()

    def _get_obs(self):
        return self.board.flatten().copy()

    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def _check_winner(self):
        for player in [1, -1]:
            # Rows, columns, diagonals
            if any(np.all(self.board[i, :] == player) for i in range(3)) \
               or any(np.all(self.board[:, j] == player) for j in range(3)) \
               or np.all(np.diag(self.board) == player) \
               or np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        return None
