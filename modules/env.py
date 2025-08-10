import numpy as np
from collections import namedtuple


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ActionSpace():
    def __init__(self) -> None:
        pass
    
    @property
    def n(self) -> int:
        """Return number of actions"""
        raise NotImplementedError
    
    @property
    def actions(self):
        """Return the actions"""
        raise NotImplementedError
    
    def sample(self):
        """Return a sample of the action space"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of actions"""
        raise NotImplementedError


class Env():
    action_space = ActionSpace()
    
    def __init__(self) -> None:
        pass
        
    def reset(self) -> tuple[np.ndarray, dict]:
        """Resets the environment and returns: (observation, info)"""
        raise NotImplementedError
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment with the provided action
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        raise NotImplementedError
        
    def observation(self) -> np.ndarray:
        """Returns an observations of the current environment"""
        raise NotImplementedError
    
    def render(self, *args, **kwargs):
        """Renders the environment"""
        raise NotImplementedError

    def close(self):
        """Closes the environment"""
        raise NotImplementedError