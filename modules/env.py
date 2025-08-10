import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ActionSpace(ABC):
    def __init__(self) -> None:
        pass
    
    @property
    @abstractmethod
    def n(self) -> int:
        """Return number of actions"""
    
    @property
    @abstractmethod
    def actions(self) -> dict | list:
        """Return the actions"""
    
    @abstractmethod
    def sample(self) -> int:
        """Return a sample of the action space"""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of actions"""


class Env(ABC):
    
    def __init__(self) -> None:
        self.action_space: ActionSpace
        
    @abstractmethod
    def reset(self) -> tuple[np.ndarray, dict]:
        """Resets the environment and returns: (observation, info)"""
    
    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment with the provided action
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        
    @abstractmethod
    def observation(self) -> np.ndarray:
        """Returns an observations of the current environment"""
    
    @abstractmethod
    def render(self, *args, **kwargs):
        """Renders the environment"""

    @abstractmethod
    def close(self):
        """Closes the environment"""