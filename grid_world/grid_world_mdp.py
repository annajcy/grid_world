from typing import List, Tuple
from dataclasses import dataclass

from rl.mdp import MDP, Action, ActionSpace, State, StateSpace
import numpy as np

@dataclass(frozen=True, slots=True)
class GridWorldState(State):
    x: int
    y: int 

    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def to_list(self) -> List[int]:
        return [self.x, self.y]

class GridWorldStateSpace(StateSpace[GridWorldState]):
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def sample(self, rng: np.random.Generator=np.random.default_rng()) -> GridWorldState:
        x = rng.choice(np.arange(self.width), 1)[0]
        y = rng.choice(np.arange(self.height), 1)[0]
        return GridWorldState(x, y)

    def contains(self, state: GridWorldState) -> bool:
        if not isinstance(state, GridWorldState):
            return False
        x, y = state.position()
        return 0 <= x < self.width and 0 <= y < self.height

    def to_list(self) -> List[GridWorldState]:
        return [GridWorldState(x, y) for y in range(self.height) for x in range(self.width)]

@dataclass(frozen=True, slots=True)
class GridWorldAction(Action):
    dx: int
    dy: int
        
    def to_list(self) -> List[int]:
        return [self.dx, self.dy]
    
    def __iter__(self) :
        return iter((self.dx, self.dy))
    
    @classmethod
    def left(cls) -> 'GridWorldAction':
        return cls(-1, 0)
    
    @classmethod  
    def right(cls) -> 'GridWorldAction':
        return cls(1, 0)
    
    @classmethod
    def up(cls) -> 'GridWorldAction':
        return cls(0, -1)
    
    @classmethod
    def down(cls) -> 'GridWorldAction':
        return cls(0, 1)

class GridWorldActionSpace(ActionSpace[GridWorldAction]):
    def __init__(self) -> None:
        self.actions = [
            GridWorldAction.left(),
            GridWorldAction.right(), 
            GridWorldAction.up(),
            GridWorldAction.down()
        ]

    def sample(self, rng: np.random.Generator=np.random.default_rng()) -> GridWorldAction:
        idx = rng.choice(np.arange(len(self.actions)), 1)[0]
        return self.actions[idx]

    def contains(self, action: GridWorldAction) -> bool:
        if not isinstance(action, GridWorldAction):
            return False
        return GridWorldAction(action.dx, action.dy) in self.actions
    
    def to_list(self) -> List[GridWorldAction]:
        return [action for action in self.actions]

class GridWorldMDP(MDP[GridWorldState, GridWorldStateSpace, GridWorldAction, GridWorldActionSpace]):
    def __init__(self, 
                 width: int,
                 height: int,
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState) -> None:
        super().__init__(GridWorldStateSpace(width, height), GridWorldActionSpace(), initial_state)
        self.goal_state = goal_state

    def initialize(self) -> None:
        self.current_state = self.initial_state
        
    def transition(self, state: GridWorldState, action: GridWorldAction) -> Tuple[GridWorldState, float]:
        
        def reward(next_state: GridWorldState) -> float:
            if next_state == self.goal_state:
                return 1.0
            elif next_state == self.initial_state:
                return -1.0
            else:
                return -0.1
        
        (x, y) = state.position()
        (dx, dy) = action
        new_state = GridWorldState(x + dx, y + dy)
        if not self.state_space.contains(new_state):
            new_state = state  # stay in the same state if out of bounds

        return (new_state, reward(new_state))

    def is_terminated(self, state: GridWorldState) -> bool:
        return False # continuing task
    
    def is_truncated(self, state: GridWorldState) -> bool:
        return False # continuing task
