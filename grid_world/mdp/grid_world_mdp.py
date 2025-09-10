from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .mdp import MDP, Action, ActionSpace, State, StateSpace
import random

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

    def sample(self) -> GridWorldState:
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return GridWorldState(x, y)

    def contains(self, state: GridWorldState) -> bool:
        if not isinstance(state, GridWorldState):
            return False
        x, y = state.position()
        return 0 <= x < self.width and 0 <= y < self.height

    def to_list(self) -> List[List[GridWorldState]]:
        return [[GridWorldState(x, y) for y in range(self.height)] for x in range(self.width)]

@dataclass(frozen=True, slots=True)
class GridWorldAction(Action):
    dx: int
    dy: int
        
    def to_list(self) -> List[int]:
        return [self.dx, self.dy]
    
    def __iter__(self):
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
    
    @classmethod
    def stay(cls) -> 'GridWorldAction':
        return cls(0, 0)
        

class GridWorldActionSpace(ActionSpace[GridWorldAction]):
    def __init__(self) -> None:
        self.actions = [
            GridWorldAction.left(),
            GridWorldAction.right(), 
            GridWorldAction.up(),
            GridWorldAction.down(),
            GridWorldAction.stay()
        ]  # left, right, up, down, stay

    def sample(self) -> GridWorldAction:
        return random.choice(self.actions)

    def contains(self, action: GridWorldAction) -> bool:
        if not isinstance(action, GridWorldAction):
            return False
        return (action.dx, action.dy) in self.actions
    
    def to_list(self) -> List[GridWorldAction]:
        return [GridWorldAction(dx, dy) for dx, dy in self.actions]
      
class GridWorldMDP(MDP[GridWorldState, GridWorldAction]):
    def __init__(self, 
                 state_space: GridWorldStateSpace, 
                 action_space: GridWorldActionSpace, 
                 start_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 forbiddens: List[GridWorldState]) -> None:
        self.state_space = state_space
        self.action_space = action_space
        
        if not self.state_space.contains(start_state):
            raise ValueError("Start state is out of bounds")
        self.start_state = start_state
        self.current_state = start_state

        if not self.state_space.contains(goal_state):
            raise ValueError("Goal state is out of bounds")
        self.goal_state = goal_state

        for forbidden in forbiddens:
            if not self.state_space.contains(forbidden):
                raise ValueError("Block state is out of bounds")
        self.forbiddens = forbiddens

    def reset(self) -> GridWorldState:
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action: GridWorldAction) -> Tuple[GridWorldState, float, bool, Dict[str, Any]]:
        (x, y) = self.current_state.position()
        (dx, dy) = action
        new_state = GridWorldState(x + dx, y + dy)
        
        if not self.state_space.contains(new_state):
            return (self.current_state, -1.0, False, {})
        
        if new_state in self.forbiddens:
            self.current_state = new_state
            return (self.current_state, -1.0, False, {})
        
        if new_state == self.goal_state:
            self.current_state = new_state
            return (self.current_state, 1.0, True, {})
        
        return (new_state, 0.0, False, {})

    def decide(self, state: GridWorldState) -> GridWorldAction:
        return self.action_space.sample()