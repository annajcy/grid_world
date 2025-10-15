from typing import Any, Tuple, List, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass

T_State = TypeVar('T_State', bound='State')
T_StateSpace = TypeVar('T_StateSpace', bound='StateSpace')
T_Action = TypeVar('T_Action', bound='Action')
T_ActionSpace = TypeVar('T_ActionSpace', bound='ActionSpace')

class State(ABC):
    @abstractmethod
    def to_list(self) -> List[Any]:
        pass
    
class StateSpace(ABC, Generic[T_State]):
    @abstractmethod
    def sample(self) -> T_State:
        pass

    @abstractmethod
    def contains(self, state: T_State) -> bool:
        pass
    
    @abstractmethod
    def to_list(self)-> List[Any]:
        pass

class Action(ABC):
    @abstractmethod
    def to_list(self) -> List[Any]:
        pass
    
class ActionSpace(ABC, Generic[T_Action]):
    @abstractmethod
    def sample(self) -> T_Action:
        pass

    @abstractmethod
    def contains(self, action: T_Action) -> bool:
        pass
    
@dataclass(frozen=True, slots=False)
class MDPStepResult(Generic[T_State, T_Action]):
    current_state: T_State
    action: T_Action
    next_state: T_State
    reward: float
    terminated: bool
    truncated: bool

class MDP(ABC, Generic[T_State, T_StateSpace, T_Action, T_ActionSpace]):

    def __init__(self, state_space: T_StateSpace, action_space: T_ActionSpace, initial_state: T_State) -> None:
        self.state_space: T_StateSpace = state_space
        self.action_space: T_ActionSpace = action_space
        self.initial_state: T_State = initial_state
        self.current_state: T_State = initial_state
        assert self.state_space.contains(self.current_state), "Initial state not in state space"

    @abstractmethod
    def initialize(self) -> Any:
        pass
    
    @abstractmethod
    def is_terminated(self, state: T_State) -> bool:
        pass
    
    @abstractmethod
    def is_truncated(self, state: T_State) -> bool:
        pass
    
    @abstractmethod
    def transition(self, state: T_State, action: T_Action) -> Tuple[T_State, float]:
        pass
    
    @abstractmethod
    def decide(self, state: T_State) -> T_Action:
        pass

    def step_from_state_action(self, state: T_State, action: T_Action, update_state: bool=False) -> MDPStepResult[T_State, T_Action]:
        assert self.state_space.contains(state), "State not in state space"
        assert self.action_space.contains(action), "Action not in action space"
        
        (next_state, reward) = self.transition(state, action)
        assert self.state_space.contains(next_state), "Next state not in state space"
        
        terminated = self.is_terminated(next_state)
        truncated = self.is_truncated(next_state)
        
        if update_state:
            self.current_state = next_state
        
        return MDPStepResult(
            current_state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated
        )

    def step_from_action(self, action: T_Action, update_state: bool=False) -> MDPStepResult[T_State, T_Action]:
        return self.step_from_state_action(
            self.current_state, 
            action, 
            update_state=update_state
        )
    
    def step(self, update_state: bool=True) -> MDPStepResult[T_State, T_Action]:
        return self.step_from_state_action(
            self.current_state, 
            self.decide(self.current_state), 
            update_state=update_state
        )