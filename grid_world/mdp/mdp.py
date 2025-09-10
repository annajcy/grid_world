from typing import Any, Tuple, List, Union, TypeVar, Generic, Dict
from abc import ABC, abstractmethod

T_State = TypeVar('T_State', bound='State')
T_Action = TypeVar('T_Action', bound='Action')

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


class MDP(ABC, Generic[T_State, T_Action]):
    @abstractmethod
    def reset(self) -> T_State:
        pass

    @abstractmethod
    def step(self, action: T_Action) -> Tuple[T_State, float, bool, Dict[str, Any]]:
        pass
  