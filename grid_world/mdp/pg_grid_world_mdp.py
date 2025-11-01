from abc import abstractmethod
from re import S
from sre_compile import dis
from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar
import numpy as np
from torch import P, nn

from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction
import torch

class PNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def state_to_input_tensor(cls, state: GridWorldState) -> torch.Tensor:
        state_array = np.array([state.x, state.y], dtype=np.float32)
        return torch.tensor(state_array).unsqueeze(0)
    
    @classmethod
    def input_tensor_to_state(cls, tensor: torch.Tensor) -> GridWorldState:
        tensor = tensor.squeeze(0)
        state = GridWorldState(x=int(tensor[0]), y=int(tensor[1]))
        return state
    
    @classmethod
    def action_probs_to_output_tensor(cls, action_probs: Dict[GridWorldAction, float]) -> torch.Tensor:
        probs_array = np.array([
            action_probs[GridWorldAction.up()],
            action_probs[GridWorldAction.down()],
            action_probs[GridWorldAction.left()],
            action_probs[GridWorldAction.right()],
        ], dtype=np.float32)
        return torch.tensor(probs_array).unsqueeze(0)
    
    @classmethod
    def output_tensor_to_action_probs(cls, tensor: torch.Tensor) -> Dict[GridWorldAction, float]:
        tensor = tensor.squeeze(0)
        action_probs = {
            GridWorldAction.up(): float(tensor[0]),
            GridWorldAction.down(): float(tensor[1]),
            GridWorldAction.left(): float(tensor[2]),
            GridWorldAction.right(): float(tensor[3]),
        }
        return action_probs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

PNetType = TypeVar('PNetType', bound=PNet)

class PolicyGradientGridWorldMDP(GridWorldMDP, Generic[PNetType]):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState,
                 policy: PNetType, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)
        self.policy = policy
        self.initial_policy_weights = policy.state_dict().copy()
        
    def initialize(self) -> None:
        super().initialize()
        self.policy.load_state_dict(self.initial_policy_weights)
        
    def get_state_policy(self, state: GridWorldState) -> Dict[GridWorldAction, float]:
        state_tensor = self.policy.state_to_input_tensor(state)
        with torch.no_grad():
            action_probs_tensor = self.policy.forward(state_tensor)
        action_probs = self.policy.output_tensor_to_action_probs(action_probs_tensor)
        return action_probs
    
    def reinforce(self):
        pass
    
