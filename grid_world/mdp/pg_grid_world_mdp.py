from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar
import numpy as np
from torch import nn
import torch.nn.functional as F
import tqdm

from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction
import torch

class PolicyNet(nn.Module):
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
    
    @classmethod
    def action_index_to_action(cls, index: int) -> GridWorldAction:
        mapping = {
            0: GridWorldAction.up(),
            1: GridWorldAction.down(),
            2: GridWorldAction.left(),
            3: GridWorldAction.right(),
        }
        return mapping[index]

    @classmethod
    def action_to_action_index(cls, action: GridWorldAction) -> int:
        mapping = {
            GridWorldAction.up(): 0,
            GridWorldAction.down(): 1,
            GridWorldAction.left(): 2,
            GridWorldAction.right(): 3,
        }
        return mapping[action]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

PolicyNetType = TypeVar('PolicyNetType', bound=PolicyNet)

class PolicyGradientGridWorldMDP(GridWorldMDP, Generic[PolicyNetType]):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState,
                 policy_net: PolicyNetType, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)
        self.policy = policy_net
        self.initial_policy_weights = policy_net.state_dict().copy()
        
    def initialize(self) -> None:
        super().initialize()
        self.policy.load_state_dict(self.initial_policy_weights)
        
    def get_state_policy(self, state: GridWorldState) -> Dict[GridWorldAction, float]:
        state_tensor = self.policy.state_to_input_tensor(state)
        with torch.no_grad():
            action_probs_tensor = F.softmax(self.policy.forward(state_tensor), dim=-1)
        action_probs = self.policy.output_tensor_to_action_probs(action_probs_tensor)
        return action_probs
    
    def reinforce(self, episode_count: int, episode_length: int, learning_rate: float = 0.1) -> None:
        self.policy.train()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        for _ in tqdm.trange(episode_count, desc="REINFORCE", dynamic_ncols=True):
            s0 = self.state_space.sample(self.rng)
            samples = self.sample_sar(s0, self.decide(s0), episode_length)
            rewards = [reward for (_, _, reward) in samples]

            qsa_list: List[float] = []
            discounted_return = 0.0
            for reward in reversed(rewards):
                discounted_return = reward + self.discount_factor * discounted_return
                qsa_list.append(discounted_return)
            qsa_list.reverse()

            qsa_tensor = torch.tensor(qsa_list, dtype=torch.float32)
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for (state, action, _), qsa_t in zip(samples, qsa_tensor):
                state_tensor = self.policy.state_to_input_tensor(state)
                logits = self.policy(state_tensor)
                action_index = self.policy.action_to_action_index(action)
                log_probs_all = F.log_softmax(logits, dim=-1) 
                log_prob_at = log_probs_all[0, action_index]
                loss = loss - log_prob_at * qsa_t
                
            loss.backward()
            optimizer.step()
