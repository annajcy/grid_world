from abc import abstractmethod
from re import S
from sre_compile import dis
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar
import numpy as np
from torch import P, nn

from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction
from .pg_grid_world_mdp import PolicyGradientGridWorldMDP, PolicyNetType
from .vf_grid_world_mdp import QNetType

import torch

class ActorCriticGridWorldMDP(
    PolicyGradientGridWorldMDP[PolicyNetType],
    Generic[PolicyNetType, QNetType]
):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        policy_net : PolicyNetType,
        q_net : QNetType,
        discount_factor: float = 0.9,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            initial_state=initial_state,
            goal_state=goal_state,
            policy_net=policy_net,
            discount_factor=discount_factor,
            rng=rng,
        )
        self.q_net = q_net
        
    def Qsa_vf(self, state: GridWorldState, action: GridWorldAction) -> Any:    
        state_action_tensor = self.q_net.state_action_to_input_tensor(state, action)
        return self.q_net(state_action_tensor)
    
    def get_opt_a_from_Qsa_vf(self, state: GridWorldState) -> GridWorldAction:
        best_action = None
        best_value = float("-inf")
        for action in self.action_space.actions:
            q_value = self.Qsa_vf(state, action).item()
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)
    
    
        
    
