from typing import Dict, Optional, Tuple
import numpy as np
from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction

class TabularGridWorldMDP(GridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState,
                 policy: Optional[Dict[Tuple[GridWorldState, GridWorldAction], float]] = None, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)

        if policy is None or not isinstance(policy, dict):
            self.policy: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
                (state, action): 1.0 / len(self.action_space.actions)
                for state in self.state_space.to_list() 
                for action in self.action_space.actions
            }
            self.initial_policy = self.policy.copy()
        else:
            self.policy = policy
            self.initial_policy = self.policy.copy()
            
    def initialize(self) -> None:
        super().initialize()
        self.policy = self.initial_policy.copy()

    def get_state_policy(self, state: GridWorldState) -> Dict[GridWorldAction, float]:
        return {
            action: self.policy[(state, action)]
            for action in self.action_space.actions
        }
    
    # value iteration 
    def value_iteration(self, threshold: float=1e-4) -> None:
        state_values = { state: 0.0 for state in self.state_space.to_list() }
        while True:
            # policy update
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_opt_action_from_Vs(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
            self.policy = new_policy
            
            # value update
            new_state_values = state_values.copy()
            for state in self.state_space.to_list():
                best_value = float('-inf')
                for action in self.action_space.actions:
                    q_value = self.get_qsa_from_Vs(state, action, state_values)
                    if q_value > best_value:
                        best_value = q_value
                new_state_values[state] = best_value
              
            # check convergence
            max_change = max(abs(state_values[s] - new_state_values[s]) for s in self.state_space.to_list())
            if max_change < threshold:
                break
            
            state_values = new_state_values
            
    # policy iteration (truncated version)
    def policy_iteration(self, threshold: float=1e-4, solve_state_value_steps: int=1000) -> None:
        while True:
            # value evaluation
            state_values = self.solve_Vs(solve_state_value_steps) 
            
            #policy improvment
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_opt_action_from_Vs(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
                    
            # check convergence
            max_change = max(abs(new_policy[key] - self.policy[key]) for key in self.policy.keys())
            self.policy = new_policy
            if max_change < threshold:
                break
    
    
    
    
            
            
                
                
    
