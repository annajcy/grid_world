from typing import Tuple, Dict
import numpy as np
from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction

class TabularGridWorldMDP(GridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state)
        self.discount_factor = discount_factor
        self.rng = rng

        self.policy: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 1.0 / len(self.action_space.actions)
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }
        
    def initialize(self) -> None:
        super().initialize()
        self.policy = {
            (state, action): 1.0 / len(self.action_space.actions)
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }
    

    def get_state_policy(self, state: GridWorldState) -> Dict[GridWorldAction, float]:
        return {
            action: self.policy[(state, action)]
            for action in self.action_space.actions
        }

    # solve state value using bellman equation
    def solve_state_value(self, steps: int=100) -> Dict[GridWorldState, float]:
        state_value: Dict[GridWorldState, float] = { state: 0.0 for state in self.state_space.to_list() }
        for _ in range(steps):
            new_state_value = state_value.copy()
            for state in self.state_space.to_list():
                value = 0.0
                for action in self.action_space.actions:
                    (next_state, reward) = self.transition(state, action)
                    action_prob = self.policy[(state, action)]
                    value += action_prob * (reward + self.discount_factor * state_value[next_state])  # v(s) = Σ_a π(a|s) [ R(s,a) + γ * v(s') ]
                new_state_value[state] = value
            state_value = new_state_value
        return state_value
    
    def get_state_action_value(self, state: GridWorldState, action: GridWorldAction, state_value_table: Dict[GridWorldState, float]) -> float:
        (next_state, reward) = self.transition(state, action)
        return reward + self.discount_factor * state_value_table[next_state]  # Q(s,a) = R(s,a) + γ * v(s')
    
    def get_optimal_action(self, state: GridWorldState, state_value_table: Dict[GridWorldState, float]) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = self.get_state_action_value(state, action, state_value_table)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample()

    def decide(self, state: GridWorldState) -> GridWorldAction:
        action_probs = self.get_state_policy(state)
        rand_val = self.rng.uniform(0, 1)
        cumulative = 0.0
        for action, prob in action_probs.items():
            cumulative += prob
            if rand_val < cumulative:
                return action
        return self.action_space.sample()  # Fallback, should not reach here if probs sum to 1

    # policy iteration (truncated version)
    def policy_iteration(self, threshold: float=1e-4, solve_state_value_steps: int=1000) -> None:
        while True:
            # value evaluation
            state_values = self.solve_state_value(solve_state_value_steps) 
            
            #policy improvment
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_optimal_action(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
                    
            # check convergence
            max_change = max(abs(new_policy[key] - self.policy[key]) for key in self.policy.keys())
            self.policy = new_policy
            if max_change < threshold:
                break
            
    def value_iteration(self, threshold: float=1e-4) -> None:
        state_values = { state: 0.0 for state in self.state_space.to_list() }
        while True:
            # policy update
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_optimal_action(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
            self.policy = new_policy   
            
            # value update
            new_state_values = state_values.copy()
            for state in self.state_space.to_list():
                best_value = float('-inf')
                for action in self.action_space.actions:
                    q_value = self.get_state_action_value(state, action, state_values)
                    if q_value > best_value:
                        best_value = q_value
                new_state_values[state] = best_value
              
            # check convergence
            max_change = max(abs(state_values[s] - new_state_values[s]) for s in self.state_space.to_list())
            
            if max_change < threshold:
                break
            
            state_values = new_state_values
