from typing import Any, Set, Tuple, Dict, List
import numpy as np
from abc import abstractmethod

from .grid_world_mdp import GridWorldMDP, GridWorldState, GridWorldAction

import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

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
    
    def get_q(self, state: GridWorldState, action: GridWorldAction, state_value_table: Dict[GridWorldState, float]) -> float:
        (next_state, reward) = self.transition(state, action)
        return reward + self.discount_factor * state_value_table[next_state]  # Q(s,a) = R(s,a) + γ * v(s')
    
    def get_optimal_action_v(self, state: GridWorldState, V: Dict[GridWorldState, float]) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = self.get_q(state, action, V)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)  # Fallback, should not reach here
    
    def get_optimal_action_q(self, state: GridWorldState, Q: Dict[Tuple[GridWorldState, GridWorldAction], float]) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = Q[(state, action)]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)  # Fallback, should not reach here

    def decide(self, state: GridWorldState) -> GridWorldAction:
        action_probs = self.get_state_policy(state)
        rand_val = self.rng.uniform(0, 1)
        cumulative = 0.0
        for action, prob in action_probs.items():
            cumulative += prob
            if rand_val < cumulative:
                return action
        return self.action_space.sample(self.rng)  # Fallback, should not reach here if probs sum to 1
            
    # value iteration 
    def value_iteration(self, threshold: float=1e-4) -> None:
        state_values = { state: 0.0 for state in self.state_space.to_list() }
        while True:
            # policy update
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_optimal_action_v(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
            self.policy = new_policy   
            
            # value update
            new_state_values = state_values.copy()
            for state in self.state_space.to_list():
                best_value = float('-inf')
                for action in self.action_space.actions:
                    q_value = self.get_q(state, action, state_values)
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
            state_values = self.solve_state_value(solve_state_value_steps) 
            
            #policy improvment
            new_policy = self.policy.copy()
            for state in self.state_space.to_list():
                optimal_action = self.get_optimal_action_v(state, state_values)
                for action in self.action_space.to_list():
                    new_policy[(state, action)] = 0.0 if action != optimal_action else 1.0
                    
            # check convergence
            max_change = max(abs(new_policy[key] - self.policy[key]) for key in self.policy.keys())
            self.policy = new_policy
            if max_change < threshold:
                break

class SampledTabularGridWorldMDP(TabularGridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)
    
    def sample_episode_sar(self, state: GridWorldState, action: GridWorldAction, episode_length: int=100) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
        episode = []
        next_state, reward = self.transition(state, action)
        episode.append((state, action, reward))
        state = next_state
        for _ in range(episode_length - 1):
            action = self.decide(state)
            next_state, reward = self.transition(state, action)
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    def sample_episode_sars(self, state: GridWorldState, action: GridWorldAction, episode_length: int=100) -> List[Tuple[GridWorldState, GridWorldAction, float, GridWorldState]]:
        episode = []
        next_state, reward = self.transition(state, action)
        episode.append((state, action, reward, next_state))
        state = next_state
        for _ in range(episode_length - 1):
            action = self.decide(state)
            next_state, reward = self.transition(state, action)
            episode.append((state, action, reward, next_state))
            state = next_state
        return episode

    def sample_episode_sar_all(self, state: GridWorldState, action: GridWorldAction, max_episode_length: int=10000) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
        is_visited: Set[Tuple[GridWorldState, GridWorldAction]] = set()
        max_set_size = len(self.state_space.to_list()) * len(self.action_space.actions)

        episode = []
        next_state, reward = self.transition(state, action)
        episode.append((state, action, reward))
        is_visited.add((state, action))
        state = next_state
        while len(is_visited) < max_set_size and len(episode) < max_episode_length:
            action = self.decide(state)
            next_state, reward = self.transition(state, action)
            episode.append((state, action, reward))
            is_visited.add((state, action))
            state = next_state
        return episode
    
    def get_q_with_episode(self, episode: List[Tuple[GridWorldState, GridWorldAction, float]]) -> Tuple[Tuple[GridWorldState, GridWorldAction], float]:
        G = 0.0
        for i in range(len(episode)):
            G = episode[len(episode) - 1 - i][2] + self.discount_factor * G  # G_t = R_{t+1} + γ * G_{t+1}
        return (episode[0][0], episode[0][1]), G

class MCTabularGridWorldMDP(SampledTabularGridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)
    
    # Monte Carlo basic
    def mc_basic(self, iterations = 100, episode_count: int=10, episode_length: int=20) -> None:
        
        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }
        
        for _ in trange(iterations, desc="MC-basic", dynamic_ncols=True):
            for state in self.state_space.to_list():
                # sample episodes to update Q
                for action in self.action_space.actions:
                    G_total = 0.0
                    for _ in range(episode_count):
                        episode = self.sample_episode_sar(state, action, episode_length)
                        (s, a), G = self.get_q_with_episode(episode)
                        G_total += G
                    G_avg = G_total / episode_count if episode_count > 0 else 0.0
                    
                    Q[(state, action)] = G_avg
                    
                # policy update
                action_values = [Q[(state, action)] for action in self.action_space.actions]
                optimal_action = None
                max_action_value = max(action_values)
                for action, value in zip(self.action_space.actions, action_values):
                    if value == max_action_value:
                        optimal_action = action
                        break
                
                for action in self.action_space.actions:
                    self.policy[(state, action)] = 1.0 if action == optimal_action else 0.0

    def mc_epsilon_greedy(self, episode_count: int=20, episode_length: int=100, epsilon: float=0.1, first_visit: bool=False) -> None:
        
        visit_count: Dict[Tuple[GridWorldState, GridWorldAction], int] = {
            (state, action): 0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }
            
        G_sum : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }

        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }
        
        for _ in trange(episode_count, desc="MC-eps-greedy", dynamic_ncols=True):
            # episode generation from all state-action pairs
            episode = self.sample_episode_sar(self.state_space.sample(self.rng), self.action_space.sample(self.rng), episode_length)
            
            G = 0.0

            for t in range(len(episode)-1, -1, -1):
                (s, a, r) = episode[t]
                G = r + self.discount_factor * G  # G_t = R_{t+1} + γ * G_{t+1}
                
                if first_visit:
                    if (s, a) in [(x[0], x[1]) for x in episode[0:t]]:
                        continue  # skip if not the first visit in this episode
                    
                G_sum[(s, a)] += G  # accumulate returns
                visit_count[(s, a)] += 1
                Q[(s, a)] = G_sum[(s, a)] / visit_count[(s, a)]  # average return
                
                # policy update
                action_values = [Q[(s, action)] for action in self.action_space.actions]
                optimal_action = None
                max_action_value = max(action_values)
                for action, value in zip(self.action_space.actions, action_values):
                    if value == max_action_value:
                        optimal_action = action
                        break

                for action in self.action_space.actions:
                    if action == optimal_action:
                        self.policy[(s, action)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(s, action)] = (epsilon / len(self.action_space.actions))

class TDTabularGridWorldMDP(SampledTabularGridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 learning_rate: float=0.1,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, rng)
        self.learning_rate = learning_rate
    
    # TD(0) learning
    def td0(self, initial_state: GridWorldState, episode_count: int, episode_length: int,  epsilon: float=0.1) -> None:
        
        V : Dict[GridWorldState, float] = {
            S: 0.0
            for S in self.state_space.to_list()
        }
        
        for _ in trange(episode_count, desc="TD(0)", dynamic_ncols=True):
            state: GridWorldState = initial_state
            
            for _ in range(episode_length):
                
                action = self.decide(state)
                next_state, reward = self.transition(state, action)
                
                # TD(0) update
                td_target = reward + self.discount_factor * V[next_state]
                V[state] = V[state] + self.learning_rate * (td_target - V[state])  # V(s) += α [ R(s,a) + γ * V(s') - V(s) ]
                
                optimal_action = self.get_optimal_action_v(state, V)
                
                for action in self.action_space.actions:
                    if action == optimal_action:
                        self.policy[(state, action)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, action)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state

    def sarsa(self, initial_state: GridWorldState, episode_count: int, episode_length: int, epsilon: float=0.1) -> None:
        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="SARSA", dynamic_ncols=True):
            state: GridWorldState = initial_state
            action: GridWorldAction = self.decide(state)

            for _ in range(episode_length):
        
                next_state, reward = self.transition(state, action)
                next_action = self.decide(next_state)
                
                td_target = reward + self.discount_factor * Q[(next_state, next_action)]
                
                # SARSA update
                Q[(state, action)] = Q[(state, action)] + self.learning_rate * (td_target - Q[(state, action)])  # Q(s,a) += α [ R(s,a) + γ * Q(s',a') - Q(s,a) ]

                optimal_action = self.get_optimal_action_q(state, Q)
                        
                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, a)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state
                action = next_action


    def expected_sarsa(self, initial_state: GridWorldState, episode_count: int, episode_length: int, epsilon: float=0.1) -> None:
        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="Expected SARSA", dynamic_ncols=True):
            state: GridWorldState = initial_state
            
            for _ in range(episode_length):
            
                action: GridWorldAction = self.decide(state)
                next_state, reward = self.transition(state, action)
                
                expected_q = 0.0
                               
                for a in self.action_space.actions:
                    action_prob = self.policy[(next_state, a)]
                    expected_q += action_prob * Q[(next_state, a)]  # E[Q(s',a')]
                        
                td_target = reward + self.discount_factor * expected_q
                # SARSA update
                Q[(state, action)] = Q[(state, action)] + self.learning_rate * (td_target - Q[(state, action)])  # Q(s,a) += α [ R(s,a) + γ * E[Q(s',a')] - Q(s,a) ]

                optimal_action = self.get_optimal_action_q(state, Q)
                        
                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, a)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state

    def q_learning_on_policy(self, initial_state: GridWorldState, episode_count: int, episode_length: int, epsilon: float=0.1) -> None:
        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="Q-learning (on)", dynamic_ncols=True):
            state: GridWorldState = initial_state
            
            for _ in range(episode_length):
            
                action: GridWorldAction = self.decide(state)
                next_state, reward = self.transition(state, action)
                
                # Q-learning update
                max_next_q = max(Q[(next_state, a)] for a in self.action_space.actions)
                td_target = reward + self.discount_factor * max_next_q
                Q[(state, action)] = Q[(state, action)] + self.learning_rate * (td_target - Q[(state, action)])  # Q(s,a) += α [ R(s,a) + γ * max_a' Q(s',a') - Q(s,a) ]

                optimal_action = self.get_optimal_action_q(state, Q)
                        
                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, a)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state
                
    def q_learning_off_policy(self, sample_list: List[List[Tuple[GridWorldState, GridWorldAction, float]]]) -> None:
        
        Q : Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list() 
            for action in self.action_space.actions
        }

        for episode in tqdm(sample_list, desc="Q-learning (off) episodes", dynamic_ncols=True):
            for t in range(len(episode) - 1):
                (state, action, reward) = episode[t]
                
                # Q-learning update
                next_state = episode[t+1][0] 
                max_next_q = max(Q[(next_state, a)] for a in self.action_space.actions)
                td_target = reward + self.discount_factor * max_next_q
                Q[(state, action)] = Q[(state, action)] + self.learning_rate * (td_target - Q[(state, action)])  # Q(s,a) += α [ R(s,a) + γ * max_a' Q(s',a') - Q(s,a) ]

                optimal_action = self.get_optimal_action_q(state, Q)
                        
                for a in self.action_space.actions:
                    self.policy[(state, a)] = 1.0 if a == optimal_action else 0.0
                    
class ValueFunctionTabularGridWorldMDP(TDTabularGridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 learning_rate: float=0.1,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, learning_rate, rng)

    @abstractmethod
    def Q_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        pass
    
    @abstractmethod
    def dQ_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        pass
    

## Neural Network Model for Q(s,a), s(x y) for 2 dims, action (dx dy) for 2 dims, total 4 dims input
class Q_torch(nn.Module):
    def __init__(self) -> None:
        super(Q_torch, self).__init__()
        self.fc0 = nn.Linear(4, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @classmethod
    def state_action_to_tensor(cls, state: GridWorldState, action: GridWorldAction) -> torch.Tensor:
        state_action_array = np.array([state.x, state.y, action.dx, action.dy], dtype=np.float32)
        return torch.tensor(state_action_array).unsqueeze(0)  # shape (1, 4)

    @classmethod
    def tensor_to_state_action(cls, tensor: torch.Tensor) -> Tuple[GridWorldState, GridWorldAction]:
        tensor = tensor.squeeze(0)  # shape (4,)
        state = GridWorldState(x=int(tensor[0]), y=int(tensor[1]))
        action = GridWorldAction(dx=int(tensor[2]), dy=int(tensor[3]))
        return state, action

class TorchValueFunctionTabularGridWorldMDP(ValueFunctionTabularGridWorldMDP):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState, 
                 discount_factor: float=0.9,
                 learning_rate: float=0.1,
                 rng: np.random.Generator=np.random.default_rng(42)) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, learning_rate, rng)
    
    def Q_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        state_action_tensor = model.state_action_to_tensor(state, action)
        q_value = model.forward(state_action_tensor)
        return q_value
    
    def dQ_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        state_action_tensor = model.state_action_to_tensor(state, action)
        state_action_tensor.requires_grad_(True)
        q_value = model.forward(state_action_tensor)
        q_value.backward()
        return state_action_tensor.grad
    
    def get_optimal_action_q_vf(self, state: GridWorldState, model: Any) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = self.Q_vf(state, action, model).item()
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)  # Fallback, should not reach here
    
    def sarsa_vf(self, initial_state: GridWorldState, episode_count: int, episode_length: int, epsilon: float=0.1) -> None:
        Q_vf : Q_torch = Q_torch()
        Q_vf.train()
        optimizer = torch.optim.SGD(Q_vf.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="SARSA-VF", dynamic_ncols=True):
            state: GridWorldState = initial_state
            action: GridWorldAction = self.decide(state)

            for _ in range(episode_length):
                
                sarsa_tensor = Q_vf.state_action_to_tensor(state, action)

                optimizer.zero_grad()
                q_value = Q_vf(sarsa_tensor)
                
                next_state, reward = self.transition(state, action)
                next_action = self.decide(next_state)

                with torch.no_grad():
                    sa_next_tensor = Q_vf.state_action_to_tensor(next_state, next_action)
                    td_target = reward + self.discount_factor * Q_vf(sa_next_tensor).item()

                # Wt+1 = Wt + α [ R(s,a) + γ * Q(s',a';Wt) - Q(s,a;Wt) ] ∇_W Q(s,a;Wt)
                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    optimal_action = self.get_optimal_action_q_vf(state, Q_vf)
                        
                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, a)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state
                action = next_action
        
    def q_learning_on_policy_vf(self, initial_state: GridWorldState, episode_count: int, episode_length: int, epsilon: float=0.1) -> None:
        Q_vf : Q_torch = Q_torch()
        Q_vf.train()
        optimizer = torch.optim.SGD(Q_vf.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="Q-learning VF (on)", dynamic_ncols=True):
            state: GridWorldState = initial_state
            
            for _ in range(episode_length):
                action = self.decide(state)
                q_learning_tensor = Q_vf.state_action_to_tensor(state, action)

                optimizer.zero_grad()
                q_value = Q_vf(q_learning_tensor)

                next_state, reward = self.transition(state, action)

                with torch.no_grad():
                    # max_a' Q(s',a';Wt)
                    max_next_q = float('-inf')
                    for a in self.action_space.actions:
                        sa_next_tensor = Q_vf.state_action_to_tensor(next_state, a)
                        q_next_value = Q_vf(sa_next_tensor).item()
                        if q_next_value > max_next_q:
                            max_next_q = q_next_value
                    td_target = reward + self.discount_factor * max_next_q

                # Wt+1 = Wt + α [ R(s,a) + γ * max_a' Q(s',a';Wt) - Q(s,a;Wt) ] ∇_W Q(s,a;Wt)
                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    optimal_action = self.get_optimal_action_q_vf(state, Q_vf)
                        
                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (epsilon / len(self.action_space.actions))
                    else:
                        self.policy[(state, a)] = (epsilon / len(self.action_space.actions))
                        
                state = next_state
                
    

    def deep_q_learning(self, 
                        sample_list: List[List[Tuple[GridWorldState, GridWorldAction, float, GridWorldState]]], 
                        batch_size: int=32, 
                        epochs_per_sample: int=50, 
                        update_interval: int = 5) -> None:
        
        Q_vf : Q_torch = Q_torch()
        Q_vf_target : Q_torch = Q_torch()
        Q_vf.train()
        optimizer = torch.optim.SGD(Q_vf.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        Q_vf_target.load_state_dict(Q_vf.state_dict())
        
        count = 0
        for sample in tqdm(sample_list, desc="DQN samples", dynamic_ncols=True):
            for epoch in trange(epochs_per_sample, desc="DQN epochs", leave=False, dynamic_ncols=True):
                # mini-batch sampling
                batch_indices = self.rng.choice(len(sample), size=min(batch_size, len(sample)), replace=False)
                batch = [sample[i] for i in batch_indices]
                
                count += 1
                if count % update_interval == 0:
                    Q_vf_target.load_state_dict(Q_vf.state_dict())
                
                for (state, action, reward, next_state) in batch:
                    
                        
                    state_action_tensor = Q_vf.state_action_to_tensor(state, action)
                    
                    optimizer.zero_grad()
                    q_value = Q_vf(state_action_tensor)
                    
                    with torch.no_grad():
                        # max_a' Q(s',a';Wt_target)
                        max_next_q = float('-inf')
                        for a in self.action_space.actions:
                            sa_next_tensor = Q_vf_target.state_action_to_tensor(next_state, a)
                            q_next_value = Q_vf_target(sa_next_tensor).item()
                            if q_next_value > max_next_q:
                                max_next_q = q_next_value
                        td_target = reward + self.discount_factor * max_next_q
                        
                    # Wt+1 = Wt + α [ R(s,a) + γ * max_a' Q(s',a';Wt_target) - Q(s,a;Wt) ] ∇_W Q(s,a;Wt)
                    target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                    loss = loss_fn(q_value, target_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        optimal_action = self.get_optimal_action_q_vf(state, Q_vf)
                        
                    for a in self.action_space.actions:
                            self.policy[(state, a)] = 1.0 if a == optimal_action else 0.0
                    
                        
                
            
            
                
                
    
