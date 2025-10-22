from typing import Set, Tuple, Dict, List
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
    
    def sample_episode(self, state: GridWorldState, action: GridWorldAction, episode_length: int=100) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
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

    def sample_episode_all(self, state: GridWorldState, action: GridWorldAction, max_episode_length: int=10000) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
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
        
        for _ in range(iterations):
            for state in self.state_space.to_list():
                # sample episodes to update Q
                for action in self.action_space.actions:
                    G_total = 0.0
                    for _ in range(episode_count):
                        episode = self.sample_episode(state, action, episode_length)
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
        
        for _ in range(episode_count):
            # episode generation from all state-action pairs
            episode = self.sample_episode(self.state_space.sample(self.rng), self.action_space.sample(self.rng), episode_length)
            
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
        
        for _ in range(episode_count):
            state: GridWorldState = initial_state
            
            for _ in range(episode_length):
                if state == self.goal_state:
                    break
                
                action = self.decide(state)
                next_state, reward = self.transition(state, action)
                
                # TD(0) update
                V[state] = V[state] + self.learning_rate * (reward + self.discount_factor * V[next_state] - V[state])  # V(s) += α [ R(s,a) + γ * V(s') - V(s) ]
                
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

        for _ in range(episode_count):
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

        for _ in range(episode_count):
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

        for _ in range(episode_count):
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

        for episode in sample_list:
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