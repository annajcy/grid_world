from abc import abstractmethod
import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from rl.mdp import MDP, Action, ActionSpace, State, StateSpace
import numpy as np

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

    def sample(self, rng: np.random.Generator=np.random.default_rng()) -> GridWorldState:
        x = rng.choice(np.arange(self.width), 1)[0]
        y = rng.choice(np.arange(self.height), 1)[0]
        return GridWorldState(x, y)

    def contains(self, state: GridWorldState) -> bool:
        if not isinstance(state, GridWorldState):
            return False
        x, y = state.position()
        return 0 <= x < self.width and 0 <= y < self.height

    def to_list(self) -> List[GridWorldState]:
        return [GridWorldState(x, y) for y in range(self.height) for x in range(self.width)]

@dataclass(frozen=True, slots=True)
class GridWorldAction(Action):
    dx: int
    dy: int
        
    def to_list(self) -> List[int]:
        return [self.dx, self.dy]
    
    def __iter__(self) :
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
    def action_counts(cls) -> int:
        return 4

class GridWorldActionSpace(ActionSpace[GridWorldAction]):
    def __init__(self) -> None:
        self.actions = [
            GridWorldAction.left(),
            GridWorldAction.right(), 
            GridWorldAction.up(),
            GridWorldAction.down()
        ]

    def sample(self, rng: np.random.Generator=np.random.default_rng()) -> GridWorldAction:
        idx = rng.choice(np.arange(len(self.actions)), 1)[0]
        return self.actions[idx]

    def contains(self, action: GridWorldAction) -> bool:
        if not isinstance(action, GridWorldAction):
            return False
        return GridWorldAction(action.dx, action.dy) in self.actions
    
    def to_list(self) -> List[GridWorldAction]:
        return [action for action in self.actions]

class GridWorldMDP(MDP[GridWorldState, GridWorldStateSpace, GridWorldAction, GridWorldActionSpace]):
    def __init__(self, 
                 width: int,
                 height: int,
                 initial_state: GridWorldState, 
                 goal_state: GridWorldState,
                 discount_factor: float=0.9,
                 rng: np.random.Generator=np.random.default_rng()) -> None:
        super().__init__(GridWorldStateSpace(width, height), GridWorldActionSpace(), initial_state)
        self.goal_state = goal_state
        self.discount_factor = discount_factor
        self.rng = rng
        
    @abstractmethod
    def get_state_policy(self, state: GridWorldState) -> Dict[GridWorldAction, float]:
        pass

    def initialize(self) -> None:
        self.current_state = self.initial_state
        
    def transition(self, state: GridWorldState, action: GridWorldAction) -> Tuple[GridWorldState, float]:
        
        def reward(next_state: GridWorldState) -> float:
            if next_state == self.goal_state:
                return 1.0
            elif next_state == self.initial_state:
                return -1.0
            else:
                return -0.1
        
        (x, y) = state.position()
        (dx, dy) = action
        new_state = GridWorldState(x + dx, y + dy)
        if not self.state_space.contains(new_state):
            new_state = state  # stay in the same state if out of bounds

        return (new_state, reward(new_state))

    def is_terminated(self, state: GridWorldState) -> bool:
        return False # continuing task
    
    def is_truncated(self, state: GridWorldState) -> bool:
        return False # continuing task
        
    def decide(self, state: GridWorldState) -> GridWorldAction:
        action_probs = self.get_state_policy(state)
        rand_val = self.rng.uniform(0, 1)
        cumulative = 0.0
        for action, prob in action_probs.items():
            cumulative += prob
            if rand_val < cumulative:
                return action
        return self.action_space.sample(self.rng)  # Fallback, should not reach here if probs sum to 1
    
    def solve_Vs(self, steps: int=100) -> Dict[GridWorldState, float]:
        state_value: Dict[GridWorldState, float] = { state: 0.0 for state in self.state_space.to_list() }
        for _ in range(steps):
            new_state_value = state_value.copy()
            for state in self.state_space.to_list():
                value = 0.0
                state_policy = self.get_state_policy(state)
                for action in self.action_space.actions:
                    (next_state, reward) = self.transition(state, action)
                    action_prob = state_policy[action]
                    value += action_prob * (reward + self.discount_factor * state_value[next_state])  # v(s) = Σ_a π(a|s) [ R(s,a) + γ * v(s') ]
                new_state_value[state] = value
            state_value = new_state_value
        return state_value
    
    def solve_Qsa(self, steps: int=100) -> Dict[Tuple[GridWorldState, GridWorldAction], float]:
        q_value: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list()
            for action in self.action_space.actions
        }
        for _ in range(steps):
            new_q_value = q_value.copy()
            for state in self.state_space.to_list():
                for action in self.action_space.actions:
                    (next_state, reward) = self.transition(state, action)
                    next_state_policy = self.get_state_policy(next_state)
                    expected_next_q = 0.0
                    for next_action in self.action_space.actions:
                        action_prob = next_state_policy[next_action]
                        expected_next_q += action_prob * q_value[(next_state, next_action)]
                    new_q_value[(state, action)] = reward + self.discount_factor * expected_next_q  # Q(s,a) = R(s,a) + γ * Σ_a' π(a'|s') * Q(s',a')
            q_value = new_q_value
        return q_value
    
    def get_qsa_from_Vs(self, state: GridWorldState, action: GridWorldAction, state_value_table: Dict[GridWorldState, float]) -> float:
        (next_state, reward) = self.transition(state, action)
        return reward + self.discount_factor * state_value_table[next_state]  # Q(s,a) = R(s,a) + γ * v(s')
    
    def get_vs_from_Qsa(self, state: GridWorldState, q_value_table: Dict[Tuple[GridWorldState, GridWorldAction], float]) -> float:
        v = 0.0
        state_policy = self.get_state_policy(state)
        for action in self.action_space.actions:
            action_prob = state_policy[action]
            v += action_prob * q_value_table[(state, action)]  # v(s) = Σ_a π(a|s) * Q(s,a)
        return v
    
    def get_opt_action_from_Vs(self, state: GridWorldState, V: Dict[GridWorldState, float]) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = self.get_qsa_from_Vs(state, action, V)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)  # Fallback, should not reach here
    
    def get_opt_action_from_Qsa(self, state: GridWorldState, Q: Dict[Tuple[GridWorldState, GridWorldAction], float]) -> GridWorldAction:
        best_action = None
        best_value = float('-inf')
        for action in self.action_space.actions:
            q_value = Q[(state, action)]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)  # Fallback, should not reach here
    
    def get_qsa_from_sar(self, episode: List[Tuple[GridWorldState, GridWorldAction, float]]) -> Tuple[Tuple[GridWorldState, GridWorldAction], float]:
        G = 0.0
        for i in range(len(episode)):
            G = episode[len(episode) - 1 - i][2] + self.discount_factor * G  # G_t = R_{t+1} + γ * G_{t+1}
        return (episode[0][0], episode[0][1]), G
    
    def sample_sar(self, state: GridWorldState, action: GridWorldAction, episode_length: int=100) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
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

    def sample_sar_all(self, state: GridWorldState, action: GridWorldAction, max_episode_length: int=10000) -> List[Tuple[GridWorldState, GridWorldAction, float]]:
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
    
    def sample_sars(self, state: GridWorldState, action: GridWorldAction, episode_length: int=100) -> List[Tuple[GridWorldState, GridWorldAction, float, GridWorldState]]:
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
    
    
    
    
