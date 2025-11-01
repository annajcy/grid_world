from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import trange, tqdm

from .grid_world_mdp import GridWorldState, GridWorldAction
from .tab_grid_world_mdp import TabularGridWorldMDP

class TDTabularGridWorldMDP(TabularGridWorldMDP):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        policy: Optional[Dict[Tuple[GridWorldState, GridWorldAction], float]] = None,
        discount_factor: float = 0.9,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__(width, height, initial_state, goal_state, policy, discount_factor, rng)

    def td0(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
    ) -> None:
        V: Dict[GridWorldState, float] = {S: 0.0 for S in self.state_space.to_list()}

        for _ in trange(episode_count, desc="TD(0)", dynamic_ncols=True):
            state: GridWorldState = initial_state

            for _ in range(episode_length):
                action = self.decide(state)
                next_state, reward = self.transition(state, action)

                td_target = reward + self.discount_factor * V[next_state]
                V[state] = V[state] + learning_rate * (td_target - V[state])

                optimal_action = self.get_opt_action_from_Vs(state, V)

                for action in self.action_space.actions:
                    if action == optimal_action:
                        self.policy[(state, action)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, action)] = epsilon / len(self.action_space.actions)

                state = next_state

    def sarsa(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
    ) -> None:
        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0 for state in self.state_space.to_list() for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="SARSA", dynamic_ncols=True):
            state: GridWorldState = initial_state
            action: GridWorldAction = self.decide(state)

            for _ in range(episode_length):
                next_state, reward = self.transition(state, action)
                next_action = self.decide(next_state)

                td_target = reward + self.discount_factor * Q[(next_state, next_action)]
                Q[(state, action)] = Q[(state, action)] + learning_rate * (td_target - Q[(state, action)])

                optimal_action = self.get_opt_action_from_Qsa(state, Q)

                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, a)] = epsilon / len(self.action_space.actions)

                state = next_state
                action = next_action

    def expected_sarsa(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
    ) -> None:
        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0 for state in self.state_space.to_list() for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="Expected SARSA", dynamic_ncols=True):
            state: GridWorldState = initial_state

            for _ in range(episode_length):
                action: GridWorldAction = self.decide(state)
                next_state, reward = self.transition(state, action)

                expected_q = 0.0

                for a in self.action_space.actions:
                    action_prob = self.policy[(next_state, a)]
                    expected_q += action_prob * Q[(next_state, a)]

                td_target = reward + self.discount_factor * expected_q
                Q[(state, action)] = Q[(state, action)] + learning_rate * (td_target - Q[(state, action)])

                optimal_action = self.get_opt_action_from_Qsa(state, Q)

                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, a)] = epsilon / len(self.action_space.actions)

                state = next_state

    def q_learning_on_policy(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
    ) -> None:
        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0 for state in self.state_space.to_list() for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="Q-learning (on)", dynamic_ncols=True):
            state: GridWorldState = initial_state

            for _ in range(episode_length):
                action: GridWorldAction = self.decide(state)
                next_state, reward = self.transition(state, action)

                max_next_q = max(Q[(next_state, a)] for a in self.action_space.actions)
                td_target = reward + self.discount_factor * max_next_q
                Q[(state, action)] = Q[(state, action)] + learning_rate * (td_target - Q[(state, action)])

                optimal_action = self.get_opt_action_from_Qsa(state, Q)

                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, a)] = epsilon / len(self.action_space.actions)

                state = next_state

    def q_learning_off_policy(
        self,
        sample_list: List[List[Tuple[GridWorldState, GridWorldAction, float]]],
        learning_rate: float = 0.1,
    ) -> None:
        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0 for state in self.state_space.to_list() for action in self.action_space.actions
        }

        for episode in tqdm(sample_list, desc="Q-learning (off) episodes", dynamic_ncols=True):
            for t in range(len(episode) - 1):
                (state, action, reward) = episode[t]

                next_state = episode[t + 1][0]
                max_next_q = max(Q[(next_state, a)] for a in self.action_space.actions)
                td_target = reward + self.discount_factor * max_next_q
                Q[(state, action)] = Q[(state, action)] + learning_rate * (td_target - Q[(state, action)])

                optimal_action = self.get_opt_action_from_Qsa(state, Q)

                for a in self.action_space.actions:
                    self.policy[(state, a)] = 1.0 if a == optimal_action else 0.0
