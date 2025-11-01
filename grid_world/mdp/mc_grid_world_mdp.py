from typing import Dict, Optional, Tuple

import numpy as np

from tqdm.auto import trange

from .grid_world_mdp import GridWorldState, GridWorldAction
from .tab_grid_world_mdp import TabularGridWorldMDP

class MCTabularGridWorldMDP(TabularGridWorldMDP):
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

    def mc_basic(self, iterations: int = 100, episode_count: int = 10, episode_length: int = 20) -> None:
        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list()
            for action in self.action_space.actions
        }

        for _ in trange(iterations, desc="MC-basic", dynamic_ncols=True):
            for state in self.state_space.to_list():
                for action in self.action_space.actions:
                    G_total = 0.0
                    for _ in range(episode_count):
                        episode = self.sample_sar(state, action, episode_length)
                        (s, a), G = self.get_qsa_from_sar(episode)
                        G_total += G
                    G_avg = G_total / episode_count if episode_count > 0 else 0.0

                    Q[(state, action)] = G_avg

                action_values = [Q[(state, action)] for action in self.action_space.actions]
                optimal_action = None
                max_action_value = max(action_values)
                for action, value in zip(self.action_space.actions, action_values):
                    if value == max_action_value:
                        optimal_action = action
                        break

                for action in self.action_space.actions:
                    self.policy[(state, action)] = 1.0 if action == optimal_action else 0.0

    def mc_epsilon_greedy(
        self,
        episode_count: int = 20,
        episode_length: int = 100,
        epsilon: float = 0.1,
        first_visit: bool = False,
    ) -> None:
        visit_count: Dict[Tuple[GridWorldState, GridWorldAction], int] = {
            (state, action): 0
            for state in self.state_space.to_list()
            for action in self.action_space.actions
        }

        G_sum: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list()
            for action in self.action_space.actions
        }

        Q: Dict[Tuple[GridWorldState, GridWorldAction], float] = {
            (state, action): 0.0
            for state in self.state_space.to_list()
            for action in self.action_space.actions
        }

        for _ in trange(episode_count, desc="MC-eps-greedy", dynamic_ncols=True):
            episode = self.sample_sar(
                self.state_space.sample(self.rng),
                self.action_space.sample(self.rng),
                episode_length,
            )

            G = 0.0

            for t in range(len(episode) - 1, -1, -1):
                (s, a, r) = episode[t]
                G = r + self.discount_factor * G

                if first_visit and (s, a) in [(x[0], x[1]) for x in episode[0:t]]:
                    continue

                G_sum[(s, a)] += G
                visit_count[(s, a)] += 1
                Q[(s, a)] = G_sum[(s, a)] / visit_count[(s, a)]

                action_values = [Q[(s, action)] for action in self.action_space.actions]
                optimal_action = None
                max_action_value = max(action_values)
                for action, value in zip(self.action_space.actions, action_values):
                    if value == max_action_value:
                        optimal_action = action
                        break

                for action in self.action_space.actions:
                    if action == optimal_action:
                        self.policy[(s, action)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(s, action)] = epsilon / len(self.action_space.actions)
