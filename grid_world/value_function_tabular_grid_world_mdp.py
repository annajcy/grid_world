from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange, tqdm

from .grid_world_mdp import GridWorldState, GridWorldAction
from .td_tabular_grid_world_mdp import TDTabularGridWorldMDP


class ValueFunctionTabularGridWorldMDP(TDTabularGridWorldMDP):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, learning_rate, rng)

    @abstractmethod
    def Q_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def dQ_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        raise NotImplementedError

    def get_optimal_action_q_vf(self, state: GridWorldState, model: Any) -> GridWorldAction:
        best_action = None
        best_value = float("-inf")
        for action in self.action_space.actions:
            q_value = self.Q_vf(state, action, model).item()
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)


class Q_torch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        return torch.tensor(state_action_array).unsqueeze(0)

    @classmethod
    def tensor_to_state_action(cls, tensor: torch.Tensor) -> Tuple[GridWorldState, GridWorldAction]:
        tensor = tensor.squeeze(0)
        state = GridWorldState(x=int(tensor[0]), y=int(tensor[1]))
        action = GridWorldAction(dx=int(tensor[2]), dy=int(tensor[3]))
        return state, action


class TorchValueFunctionTabularGridWorldMDP(ValueFunctionTabularGridWorldMDP):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__(width, height, initial_state, goal_state, discount_factor, learning_rate, rng)

    def Q_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        state_action_tensor = model.state_action_to_tensor(state, action)
        return model.forward(state_action_tensor)

    def dQ_vf(self, state: GridWorldState, action: GridWorldAction, model: Any) -> Any:
        state_action_tensor = model.state_action_to_tensor(state, action)
        state_action_tensor.requires_grad_(True)
        q_value = model.forward(state_action_tensor)
        q_value.backward()
        return state_action_tensor.grad

    def sarsa_vf(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
    ) -> None:
        model: Q_torch = Q_torch()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="SARSA-VF", dynamic_ncols=True):
            state: GridWorldState = initial_state
            action: GridWorldAction = self.decide(state)

            for _ in range(episode_length):
                sarsa_tensor = model.state_action_to_tensor(state, action)

                optimizer.zero_grad()
                q_value = model(sarsa_tensor)

                next_state, reward = self.transition(state, action)
                next_action = self.decide(next_state)

                with torch.no_grad():
                    sa_next_tensor = model.state_action_to_tensor(next_state, next_action)
                    td_target = reward + self.discount_factor * model(sa_next_tensor).item()

                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    optimal_action = self.get_optimal_action_q_vf(state, model)

                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, a)] = epsilon / len(self.action_space.actions)

                state = next_state
                action = next_action

    def q_learning_on_policy_vf(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
    ) -> None:
        model: Q_torch = Q_torch()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="Q-learning VF (on)", dynamic_ncols=True):
            state: GridWorldState = initial_state

            for _ in range(episode_length):
                action = self.decide(state)
                q_learning_tensor = model.state_action_to_tensor(state, action)

                optimizer.zero_grad()
                q_value = model(q_learning_tensor)

                next_state, reward = self.transition(state, action)

                with torch.no_grad():
                    max_next_q = float("-inf")
                    for a in self.action_space.actions:
                        sa_next_tensor = model.state_action_to_tensor(next_state, a)
                        q_next_value = model(sa_next_tensor).item()
                        if q_next_value > max_next_q:
                            max_next_q = q_next_value
                    td_target = reward + self.discount_factor * max_next_q

                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    optimal_action = self.get_optimal_action_q_vf(state, model)

                for a in self.action_space.actions:
                    if a == optimal_action:
                        self.policy[(state, a)] = 1 - (len(self.action_space.actions) - 1) * (
                            epsilon / len(self.action_space.actions)
                        )
                    else:
                        self.policy[(state, a)] = epsilon / len(self.action_space.actions)

                state = next_state

    def deep_q_learning(
        self,
        sample_list: List[List[Tuple[GridWorldState, GridWorldAction, float, GridWorldState]]],
        batch_size: int = 32,
        epochs_per_sample: int = 50,
        update_interval: int = 5,
    ) -> None:
        model: Q_torch = Q_torch()
        target_model: Q_torch = Q_torch()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        target_model.load_state_dict(model.state_dict())

        count = 0
        for sample in tqdm(sample_list, desc="DQN samples", dynamic_ncols=True):
            for _ in trange(epochs_per_sample, desc="DQN epochs", leave=False, dynamic_ncols=True):
                batch_indices = self.rng.choice(len(sample), size=min(batch_size, len(sample)), replace=False)
                batch = [sample[i] for i in batch_indices]

                count += 1
                if count % update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

                for (state, action, reward, next_state) in batch:
                    state_action_tensor = model.state_action_to_tensor(state, action)

                    optimizer.zero_grad()
                    q_value = model(state_action_tensor)

                    with torch.no_grad():
                        max_next_q = float("-inf")
                        for a in self.action_space.actions:
                            sa_next_tensor = target_model.state_action_to_tensor(next_state, a)
                            q_next_value = target_model(sa_next_tensor).item()
                            if q_next_value > max_next_q:
                                max_next_q = q_next_value
                        td_target = reward + self.discount_factor * max_next_q

                    target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                    loss = loss_fn(q_value, target_tensor)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        optimal_action = self.get_optimal_action_q_vf(state, model)

                    for a in self.action_space.actions:
                        self.policy[(state, a)] = 1.0 if a == optimal_action else 0.0
