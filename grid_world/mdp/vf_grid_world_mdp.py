from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange, tqdm
from .tab_grid_world_mdp import TabularGridWorldMDP
from .grid_world_mdp import GridWorldState, GridWorldAction

class QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def state_action_to_input_tensor(cls, state: GridWorldState, action: GridWorldAction) -> torch.Tensor:
        state_action_array = np.array([state.x, state.y, action.dx, action.dy], dtype=np.float32)
        return torch.tensor(state_action_array).unsqueeze(0)

    @classmethod
    def input_tensor_to_state_action(cls, tensor: torch.Tensor) -> Tuple[GridWorldState, GridWorldAction]:
        tensor = tensor.squeeze(0)
        state = GridWorldState(x=int(tensor[0]), y=int(tensor[1]))
        action = GridWorldAction(dx=int(tensor[2]), dy=int(tensor[3]))
        return state, action

QNetType = TypeVar('QNetType', bound=QNet)

class ValueFunctionTabularGridWorldMDP(TabularGridWorldMDP, Generic[QNetType]):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        q_net : QNetType,
        policy: Optional[Dict[Tuple[GridWorldState, GridWorldAction], float]] = None,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        rng: np.random.Generator = np.random.default_rng(42),
    ) -> None:
        super().__init__(width, height, initial_state, goal_state, policy, discount_factor, rng)
        self.learning_rate = learning_rate
        self.model = q_net

    def Qsa_vf(self, state: GridWorldState, action: GridWorldAction) -> Any:
        state_action_tensor = self.model.state_action_to_input_tensor(state, action)
        return self.model(state_action_tensor)

    def dQsa_vf(self, state: GridWorldState, action: GridWorldAction) -> Any:
        state_action_tensor = self.model.state_action_to_input_tensor(state, action)
        state_action_tensor.requires_grad_(True)
        q_value = self.model(state_action_tensor)
        q_value.backward()
        return state_action_tensor.grad
    
    def get_opt_a_from_Qsa_vf(self, state: GridWorldState) -> GridWorldAction:
        best_action = None
        best_value = float("-inf")
        for action in self.action_space.actions:
            q_value = self.Qsa_vf(state, action).item()
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action if best_action is not None else self.action_space.sample(self.rng)

    def sarsa_vf(
        self,
        initial_state: GridWorldState,
        episode_count: int,
        episode_length: int,
        epsilon: float = 0.1,
    ) -> None:
        
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="SARSA-VF", dynamic_ncols=True):
            state: GridWorldState = initial_state
            action: GridWorldAction = self.decide(state)

            for _ in range(episode_length):
                sarsa_tensor = self.model.state_action_to_input_tensor(state, action)

                optimizer.zero_grad()
                q_value = self.model(sarsa_tensor)

                next_state, reward = self.transition(state, action)
                next_action = self.decide(next_state)

                with torch.no_grad():
                    sa_next_tensor = self.model.state_action_to_input_tensor(next_state, next_action)
                    td_target = reward + self.discount_factor * self.model(sa_next_tensor).item()

                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    optimal_action = self.get_opt_a_from_Qsa_vf(state)

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

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in trange(episode_count, desc="Q-learning VF (on)", dynamic_ncols=True):
            state: GridWorldState = initial_state

            for _ in range(episode_length):
                action = self.decide(state)
                q_learning_tensor = self.model.state_action_to_input_tensor(state, action)

                optimizer.zero_grad()
                q_value = self.model(q_learning_tensor)

                next_state, reward = self.transition(state, action)

                with torch.no_grad():
                    max_next_q = float("-inf")
                    for a in self.action_space.actions:
                        sa_next_tensor = self.model.state_action_to_input_tensor(next_state, a)
                        q_next_value = self.model(sa_next_tensor).item()
                        if q_next_value > max_next_q:
                            max_next_q = q_next_value
                    td_target = reward + self.discount_factor * max_next_q

                target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                loss = loss_fn(q_value, target_tensor)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    optimal_action = self.get_opt_a_from_Qsa_vf(state)

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
        
        target_model: QNetType = self.model.__class__()  # Create a new instance of the same class
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        target_model.load_state_dict(self.model.state_dict())

        count = 0
        for sample in tqdm(sample_list, desc="DQN samples", dynamic_ncols=True):
            for _ in trange(epochs_per_sample, desc="DQN epochs", leave=False, dynamic_ncols=True):
                batch_indices = self.rng.choice(len(sample), size=min(batch_size, len(sample)), replace=False)
                batch = [sample[i] for i in batch_indices]

                count += 1
                if count % update_interval == 0:
                    target_model.load_state_dict(self.model.state_dict())

                for (state, action, reward, next_state) in batch:
                    state_action_tensor = self.model.state_action_to_input_tensor(state, action)

                    optimizer.zero_grad()
                    q_value = self.model(state_action_tensor)

                    with torch.no_grad():
                        max_next_q = float("-inf")
                        for a in self.action_space.actions:
                            sa_next_tensor = target_model.state_action_to_input_tensor(next_state, a)
                            q_next_value = target_model(sa_next_tensor).item()
                            if q_next_value > max_next_q:
                                max_next_q = q_next_value
                        td_target = reward + self.discount_factor * max_next_q

                    target_tensor = torch.tensor([[td_target]], dtype=q_value.dtype, device=q_value.device)
                    loss = loss_fn(q_value, target_tensor)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        optimal_action = self.get_opt_a_from_Qsa_vf(state)

                    for a in self.action_space.actions:
                        self.policy[(state, a)] = 1.0 if a == optimal_action else 0.0
