from typing import Generic, TypeVar
import numpy as np
from tqdm.auto import trange

from .grid_world_mdp import GridWorldState, GridWorldAction
from .pg_grid_world_mdp import PolicyGradientGridWorldMDP, PolicyNetType

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def state_to_input_tensor(cls, state: GridWorldState) -> torch.Tensor:
        state_array = np.array([state.x, state.y], dtype=np.float32)
        return torch.tensor(state_array).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


ValueNetType = TypeVar('ValueNetType', bound=ValueNet)

class ActorCriticGridWorldMDP(
    PolicyGradientGridWorldMDP[PolicyNetType],
    Generic[PolicyNetType, ValueNetType]
):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: GridWorldState,
        goal_state: GridWorldState,
        policy_net : PolicyNetType,
        value_net : ValueNetType,
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
        self.value_net = value_net
        self.initial_value_weights = value_net.state_dict().copy()
        
    def initialize(self) -> None:
        super().initialize()
        self.value_net.load_state_dict(self.initial_value_weights)
    
    def advantage_actor_critic(self, 
                       episode_count: int, 
                       episode_length: int, 
                       lr_policy: float = 0.01, 
                       lr_action_value: float = 0.01):
        """
        Advantage Actor-Critic (A2C) / TD Actor-Critic
        
        Actor: 更新策略网络 π(a|s, θ)，使用策略梯度
        Critic: 更新价值网络 v(s, w)，使用 TD 误差
        
        Args:
            initial_state: 初始状态
            episode_count: 训练的 episode 数量
            episode_length: 每个 episode 的最大步数
            lr_policy: Actor (策略网络) 的学习率 (α_θ)
            lr_action_value: Critic (价值网络) 的学习率 (α_w)
        """
        optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=lr_action_value)
        
        self.policy.train()
        self.value_net.train()
        
        for _ in trange(episode_count, desc="Advantage Actor-Critic", dynamic_ncols=True):
            state: GridWorldState = self.state_space.sample(self.rng)

            for _ in range(episode_length):
                action: GridWorldAction = self.decide(state)
                next_state, reward = self.transition(state, action)
                
                state_tensor_value = self.value_net.state_to_input_tensor(state)
                value_state = self.value_net(state_tensor_value)
                reward_tensor = torch.tensor([[reward]], dtype=value_state.dtype, device=value_state.device)
                
                with torch.no_grad():
                    next_state_tensor = self.value_net.state_to_input_tensor(next_state)
                    value_next = self.value_net(next_state_tensor)
                    td_target = reward_tensor + self.discount_factor * value_next
                
                optimizer_value.zero_grad()
                td_error = td_target - value_state
                critic_loss = td_error.pow(2).mean()
                critic_loss.backward()
                optimizer_value.step()
                
                optimizer_policy.zero_grad()
                state_tensor_policy = self.policy.state_to_input_tensor(state)
                logits = self.policy(state_tensor_policy)
                action_index = self.policy.action_to_action_index(action)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_prob_action = log_probs[0, action_index]
                
                advantage = td_error.detach()
                actor_loss = -log_prob_action * advantage
                actor_loss.backward()
                optimizer_policy.step()
                
                state = next_state
