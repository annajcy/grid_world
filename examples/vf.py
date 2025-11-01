import numpy as np
from grid_world import TabularGridWorldMDP, GridWorldState, RLGridWorldRenderer, ValueFunctionTabularGridWorldMDP, QNet
from torch import nn
import torch

def show(renderer: RLGridWorldRenderer, mdp: TabularGridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        mdp.step()

class SimpleQNet(QNet):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(4, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(3, 2)
    discount_factor = 0.9
    bellman_solve_steps = 100
    
    sarsa_vf_rng = np.random.default_rng(21)
    sarsa_vf_mdp = ValueFunctionTabularGridWorldMDP[SimpleQNet](
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        q_net=SimpleQNet(),
        discount_factor=discount_factor,
        learning_rate=0.005,
        rng=sarsa_vf_rng
    )

    sarsa_vf_mdp.sarsa_vf(initial_state=GridWorldState(0, 0), episode_count=600, episode_length=50, epsilon=0.1)
    
    sarsa_vf_renderer = RLGridWorldRenderer(
        grid_world_mdp=sarsa_vf_mdp,
        caption='Tabular Grid World - SARSA with Value Function Approximation',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    sarsa_vf_state_values = sarsa_vf_mdp.solve_Vs(steps=bellman_solve_steps)
    sarsa_vf_renderer.update_state_values(sarsa_vf_state_values)
    show(sarsa_vf_renderer, sarsa_vf_mdp)
    sarsa_vf_renderer.close()
    
    q_learning_vf_rng = np.random.default_rng(21)
    q_learning_vf_mdp = ValueFunctionTabularGridWorldMDP[SimpleQNet](
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        q_net=SimpleQNet(),
        discount_factor=discount_factor,
        learning_rate=0.005,
        rng=q_learning_vf_rng
    )
    q_learning_vf_mdp.q_learning_on_policy_vf(initial_state=GridWorldState(0, 0), episode_count=600, episode_length=50, epsilon=0.1)
    q_learning_vf_renderer = RLGridWorldRenderer(
        grid_world_mdp=q_learning_vf_mdp,
        caption='Tabular Grid World - Q-Learning with Value Function Approximation',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    q_learning_vf_state_values = q_learning_vf_mdp.solve_Vs(steps=bellman_solve_steps)
    q_learning_vf_renderer.update_state_values(q_learning_vf_state_values)
    show(q_learning_vf_renderer, q_learning_vf_mdp)
    q_learning_vf_renderer.close()

    dqn_rng = np.random.default_rng(21)
    dqn_mdp = ValueFunctionTabularGridWorldMDP[SimpleQNet](
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        q_net=SimpleQNet(),
        discount_factor=discount_factor,
        learning_rate=0.005,
        rng=dqn_rng
    )
    
    dqn_mdp.deep_q_learning(
        sample_list=[dqn_mdp.sample_sars(
            state=GridWorldState(0, 0),
            action=dqn_mdp.decide(state=GridWorldState(0, 0)),
            episode_length=50
        ) for _ in range(100)],
        batch_size=16,
        epochs_per_sample=50,
        update_interval=10
    )

    dqn_state_values = dqn_mdp.solve_Vs(steps=bellman_solve_steps)
    dqn_renderer = RLGridWorldRenderer(
        grid_world_mdp=dqn_mdp,
        caption='Tabular Grid World - Deep Q-Learning',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    dqn_renderer.update_state_values(dqn_state_values)
    show(dqn_renderer, dqn_mdp)
    dqn_renderer.close()

if __name__ == "__main__":
    main()
