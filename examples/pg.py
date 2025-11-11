import numpy as np
import torch
from torch import nn

from grid_world import GridWorldMDP, GridWorldState, RLGridWorldRenderer, PolicyGradientGridWorldMDP, PolicyNet

def show(renderer: RLGridWorldRenderer, mdp: GridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        mdp.step()
        
class SimplePolicyNet(PolicyNet):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
    
    def init(self, rng=torch.Generator().manual_seed(42)) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1, generator=rng)
                nn.init.zeros_(m.bias)

def main():

    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(3, 2)
    discount_factor = 0.9
    bellman_solve_steps = 100
    
    policy_net = SimplePolicyNet()
    p_net_seed = torch.Generator().manual_seed(42)
    policy_net.init(rng=p_net_seed)
    
    reinforce_rng = np.random.default_rng(21)
    reinforce_mdp = PolicyGradientGridWorldMDP[SimplePolicyNet](
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        policy_net=SimplePolicyNet(),
        discount_factor=discount_factor,
        rng=reinforce_rng
    )
    
    reinforce_mdp.reinforce(
        episode_count=2000,
        episode_length=50,
        learning_rate=0.001
    )
    
    reinforce_state_values = reinforce_mdp.solve_Vs(steps=bellman_solve_steps)
    reinforce_renderer = RLGridWorldRenderer(
        grid_world_mdp=reinforce_mdp,
        caption='Policy Gradient Grid World',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    reinforce_renderer.update_state_values(reinforce_state_values)
    show(reinforce_renderer, reinforce_mdp)
    reinforce_renderer.close()

if __name__ == "__main__":
    main()
