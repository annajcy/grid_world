from email import policy
import numpy as np

from torch import nn
import torch

from grid_world import GridWorldMDP, GridWorldState, PolicyNet, RLGridWorldRenderer
from grid_world.mdp.ac_grid_world_mdp import ActorCriticGridWorldMDP, ValueNet
    
def show(renderer: RLGridWorldRenderer, mdp: GridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        mdp.step()
        
class SimplePolicyNet(PolicyNet):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 4)

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
    
class SimpleValueNet(ValueNet):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
    value_net = SimpleValueNet()
    p_net_seed = torch.Generator().manual_seed(42)
    v_net_seed = torch.Generator().manual_seed(21)
    policy_net.init(rng=p_net_seed)
    value_net.init(rng=v_net_seed)
    
    ac_rng = np.random.default_rng(21)
    ac_mdp = ActorCriticGridWorldMDP[SimplePolicyNet, SimpleValueNet](
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=ac_rng,
        policy_net=policy_net,
        value_net=value_net
    )
    
    ac_mdp.Q_actor_critic(
        episode_count=500,
        episode_length=100,
        lr_policy=0.0001,
        lr_action_value=0.0001
    )
    
    ac_state_values = ac_mdp.solve_Vs(steps=bellman_solve_steps)
    ac_renderer = RLGridWorldRenderer(
        grid_world_mdp=ac_mdp,
        caption='Actor-Critic Grid World',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    ac_renderer.update_state_values(ac_state_values)
    show(ac_renderer, ac_mdp)
    ac_renderer.close()
    
    
if __name__ == "__main__":
    main()
