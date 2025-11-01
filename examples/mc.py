import numpy as np
from grid_world import TabularGridWorldMDP, GridWorldState, RLGridWorldRenderer, MCTabularGridWorldMDP

def show(renderer: RLGridWorldRenderer, mdp: TabularGridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        mdp.step()

def main():

    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(3, 2)
    discount_factor = 0.9
    bellman_solve_steps = 100

    basic_rng = np.random.default_rng(21)
    basic_mdp = MCTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=basic_rng
    )
    
    basic_mdp.mc_basic(iterations=50, episode_count=3, episode_length=8)
    basic_state_values = basic_mdp.solve_Vs(steps=bellman_solve_steps)
    basic_renderer = RLGridWorldRenderer(
        grid_world_mdp=basic_mdp,
        caption='Tabular Grid World - Monte Carlo Basic',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    basic_renderer.update_state_values(basic_state_values)
    show(basic_renderer, basic_mdp)
    basic_renderer.close()

    greedy_rng = np.random.default_rng(21)
    greedy_mdp = MCTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=greedy_rng
    )
    greedy_mdp.mc_epsilon_greedy(episode_count=300, episode_length=50, epsilon=0.05)
    greedy_state_values = greedy_mdp.solve_Vs(steps=bellman_solve_steps)
    greedy_renderer = RLGridWorldRenderer(
        grid_world_mdp=greedy_mdp,
        caption='Tabular Grid World - Monte Carlo Epsilon Greedy',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    greedy_renderer.update_state_values(greedy_state_values)
    show(greedy_renderer, greedy_mdp)
    greedy_renderer.close()

    for s in greedy_state_values.keys():
        basic_value = basic_state_values[s]
        greedy_value = greedy_state_values[s]
        assert abs(basic_value - greedy_value) < 0.3
    print("Monte Carlo Epsilon Greedy successfully approximates the state values.")

if __name__ == "__main__":
    main()
