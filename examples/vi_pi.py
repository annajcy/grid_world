from grid_world import TabularGridWorldMDP, GridWorldState, TabularGridWorldRenderer
import numpy as np

def show(renderer: TabularGridWorldRenderer, mdp: TabularGridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        print(mdp.step())

def main():
    
    rng = np.random.default_rng(21)
    
    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(4, 3)
    discount_factor = 0.9
    bellman_solve_steps = 100
    
    mdp = TabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=rng
    )
    
    state_values = mdp.solve_state_value(steps=bellman_solve_steps)
    renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=mdp,
        caption='Tabular Grid World - Original',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    renderer.update_state_values(state_values)
    show(renderer, mdp)
    renderer.close()
    
    mdp.initialize()
    mdp.policy_iteration(threshold=1e-4, solve_state_value_steps=bellman_solve_steps)
    state_values = mdp.solve_state_value(steps=bellman_solve_steps)
    renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=mdp,
        caption='Tabular Grid World - Policy iteration',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    renderer.update_state_values(state_values)
    show(renderer, mdp)
    renderer.close()
    
    mdp.initialize()
    mdp.value_iteration(threshold=1e-4)
    state_values = mdp.solve_state_value(steps=bellman_solve_steps)
    renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=mdp,
        caption='Tabular Grid World - Value iteration',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    renderer.update_state_values(state_values)
    show(renderer, mdp)
    renderer.close()
    

if __name__ == "__main__":
    main()
