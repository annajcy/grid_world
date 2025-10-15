from grid_world import TabularGridWorldMDP, GridWorldState, TabularGridWorldRenderer, MCTabularGridWorldMDP

def show(renderer: TabularGridWorldRenderer, mdp: TabularGridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        res = mdp.step()
        # print(res)

def main():

    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(4, 3)
    discount_factor = 0.9
    bellman_solve_steps = 100
    
    mdp = MCTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor
    )

    mdp.monte_carlo_basic(iterations=100, episode_count=10, episode_length=20)
    state_values = mdp.solve_state_value(steps=bellman_solve_steps)
    renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=mdp,
        caption='Tabular Grid World - Monte Carlo Basic',
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
