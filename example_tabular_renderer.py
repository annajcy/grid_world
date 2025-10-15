from grid_world import TabularGridWorldMDP, GridWorldState, TabularGridWorldRenderer

def show(renderer: TabularGridWorldRenderer, mdp: TabularGridWorldMDP):
    while renderer.running:
        renderer.handle_events()
        renderer.render(fps=30)
        mdp.step() 
    renderer.close()

def main():

    width, height = 5, 4
    initial_state = GridWorldState(0, 0)
    goal_state = GridWorldState(4, 3)
    
    mdp = TabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=0.9
    )
    
    renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=mdp,
        caption='Tabular Grid World - Policy Iteration',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    
    mdp.policy_iteration(threshold=1e-4, solve_state_value_steps=100)
    mdp.value_iteration(threshold=1e-4)
    
    state_values = mdp.solve_state_value(steps=1000)
    renderer.update_state_values(state_values)
    show(renderer, mdp)

if __name__ == "__main__":
    main()
