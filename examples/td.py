import numpy as np
from grid_world import TabularGridWorldMDP, GridWorldState, TabularGridWorldRenderer, TDTabularGridWorldMDP

def show(renderer: TabularGridWorldRenderer, mdp: TabularGridWorldMDP):
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

    td_rng = np.random.default_rng(21)
    td_mdp = TDTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=td_rng,
        learning_rate=0.1
    )

    td_mdp.td0(initial_state=GridWorldState(0, 0), episode_count=1000, episode_length=100)
    td_state_values = td_mdp.solve_state_value(steps=bellman_solve_steps)
    td_renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=td_mdp,
        caption='Tabular Grid World - TD(0)',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    td_renderer.update_state_values(td_state_values)
    show(td_renderer, td_mdp)
    td_renderer.close()

    sarsa_rng = np.random.default_rng(21)
    sarsa_mdp = TDTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=sarsa_rng
    )
    
    sarsa_mdp.sarsa(initial_state=GridWorldState(0, 0), episode_count=1000, episode_length=50)
    sarsa_state_values = sarsa_mdp.solve_state_value(steps=bellman_solve_steps)
    sarsa_renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=sarsa_mdp,
        caption='Tabular Grid World - SARSA',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    sarsa_renderer.update_state_values(sarsa_state_values)
    show(sarsa_renderer, sarsa_mdp)
    sarsa_renderer.close()
    
    ex_sarsa_rng = np.random.default_rng(21)
    ex_sarsa_mdp = TDTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=ex_sarsa_rng
    )

    ex_sarsa_mdp.sarsa(initial_state=GridWorldState(0, 0), episode_count=1000, episode_length=50)
    ex_sarsa_state_values = ex_sarsa_mdp.solve_state_value(steps=bellman_solve_steps)
    ex_sarsa_renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=ex_sarsa_mdp,
        caption='Tabular Grid World - expected SARSA',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    ex_sarsa_renderer.update_state_values(ex_sarsa_state_values)
    show(ex_sarsa_renderer, ex_sarsa_mdp)
    ex_sarsa_renderer.close()
    
    
    q_learning_rng = np.random.default_rng(21)
    q_learning_mdp = TDTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=q_learning_rng
    )

    q_learning_mdp.q_learning_on_policy(initial_state=GridWorldState(0, 0), episode_count=1000, episode_length=50)
    q_learning_state_values = q_learning_mdp.solve_state_value(steps=bellman_solve_steps)
    q_learning_renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=q_learning_mdp,
        caption='Tabular Grid World - Q-Learning on Policy ',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    q_learning_renderer.update_state_values(q_learning_state_values)
    show(q_learning_renderer, q_learning_mdp)
    q_learning_renderer.close()
    
    q_learning_rng_off = np.random.default_rng(21)
    q_learning_mdp_off = TDTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        rng=q_learning_rng_off
    )

    q_learning_mdp_off.q_learning_off_policy([q_learning_mdp_off.sample_episode_sar(
        state=GridWorldState(0, 0), 
        action=q_learning_mdp_off.decide(state=GridWorldState(0, 0)), 
        episode_length=50
    ) for _ in range(1000)])
    
    q_learning_state_values_off = q_learning_mdp_off.solve_state_value(steps=bellman_solve_steps)
    q_learning_renderer_off = TabularGridWorldRenderer(
        tabular_gw_mdp=q_learning_mdp_off,
        caption='Tabular Grid World - Q-Learning off Policy ',
        screen_width=800,
        screen_height=600,
        show_policy=True,
        show_values=True
    )
    q_learning_renderer_off.update_state_values(q_learning_state_values_off)
    show(q_learning_renderer_off, q_learning_mdp_off)
    q_learning_renderer_off.close()

if __name__ == "__main__":
    main()
