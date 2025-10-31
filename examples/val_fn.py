import numpy as np

from grid_world import TabularGridWorldMDP, GridWorldState, TabularGridWorldRenderer, TorchValueFunctionTabularGridWorldMDP

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
    
    # vf_rng = np.random.default_rng(21)
    # vf_mdp = TorchValueFunctionTabularGridWorldMDP(
    #     width=width,
    #     height=height,
    #     initial_state=initial_state,
    #     goal_state=goal_state,
    #     discount_factor=discount_factor,
    #     learning_rate=0.005,
    #     rng=vf_rng
    # )

    # #vf_mdp.sarsa_vf(initial_state=GridWorldState(0, 0), episode_count=2000, episode_length=50, epsilon=0.1)
    # vf_mdp.q_learning_on_policy_vf(initial_state=GridWorldState(0, 0), episode_count=2000, episode_length=50, epsilon=0.1)
    
    # vf_renderer = TabularGridWorldRenderer(
    #     tabular_gw_mdp=vf_mdp,
    #     caption='Tabular Grid World - SARSA with Value Function Approximation',
    #     screen_width=800,
    #     screen_height=600,
    #     show_policy=True,
    #     show_values=True
    # )
    # vf_state_values = vf_mdp.solve_state_value(steps=bellman_solve_steps)
    # vf_renderer.update_state_values(vf_state_values)
    # show(vf_renderer, vf_mdp)
    # vf_renderer.close()

    dqn_rng = np.random.default_rng(21)

    dqn_mdp = TorchValueFunctionTabularGridWorldMDP(
        width=width,
        height=height,
        initial_state=initial_state,
        goal_state=goal_state,
        discount_factor=discount_factor,
        learning_rate=0.005,
        rng=dqn_rng
    )
    
    dqn_mdp.deep_q_learning(
        sample_list=[dqn_mdp.sample_episode_sars(
            state=GridWorldState(0, 0),
            action=dqn_mdp.decide(state=GridWorldState(0, 0)),
            episode_length=50
        ) for _ in range(100)],
        batch_size=32,
        epochs_per_sample=50,
        update_interval=10
    )

    dqn_state_values = dqn_mdp.solve_state_value(steps=bellman_solve_steps)
    dqn_renderer = TabularGridWorldRenderer(
        tabular_gw_mdp=dqn_mdp,
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
