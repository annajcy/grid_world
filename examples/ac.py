import numpy as np

from grid_world import TabularGridWorldMDP, GridWorldState, RLGridWorldRenderer, ValueFunctionTabularGridWorldMDP

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
    
if __name__ == "__main__":
    main()
