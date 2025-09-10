import sys
import grid_world as gw
import random
import numpy as np

def main():
    gw_state_space = gw.GridWorldStateSpace(width=5, height=5)
    gw_action_space = gw.GridWorldActionSpace()

    gw_mdp = gw.TabularGridWorldMDP(state_space=gw_state_space, 
                             action_space=gw_action_space, 
                             start_state=gw.GridWorldState(0, 0), 
                             goal_state=gw.GridWorldState(4, 4), 
                             forbiddens=[
                                 gw.GridWorldState(1, 3), 
                                 gw.GridWorldState(3, 4), 
                                 gw.GridWorldState(2, 1)
                             ])
    print("Initial State:", gw_mdp.current_state.to_list())
    print("State Space:", gw_state_space.to_list())
    print("Action Space:", gw_action_space.to_list())
    print("Goal State:", gw_mdp.goal_state.to_list())
    print("Forbiddens:", [s.to_list() for s in gw_mdp.forbiddens])
    print("Initial Transition Probabilities:", gw_mdp.policy)
    
    gw_renderer = gw.grid_world_renderer.GridWorldRenderer(gw_mdp)

    while gw_renderer.running:
        gw_renderer.handle_events()
        gw_renderer.render()
        
        action = gw_mdp.decide(gw_mdp.current_state)
        print("Current State:", gw_mdp.current_state.to_list())
        print("Action:", action.to_list())
        
        (next_state, reward, done, info) = gw_mdp.step(action)
        print("Next State:", next_state.to_list())
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("-----")
        
        if done:
            print("Reached the goal!")
            # break

    gw_renderer.close()

if __name__ == "__main__":
    main()
