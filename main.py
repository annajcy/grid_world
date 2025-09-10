import grid_world as gw
import random
import numpy as np

def main():
    gw_state_space = gw.GridWorldStateSpace(width=5, height=5)
    gw_action_space = gw.GridWorldActionSpace()

    gw_mdp = gw.GridWorldMDP(state_space=gw_state_space, 
                             action_space=gw_action_space, 
                             start_state=gw.GridWorldState(0, 0), 
                             goal_state=gw.GridWorldState(4, 4), 
                             forbiddens=[
                                 gw.GridWorldState(1, 1), 
                                 gw.GridWorldState(1, 2), 
                                 gw.GridWorldState(2, 1)
                             ])
    print("Initial State:", gw_mdp.current_state.to_list())
    action = gw_mdp.decide(gw_mdp.current_state)
    print("Action:", action.to_list())
    (next_state, reward, done, info) = gw_mdp.step(action)
    print("Next State:", next_state.to_list())
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

if __name__ == "__main__":
    main()
