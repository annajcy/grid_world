# Code for the Grid-World Environment 

## Overview

We added the code for the grid-world environment in my book. Interested readers can develop and test their own algorithms in this environment. Both Python and MATLAB versions are provided.

Please note that we do not provide the code of all the algorithms involved in the book. That is because they are the homework for the students in offline teaching: the students need to develop their own algorithms using the provided environment. Nevertheless, there are third-party implementations of some algorithms. Interested readers can check the links on the home page of the book.

I need to thank my PhD students, Yize Mi and Jianan Li, who are also the Teaching Assistants of my offline teaching. They contributed greatly to the code.

You are welcome to provide any feedback about the code such as bugs if detected.

----

## Python Version

### Requirements

- We support Python 3.7, 3.8, 3.9,  3.10 and 3.11. Make sure the following packages are installed: `numpy` and `matplotlib`.


### How to Run the Default Example

To run the example, please follow the procedures :

1. Change the directory to the file `examples/`

```bash
cd examples
```

2. Run the script:

```bash
python example_grid_world.py
```

You will see an animation as shown below:

- The blue star denotes the agent's current position within the grid world.
- The arrows on each grid illustrate the policy for that state. 
- The green line traces the agent's historical trajectory. 
- Obstacles are marked with yellow grids. 
- The target state is indicated by a blue grid. 
- The numerical values displayed on each grid represent the state values, which are initially generated as random numbers between 0 and 10. You may need to design your own algorithms to calculate these state values later on. 
- The horizontal number list above the grid world represents the horizontal coordinates (x-axis) of each grid.
- The vertical number list on the left side represents their vertical coordinates (y-axis).

![](plots/sample4.png)

### Customize the Parameters of the Grid World Environment

If you would like to customize your own grid world environment, please open `examples/arguments.py` and then change the following arguments:

"**env-size**", "**start-state**", "**target-state**", "**forbidden-states**", "**reward-target**", "**reward-forbidden**", "**reward-step**":

- "env-size" is represented as a tuple, where the first element represents the column index (horizontal coordinate), and the second element represents the row index (vertical coordinate).

- "start-state" denotes where the agent starts.

- "target-state" denotes the position of the target. 

- "forbidden-states" denotes the positions of obstacles. 

- "reward-target", "reward-forbidden" and "reward-step" represent the reward when reaching the target, the reward when entering a forbidden area, and the reward for each step, respectively.  

An example is shown below:

To specify the target state, modify the default value in the following sentence:

```python
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4,4))
```

Please note that the coordinate system used for all states within the environment—such as the start state, target state, and forbidden states—adheres to the conventional Python setup. In this system, the point `(0, 0)` is commonly designated as the origin of coordinates.



If you want to save figures in each step, please modify the "debug" argument to  "True":

```bash
parser.add_argument("--debug", type=bool, default=True)
```



### Create an Instance

If you would like to use the grid world environment to test your own RL algorithms, it is necessary to create an instance. The procedure for creating an instance and interacting with it can be found in `examples/example_grid_world.py`:

```python
from src.grid_world import GridWorld

 	env = GridWorld()
    state = env.reset()               
    for t in range(20):
        env.render()
        action = np.random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, Next state: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

```

![](plots/sample1.png)

- The policy is constructed as a matrix form shown below, which can be designed to be deterministic or stochastic. The example is a stochastic version:


 ```python
     # Add policy
     policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                       
     policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis] 
 ```

- Moreover, to change the shape of the arrows, you can open `src/grid_world.py`:


 ```python
self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))   
 ```



![](plots/sample2.png)

-  To add state value to each grid:


```python
values = np.random.uniform(0,10,(env.num_states,))
env.add_state_values(values)
```

![](plots/sample3.png)

- To render the environment:


```python
env.render(animation_interval=3)    # the figure will stop for 3 seconds
```