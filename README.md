# Grid World

A comprehensive Grid World environment for reinforcement learning education and experimentation, featuring implementations of classic and modern RL algorithms with interactive visualization.

## 📖 Overview

Grid World is a Python-based reinforcement learning framework that provides:
- A customizable grid world environment built on a clean MDP (Markov Decision Process) abstraction
- Implementations of multiple RL algorithms from dynamic programming to deep reinforcement learning
- Interactive visualization using Pygame for real-time policy and value function rendering
- Educational examples demonstrating each algorithm's behavior

This project is ideal for learning, teaching, and experimenting with reinforcement learning concepts.

## ✨ Features

### Implemented Algorithms

**Dynamic Programming:**
- Value Iteration (VI)
- Policy Iteration (PI)

**Monte Carlo Methods:**
- Basic Monte Carlo
- Monte Carlo with Epsilon-Greedy Policy

**Temporal Difference Learning:**
- TD(0) - Temporal Difference Learning
- SARSA - On-policy TD Control
- Expected SARSA
- Q-Learning (On-policy and Off-policy)

**Function Approximation:**
- SARSA with Value Function Approximation
- Q-Learning with Value Function Approximation
- Deep Q-Network (DQN)

**Policy Gradient Methods:**
- REINFORCE Algorithm

**Actor-Critic Methods:**
- Advantage Actor-Critic (A2C)

### Key Features
- 🎮 Interactive visualization with Pygame
- 📊 Real-time display of state values and policies
- 🔧 Configurable grid size, initial/goal states, and hyperparameters
- 🎯 Clean abstraction layer separating MDP logic from rendering
- 🧪 Multiple example scripts demonstrating each algorithm

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- conda (recommended) or pip

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/annajcy/grid_world.git
cd grid_world
```

2. **Install uv package manager:**
```bash
conda install conda-forge::uv
```

3. **Sync dependencies and install package:**
```bash
uv sync 
uv pip install -e .
```

Alternatively, you can use pip directly:
```bash
pip install -e .
```

## 📚 Usage

### Running Examples

The `examples` folder contains demonstrations of all implemented algorithms. Each script showcases a different RL approach:

#### Value Iteration and Policy Iteration
```bash
python examples/vi_pi.py
```
Demonstrates dynamic programming methods for solving the MDP exactly.

#### Monte Carlo Methods
```bash
python examples/mc.py
```
Shows Monte Carlo Basic and Monte Carlo Epsilon-Greedy learning.

#### Temporal Difference Learning
```bash
python examples/td.py
```
Demonstrates TD(0), SARSA, Expected SARSA, Q-Learning (on-policy and off-policy).

#### Value Function Approximation
```bash
python examples/vf.py
```
Shows SARSA, Q-Learning, and Deep Q-Network (DQN) with neural network function approximation.

#### Policy Gradient
```bash
python examples/pg.py
```
Demonstrates the REINFORCE algorithm using a policy network.

#### Actor-Critic
```bash
python examples/ac.py
```
Shows the Advantage Actor-Critic (A2C) algorithm.

### Basic Usage Example

```python
import numpy as np
from grid_world import (
    GridWorldState, 
    TabularGridWorldMDP, 
    RLGridWorldRenderer
)

# Configure the grid world
width, height = 5, 4
initial_state = GridWorldState(0, 0)
goal_state = GridWorldState(4, 3)
discount_factor = 0.9

# Create MDP
rng = np.random.default_rng(42)
mdp = TabularGridWorldMDP(
    width=width,
    height=height,
    initial_state=initial_state,
    goal_state=goal_state,
    discount_factor=discount_factor,
    rng=rng
)

# Solve using value iteration
mdp.value_iteration(threshold=1e-4)
state_values = mdp.solve_Vs(steps=100)

# Visualize results
renderer = RLGridWorldRenderer(
    grid_world_mdp=mdp,
    caption='Grid World - Value Iteration',
    screen_width=800,
    screen_height=600,
    show_policy=True,
    show_values=True
)
renderer.update_state_values(state_values)

# Interactive rendering loop
while renderer.running:
    renderer.handle_events()
    renderer.render(fps=30)
    mdp.step()

renderer.close()
```

## 🏗️ Project Structure

```
grid_world/
├── examples/           # Example scripts for each algorithm
│   ├── vi_pi.py       # Value & Policy Iteration
│   ├── mc.py          # Monte Carlo methods
│   ├── td.py          # Temporal Difference methods
│   ├── vf.py          # Value Function Approximation
│   ├── pg.py          # Policy Gradient (REINFORCE)
│   └── ac.py          # Actor-Critic
├── grid_world/        # Main package
│   ├── mdp/           # MDP implementations
│   │   ├── grid_world_mdp.py          # Base grid world MDP
│   │   ├── tab_grid_world_mdp.py      # Tabular methods (VI, PI)
│   │   ├── mc_grid_world_mdp.py       # Monte Carlo methods
│   │   ├── td_grid_world_mdp.py       # TD methods
│   │   ├── vf_grid_world_mdp.py       # Value function approximation
│   │   ├── pg_grid_world_mdp.py       # Policy gradient
│   │   └── ac_grid_world_mdp.py       # Actor-Critic
│   └── renderer/      # Visualization components
│       ├── grid_world_renderer.py     # Base renderer
│       └── rl_grid_world_renderer.py  # RL-specific renderer
├── rl/                # RL framework abstractions
│   ├── mdp.py         # MDP base classes
│   └── renderer.py    # Renderer base classes
└── pyproject.toml     # Project configuration
```

## 🎓 Algorithm Descriptions

### Dynamic Programming
- **Value Iteration**: Iteratively computes optimal state values by applying the Bellman optimality equation until convergence.
- **Policy Iteration**: Alternates between policy evaluation (computing values for current policy) and policy improvement until reaching the optimal policy.

### Monte Carlo Methods
- **Monte Carlo Basic**: Learns from complete episodes by averaging returns for each state-action pair.
- **Epsilon-Greedy MC**: Balances exploration and exploitation using an ε-greedy policy during learning.

### Temporal Difference Learning
- **TD(0)**: Updates value estimates based on single-step predictions without waiting for episode completion.
- **SARSA**: On-policy TD control that learns Q-values following the current policy.
- **Expected SARSA**: Similar to SARSA but uses expected value over all actions instead of sampled next action.
- **Q-Learning**: Off-policy TD control that learns optimal Q-values regardless of behavior policy.

### Function Approximation
- Uses neural networks to approximate value functions, enabling generalization to unseen states.
- **DQN**: Combines Q-learning with deep neural networks and experience replay for stable learning.

### Policy Gradient
- **REINFORCE**: Directly optimizes the policy by following the gradient of expected returns.

### Actor-Critic
- **A2C**: Combines policy gradient (actor) with value function estimation (critic) for more stable learning with lower variance.

## 🔧 Dependencies

- **numpy**: Numerical computations and array operations
- **torch**: Deep learning framework for neural network implementations
- **pygame**: Interactive visualization and rendering
- **matplotlib**: Plotting and visualization utilities
- **tqdm**: Progress bars for training loops
- **mypy**: Static type checking (development)

## 🐛 Troubleshooting

### Common Issues

**Issue: Pygame window not responding**
- Make sure you're calling `renderer.handle_events()` in your rendering loop
- The window requires event handling to remain responsive

**Issue: ModuleNotFoundError**
- Ensure you've installed the package with `uv pip install -e .`
- Verify you're in the correct Python environment

**Issue: Training not converging**
- Try adjusting hyperparameters (learning rate, episode count, etc.)
- Some algorithms require more episodes or different initialization
- Check that the goal state is reachable from the initial state

**Issue: Import errors after installation**
- Make sure to install in editable mode with `-e` flag
- Restart your Python interpreter after installation

## 📄 License

This project is open source. Please check the repository for license details.

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Adding new RL algorithms
- Improving documentation
- Fixing bugs
- Adding tests
- Enhancing visualization

Please feel free to open issues or submit pull requests.

## 🙏 Acknowledgments

This project implements classic reinforcement learning algorithms based on concepts from:
- Sutton & Barto's "Reinforcement Learning: An Introduction"
- Modern deep RL research papers

## 📞 Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository.

