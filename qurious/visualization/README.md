# Qurious Visualization Module

This module provides comprehensive visualization capabilities for grid world environments in the Qurious reinforcement learning framework. It's designed to be modular, extensible, and produce high-quality visualizations suitable for teaching and research purposes.

## Features

- **ASCII Visualization**: Quick text-based rendering for debugging in the terminal
- **Matplotlib Visualization**: High-quality visual rendering with configurable colors and styles
- **Policy Visualization**: Render deterministic or stochastic policies with arrows proportional to action probabilities
- **Value Function Visualization**: Visualize state values as a heatmap overlay
- **Action-Value Visualization**: Display Q-values for each action in each state
- **Trajectory Visualization**: Show agent paths through the environment
- **Eligibility Trace Visualization**: Display eligibility traces as a heatmap
- **Animation Support**: Create animated GIFs of agent episodes
- **Composable Layers**: Mix and match different visualization elements
- **Customizable Styling**: Configure colors, transparency, text settings, and more

## Examples

### Basic Grid World

```python
from qurious.environments import GridWorld
from qurious.visualization import GridWorldVisualizer, GridLayer, AgentLayer

# Create a grid world
env = GridWorld(width=5, height=4, obstacles=[(1, 2), (2, 1)])

# Create a visualizer
vis = GridWorldVisualizer(env)
vis.add_layer(GridLayer())
vis.add_layer(AgentLayer())

# Show ASCII rendering
print(vis.render_ascii())

# Create and save matplotlib rendering
fig, ax = vis.render_matplotlib()
fig.savefig("grid_world.png")
```

### Visualizing a Policy

```python
from qurious.environments import GridWorld
from qurious.policy import DeterministicTabularPolicy
from qurious.visualization import create_gridworld_visualizer

# Create a grid world and policy
env = GridWorld(width=5, height=4)
policy = DeterministicTabularPolicy(env.get_n_states(), env.get_n_actions())

# Set policy actions (this is just an example)
for s in range(env.get_n_states()):
    policy.update(s, 1)  # Always move right

# Create a visualizer with the policy
vis = create_gridworld_visualizer(env, policy=policy)

# Render and save
fig, ax = vis.render_matplotlib()
fig.savefig("policy_visualization.png")
```

### Animating an Episode

```python
from qurious.environments import GridWorld
from qurious.experience import Transition
from qurious.visualization import (
    GridWorldVisualizer, GridLayer, AgentLayer, 
    TrajectoryLayer, EpisodeAnimator
)

# Create a grid world
env = GridWorld(width=5, height=4)

# Create some transitions (normally these would come from an agent)
transitions = [
    Transition(0, 1, -0.1, 1, False),  # State 0, action right, to state 1
    Transition(1, 1, -0.1, 2, False),  # State 1, action right, to state 2
    Transition(2, 2, -0.1, 7, False),  # State 2, action down, to state 7
    Transition(7, 1, -0.1, 8, False),  # State 7, action right, to state 8
    Transition(8, 1, 10.0, 9, True),   # State 8, action right, to goal
]

# Create a visualizer
vis = GridWorldVisualizer(env)
vis.add_layer(GridLayer())
vis.add_layer(AgentLayer())
vis.add_layer(TrajectoryLayer())

# Create an animator and save the animation
animator = EpisodeAnimator(env, vis, transitions)
anim = animator.animate(save_path="episode.gif", fps=2)
```

## Module Structure

- `base.py`: Contains the core `GridWorldVisualizer` class and `Layer` abstract base class
- `grid_agent_layers.py`: Basic layers for grid structure and agent position
- `policy_layer.py`: Visualization of policies
- `value_function_layer.py`: Visualization of state value functions
- `action_value_layer.py`: Visualization of action value functions
- `trajectory_layer.py`: Visualization of agent paths and trajectories
- `eligibility_trace_layer.py`: Visualization of eligibility traces
- `__init__.py`: Module initialization and convenience functions

## Folder Structure

```
qurious/
├── visualization/
│   ├── __init__.py
│   ├── base.py
│   ├── grid_agent_layers.py
│   ├── policy_layer.py
│   ├── value_function_layer.py
│   ├── action_value_layer.py
│   ├── trajectory_layer.py
│   └── eligibility_trace_layer.py
└── examples/
    └── visualization_examples.py
```

## Extending the Visualization System

The system is designed to be easily extended with new visualization layers:

1. Create a new class that inherits from `Layer`
2. Implement the `render_ascii` and `render_matplotlib` methods
3. Add the layer to a visualizer instance using `add_layer`

Example:

```python
class CustomLayer(Layer):
    def __init__(self, name="Custom", enabled=True):
        super().__init__(name, enabled)
    
    def render_ascii(self, grid, env):
        # Implement ASCII rendering
        return grid
    
    def render_matplotlib(self, fig, ax, grid, env):
        # Implement matplotlib rendering
        pass

# Usage
vis = GridWorldVisualizer(env)
vis.add_layer(CustomLayer())
```
