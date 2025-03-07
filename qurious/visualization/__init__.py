# Create qurious.visualization module
from qurious.visualization.base import GridWorldVisualizer, Layer
from qurious.visualization.grid_agent_layers import GridLayer, AgentLayer
from qurious.visualization.policy_layer import PolicyLayer
from qurious.visualization.state_value_layer import StateValueLayer
from qurious.visualization.action_value_layer import ActionValueLayer
from qurious.visualization.trajectory_layer import TrajectoryLayer, EpisodeAnimator
from qurious.visualization.eligibility_trace_layer import EligibilityTraceLayer
from qurious.visualization.utils import clear_output

# Shortcut function to create a preconfigured visualizer with common layers
def create_gridworld_visualizer(env, policy=None, value_fn=None, action_value_fn=None):
    """
    Create a GridWorldVisualizer with common layers.

    Args:
        env: The grid world environment
        policy: Optional policy to visualize
        value_fn: Optional state value function to visualize
        action_value_fn: Optional action value function to visualize

    Returns:
        Configured GridWorldVisualizer instance
    """
    vis = GridWorldVisualizer(env)

    # Add grid and agent layers
    vis.add_layer(GridLayer())
    vis.add_layer(AgentLayer())

    # Add policy layer if provided
    if policy:
        vis.add_layer(PolicyLayer(policy, enabled=True))

    # Add value function layer if provided
    if value_fn:
        vis.add_layer(StateValueLayer(value_fn, enabled=True))

    # Add action value layer if provided
    if action_value_fn:
        vis.add_layer(ActionValueLayer(action_value_fn, enabled=False))

    return vis


__all__ = [
    "GridWorldVisualizer",
    "Layer",
    "GridLayer",
    "AgentLayer",
    "PolicyLayer",
    "StateValueLayer",
    "ActionValueLayer",
    "TrajectoryLayer",
    "EligibilityTraceLayer",
    "EpisodeAnimator",
    "create_gridworld_visualizer",
    "clear_output",
]
