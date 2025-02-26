import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import networkx as nx


def visualize_grid_world(env, agent=None, value_function=None, plot_type="policy"):
    """
    Visualize a grid world environment with policy or value function.

    Args:
        env: Grid world environment
        agent: Agent with a policy (required for 'policy' plot)
        value_function: Value function (required for 'value' or 'q_values' plot)
        plot_type (str): Type of plot ('policy', 'value', or 'q_values')
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Define colors
    cmap = ListedColormap(["white", "gray", "green", "red"])

    # Create grid
    grid = np.zeros((env.height, env.width))

    # Mark obstacles as 1
    for r, c in env.obstacles:
        if 0 <= r < env.height and 0 <= c < env.width:
            grid[r, c] = 1

    # Mark goals as 2
    for r, c in env.goal_pos:
        if 0 <= r < env.height and 0 <= c < env.width:
            grid[r, c] = 2

    # Mark agent position as 3
    r, c = env.position
    grid[r, c] = 3

    # Plot grid
    ax.imshow(grid, cmap=cmap)

    # Add grid lines
    for i in range(env.width + 1):
        ax.axvline(i - 0.5, color="black", linewidth=1)
    for i in range(env.height + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)

    # Plot policy or value function
    if plot_type == "policy" and agent is not None:
        # Direction vectors for actions [UP, RIGHT, DOWN, LEFT]
        action_dirs = [
            (0, -0.4),  # UP: no x change, negative y (up in plot)
            (0.4, 0),  # RIGHT: positive x, no y change
            (0, 0.4),  # DOWN: no x change, positive y (down in plot)
            (-0.4, 0),  # LEFT: negative x, no y change
        ]

        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles and goal cells
                if grid[row, col] in [1, 2]:
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Get action probabilities
                action_probs = agent.policy.get_action_probabilities(state)

                # Find max probability for scaling
                max_prob = max(action_probs)

                # Plot arrows for each action with length proportional to probability
                for action, prob in enumerate(action_probs):
                    if prob > 0.01:  # Only plot if probability is significant
                        # Scale arrow by relative probability (compared to max)
                        scale = 0.3 + 0.7 * (prob / max_prob) if max_prob > 0 else 0

                        # Get direction vector and scale it
                        dx, dy = action_dirs[action]
                        dx *= scale
                        dy *= scale

                        # Draw arrow
                        ax.arrow(
                            col,
                            row,  # Start position (center of cell)
                            dx,
                            dy,  # Arrow direction and length
                            head_width=0.15 * scale,
                            head_length=0.15 * scale,
                            fc="black",
                            ec="black",
                            length_includes_head=True,
                        )

    elif plot_type == "value" and value_function is not None:
        # Plot state values
        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles
                if grid[row, col] == 1:
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Get value
                value = value_function.estimate(state)

                # Plot value
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=16, color="black")

    elif plot_type == "q_values" and value_function is not None:
        # Define action coordinates (relative to cell center)
        action_coords = [
            (0.5, 0.2),  # UP
            (0.8, 0.5),  # RIGHT
            (0.5, 0.8),  # DOWN
            (0.2, 0.5),  # LEFT
        ]

        # Plot Q-values for each state-action pair
        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles
                if grid[row, col] == 1:
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Plot Q-values for each action
                for action in range(4):
                    q_value = value_function.estimate(state, action)
                    dx, dy = action_coords[action]
                    ax.text(
                        col + dx - 0.5,
                        row + dy - 0.5,
                        f"{q_value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="blue",
                    )

    # Add labels
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xticklabels(range(env.width))
    ax.set_yticklabels(range(env.height))

    # Set title based on plot type
    if plot_type == "policy":
        ax.set_title("Grid World with Policy")
    elif plot_type == "value":
        ax.set_title("Grid World with State Values")
    elif plot_type == "q_values":
        ax.set_title("Grid World with Q-Values")
    else:
        ax.set_title("Grid World")

    # Create legend
    legend_elements = [
        patches.Patch(facecolor="white", edgecolor="black", label="Empty"),
        patches.Patch(facecolor="gray", edgecolor="black", label="Obstacle"),
        patches.Patch(facecolor="green", edgecolor="black", label="Goal"),
        patches.Patch(facecolor="red", edgecolor="black", label="Agent"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    return fig, ax


def visualize_mdp(mdp, env=None, layout_type="spring", show_rewards=True, threshold=0.01):
    """
    Visualize an MDP as a directed graph.

    Args:
        mdp: MarkovDecisionProcess object
        env: Original environment (optional, for state labeling)
        layout_type: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', 'planar')
        show_rewards: Whether to show rewards on edges
        threshold: Minimum transition probability to display

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add all states as nodes
    for s in range(mdp.states):
        # Label terminal states
        is_terminal = s in mdp.terminal_states

        # Create node label
        if env is not None:
            pos = env.index_to_state(s)
            label = f"S{s}: {pos}"
        else:
            label = f"S{s}"

        # Add node with properties
        G.add_node(s, label=label, terminal=is_terminal)

    # Add transitions as edges
    for s in range(mdp.states):
        for a in range(mdp.actions):
            action_name = ["UP", "RIGHT", "DOWN", "LEFT"][a] if mdp.actions == 4 else f"A{a}"

            for s_next in range(mdp.states):
                prob = mdp.transition_probs[s, a, s_next]
                reward = mdp.rewards[s, a, s_next]

                # Only add significant transitions
                if prob > threshold:
                    G.add_edge(
                        s,
                        s_next,
                        action=action_name,
                        probability=prob,
                        reward=reward,
                        weight=prob,  # For layout algorithms
                        label=f"{action_name}: {prob:.2f}" + (f", R={reward:.1f}" if show_rewards else ""),
                    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Set up node positions based on layout type
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == "planar":
        try:
            pos = nx.planar_layout(G)
        except nx.NetworkXException:
            # Fallback if graph is not planar
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw nodes
    terminal_nodes = [n for n in G.nodes if G.nodes[n]["terminal"]]
    non_terminal_nodes = [n for n in G.nodes if not G.nodes[n]["terminal"]]

    # Draw non-terminal nodes
    nx.draw_networkx_nodes(G, pos, nodelist=non_terminal_nodes, node_color="lightblue", node_size=500, alpha=0.8)

    # Draw terminal nodes with different color
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, node_color="lightgreen", node_size=500, alpha=0.8)

    # Draw edges with varying thickness based on probability
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=1,  # + 3 * data["probability"],
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=10 + 10 * data["probability"],
            connectionstyle=f"arc3,rad={0.1 if u != v else 0.3}",
        )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"))

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Set title and remove axis
    plt.title("MDP State Transition Graph")
    plt.axis("off")

    return fig, ax
