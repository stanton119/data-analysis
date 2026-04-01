# Design: FrozenLake Q-Learning Visualization

## Overview
A live-updating Jupyter Notebook dashboard to visualize the training progress of a Q-Learning agent on the `FrozenLake-v1` environment.

## Visual Layout
- **Two-Panel Dashboard (Side-by-Side):**
    1.  **Metric Plot (Left):** A line chart showing the rolling average (window of 100) of rewards/success rate over episodes.
    2.  **Q-Table Heatmap (Right):** A 4x4 grid representing the 16 states.
        - **Colors:** Intensity based on `max(Q[state])`.
        - **Annotations:** Arrows (↑, ↓, ←, →) indicating the greedy action for each state.

## Components
### 1. `src/q_learning/agent.py`
- `QLearningAgent` class:
    - `train(episodes, alpha, gamma, epsilon_decay)`: Core loop.
    - `get_best_action(state)`: Returns `argmax(Q[state])`.
- `TrainingMetrics`: A simple dataclass to store success history.

### 2. `src/q_learning/visualization.py`
- `plot_status(metrics, Q_table)`:
    - Uses `matplotlib` to create the two-panel figure.
    - Uses `IPython.display.clear_output(wait=True)` for live updates.

### 3. `notebooks/frozen_lake_training.ipynb`
- Scaffolding to import the agent and visualization tools.
- A cell that runs the training loop and calls `plot_status` every $N$ episodes (e.g., 50).

## Data Flow
1.  **Training Loop:** Agent interacts with `gymnasium`.
2.  **Metric Collection:** Each episode's result (success/fail) is appended to `TrainingMetrics`.
3.  **Visualization Trigger:** Every $N$ episodes, `plot_status` is called.
4.  **Display Update:** `matplotlib` renders the plots, and `clear_output` refreshes the notebook cell.

## Success Criteria
- The notebook shows a live-updating success rate curve.
- The Q-Table heatmap clearly shows the learned "path" to the goal.
- Training can be interrupted or completed with a final visualization.
