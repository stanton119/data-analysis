import gymnasium as gym
import numpy as np


def run_q_learning_with_stats(num_episodes=5, print_steps=True):
    # Reward definition for FrozenLake-v1:
    # - Reach Goal (G): +1
    # - Reach Hole (H): 0
    # - Reach Frozen (F): 0
    # It is a sparse reward system.

    env = gym.make("FrozenLake-v1", is_slippery=False)

    # Initialize Q-table
    Q = np.random.rand(env.observation_space.n, env.action_space.n)

    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Exploration rate

    history = []

    if print_steps:
        print(
            f"{'Episode':<10} | {'Step':<5} | {'State':<5} | {'Action':<6} | {'Reward':<6} | {'Next State':<10}"
        )
        print("-" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0

        while not done:
            # epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Record transition
            transition = {
                "episode": episode,
                "step": step,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
            history.append(transition)

            # Print stats
            if print_steps:
                print(
                    f"{episode:<10} | {step:<5} | {state:<5} | {action:<6} | {reward:<6.1f} | {next_state:<10}"
                )

            # Q-update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            step += 1

            if done and reward == 1.0:
                if print_steps:
                    print(f"*** Goal reached in episode {episode} at step {step}! ***")
            elif done:
                pass  # Fell into hole or max steps

    return history, Q


if __name__ == "__main__":
    logs, Q = run_q_learning_with_stats(num_episodes=2)
    print(f"\nCaptured {len(logs)} transitions in function output.")
    if logs:
        print(f"First transition example: {logs[0]}")
