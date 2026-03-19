# Volume 3 Starter: Linear DQN (Function Approximation)
# Chapter 21 exercise — Linear value function approximation with semi-gradient TD
import numpy as np
import matplotlib.pyplot as plt


def features(state, grid_size=5):
    """
    Feature vector for a gridworld state (row, col).
    Returns a 6-dimensional feature vector: [1, row/n, col/n, row²/n², col²/n², row*col/n²]

    Args:
        state: (row, col) tuple
        grid_size: int, size of the grid
    """
    row, col = state
    n = grid_size - 1   # normalize by max value
    # TODO: return feature vector
    return np.array([
        1,               # bias term
        row / n,         # normalized row
        col / n,         # normalized col
        # TODO: add row^2/n^2, col^2/n^2, row*col/n^2
    ])


def gridworld_step(state, action, grid_size=5):
    """5×5 gridworld. Goal at (4,4). -1 per step, +10 at goal."""
    row, col = state
    dr = [-1, 1, 0, 0]; dc = [0, 0, -1, 1]
    nr = row + dr[action]; nc = col + dc[action]
    if not (0 <= nr < grid_size and 0 <= nc < grid_size):
        return state, -1, False
    if (nr, nc) == (grid_size-1, grid_size-1):
        return (nr, nc), 10, True
    return (nr, nc), -1, False


def linear_td_training(n_episodes=3000, alpha=0.01, gamma=0.99, epsilon=0.1, n_features=6):
    """
    Semi-gradient TD(0) with linear function approximation.
    V(s;w) = w · φ(s)

    Returns: w (weight vector), returns (list of episode returns)
    """
    w = np.zeros(n_features)   # weight vector
    episode_returns = []

    for ep in range(n_episodes):
        state = (0, 0)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # TODO: epsilon-greedy action selection
            # Compute V(s;w) = w · features(s) for each action's next state
            # Choose best action or random action with probability epsilon
            action = None   # replace with epsilon-greedy

            next_state, reward, done = gridworld_step(state, action)

            # TODO: semi-gradient TD(0) update
            # phi_s = features(state)
            # V_s = w @ phi_s
            # V_s_next = 0 if done else w @ features(next_state)
            # delta = reward + gamma * V_s_next - V_s
            # w += alpha * delta * phi_s

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)

    return w, episode_returns


if __name__ == "__main__":
    w, returns = linear_td_training()

    # Plot learning curve
    window = 100
    smoothed = [np.mean(returns[max(0,i-window):i+1]) for i in range(len(returns))]
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('Linear Function Approximation (Semi-gradient TD)')
    plt.tight_layout()
    plt.show()

    # Visualize learned value function
    V = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            phi = features((r, c))
            V[r, c] = w @ phi

    plt.figure(figsize=(5, 4))
    plt.imshow(V, cmap='Blues')
    plt.colorbar(label='V(s;w)')
    plt.title('Learned Value Function (Linear FA)')
    for r in range(5):
        for c in range(5):
            plt.text(c, r, f'{V[r,c]:.1f}', ha='center', va='center', fontsize=7)
    plt.tight_layout()
    plt.show()

    print("Learned weights:", np.round(w, 3))
