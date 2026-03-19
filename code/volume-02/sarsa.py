# Volume 2 Starter: SARSA (On-Policy TD Control)
# Chapter 13 exercise
import numpy as np
import matplotlib.pyplot as plt
import random


class GridworldEnv:
    """5×5 gridworld with cliff. Start (0,0), goal (4,4).
    Falling off the cliff (any state in row 4 except (4,4)) gives -100 and resets.
    """

    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        """
        action: 0=up, 1=down, 2=left, 3=right
        Returns: next_state, reward, done
        """
        row, col = self.state
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        nr = row + dr[action]
        nc = col + dc[action]

        # Handle wall
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self.state, -1, False

        # Handle goal
        if (nr, nc) == self.goal:
            self.state = (nr, nc)
            return (nr, nc), 10, True

        self.state = (nr, nc)
        return (nr, nc), -1, False


def sarsa(env, n_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    SARSA: on-policy TD control.
    Update: Q(s,a) += alpha * [r + gamma*Q(s',a') - Q(s,a)]
    where a' is the ACTUAL next action taken (not max).

    Returns:
        Q: dict mapping (state, action) -> float
        returns: list of total rewards per episode
    """
    n_actions = 4
    Q = {}
    episode_returns = []

    def get_q(s, a):
        return Q.get((s, a), 0.0)

    def epsilon_greedy(s):
        # TODO: return random action with prob epsilon, else argmax Q(s,.)
        pass

    for ep in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(state)   # SARSA: choose action before loop
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            # TODO: take step
            next_state, reward, done = None, None, None   # env.step(action)

            # TODO: choose NEXT action with epsilon-greedy (not max!)
            next_action = None   # epsilon_greedy(next_state)

            # TODO: SARSA update
            # target = reward + gamma * Q(next_state, next_action) if not done else reward
            # Q[(state, action)] += alpha * (target - Q[(state, action)])

            state = next_state
            action = next_action   # SARSA: use the chosen next action
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)

    return Q, episode_returns


if __name__ == "__main__":
    env = GridworldEnv()
    Q_sarsa, returns_sarsa = sarsa(env, n_episodes=1000)

    # Compare with Q-learning if you have it
    window = 50
    smoothed = [np.mean(returns_sarsa[max(0,i-window):i+1]) for i in range(len(returns_sarsa))]

    plt.figure(figsize=(8, 4))
    plt.plot(smoothed, label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('SARSA on Gridworld')
    plt.legend()
    plt.tight_layout()
    plt.show()
