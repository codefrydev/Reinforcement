# Volume 1 Starter: Epsilon-Greedy Bandit
# Chapter 2 exercise
import random
import numpy as np
import matplotlib.pyplot as plt

class BanditEnv:
    """k-armed bandit with Gaussian rewards."""
    
    def __init__(self, k=10, seed=42):
        rng = np.random.default_rng(seed)
        self.means = rng.normal(0, 1, k)
        self.k = k
        self._rng = rng
    
    def pull(self, arm):
        """Return reward ~ Normal(means[arm], 1)."""
        # TODO: return self.means[arm] + self._rng.standard_normal()
        pass
    
    def optimal_arm(self):
        return int(np.argmax(self.means))


def epsilon_greedy(Q, epsilon, rng):
    """
    Epsilon-greedy action selection.
    
    Args:
        Q: array of Q-values, shape (k,)
        epsilon: float, exploration probability
        rng: numpy RNG
    
    Returns:
        action: int
    """
    # TODO: with prob epsilon choose random, else choose argmax(Q)
    pass


def run_bandit(k=10, epsilon=0.1, steps=1000, n_runs=200, seed=0):
    """
    Run epsilon-greedy bandit for n_runs independent runs.
    Returns average reward at each step (shape: steps).
    """
    rng = np.random.default_rng(seed)
    avg_rewards = np.zeros(steps)
    
    for run in range(n_runs):
        env = BanditEnv(k=k, seed=run)
        Q = np.zeros(k)      # Q-value estimates
        N = np.zeros(k)      # pull counts
        
        for t in range(steps):
            # TODO: select action, pull arm, update Q with incremental mean
            action = epsilon_greedy(Q, epsilon, rng)
            reward = env.pull(action)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            avg_rewards[t] += reward
    
    return avg_rewards / n_runs


if __name__ == "__main__":
    # Compare epsilon=0.1 vs epsilon=0 (greedy)
    rewards_eg = run_bandit(epsilon=0.1)
    rewards_greedy = run_bandit(epsilon=0.0)
    
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_eg, label='ε=0.1')
    plt.plot(rewards_greedy, label='ε=0.0 (greedy)')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.title('Epsilon-greedy vs Greedy (10-armed testbed)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bandit_comparison.png')
    plt.show()
    print("Plot saved to bandit_comparison.png")
