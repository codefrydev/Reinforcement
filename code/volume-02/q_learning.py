# Volume 2 Starter: Q-Learning
# Chapter 14 exercise
import numpy as np
import matplotlib.pyplot as plt
import random

class GridworldEnv:
    """5×5 gridworld. Start (0,0), goal (4,4)."""
    
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
        
        # TODO: handle wall (out of bounds) -> stay, reward=-1
        # TODO: handle goal -> reward=+10, done=True
        # TODO: other steps -> reward=-1, done=False
        pass


def q_learning(env, n_episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Tabular Q-learning.
    
    Returns:
        Q: dict mapping (state, action) -> float
        returns: list of total rewards per episode
    """
    n_actions = 4
    Q = {}  # Q[(state, action)] = value
    episode_returns = []
    
    def get_q(s, a):
        return Q.get((s, a), 0.0)
    
    def best_action(s):
        return max(range(n_actions), key=lambda a: get_q(s, a))
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            # TODO: epsilon-greedy action selection
            action = None  # replace with epsilon-greedy
            
            # TODO: take step
            next_state, reward, done = None, None, None  # replace with env.step(action)
            
            # TODO: Q-learning update
            # target = reward if done else reward + gamma * max_a Q(next_state, a)
            # Q[(state, action)] += alpha * (target - Q[(state, action)])
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_returns.append(total_reward)
    
    return Q, episode_returns


if __name__ == "__main__":
    env = GridworldEnv(size=5)
    Q, returns = q_learning(env)
    
    # Plot learning curve
    # Smooth with running average
    window = 100
    smoothed = [np.mean(returns[max(0,i-window):i+1]) for i in range(len(returns))]
    
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('Q-Learning on 5×5 Gridworld')
    plt.tight_layout()
    plt.savefig('q_learning_curve.png')
    plt.show()
    
    # Visualize value function
    V = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            V[r, c] = max(Q.get(((r,c), a), 0.0) for a in range(4))
    
    plt.figure(figsize=(5, 4))
    plt.imshow(V, cmap='Blues')
    plt.colorbar(label='max_a Q(s,a)')
    plt.title('Value Function')
    for r in range(5):
        for c in range(5):
            plt.text(c, r, f'{V[r,c]:.1f}', ha='center', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig('value_function.png')
    plt.show()
