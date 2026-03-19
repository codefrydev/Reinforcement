# Volume 4 Starter: REINFORCE (Monte Carlo Policy Gradient)
# Chapter 32 exercise — Pure NumPy implementation (no PyTorch)
# Note: for real applications, use PyTorch — this is for understanding the algorithm
import numpy as np
import matplotlib.pyplot as plt


def softmax(logits):
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def policy_forward(theta, state):
    """
    Linear softmax policy: π(a|s; θ) = softmax(θ[state])

    Args:
        theta: weight matrix, shape (n_states, n_actions)
        state: int, state index

    Returns:
        probs: probability distribution over actions
    """
    # TODO: return softmax(theta[state])
    return softmax(theta[state])


def log_prob(theta, state, action):
    """Log probability of action in state."""
    probs = policy_forward(theta, state)
    return np.log(probs[action] + 1e-8)


def sample_episode(theta, env_step, env_reset, max_steps=50):
    """
    Sample one episode using the current policy.

    Returns:
        states, actions, rewards: lists of trajectory data
    """
    state = env_reset()
    states, actions, rewards = [], [], []
    done = False
    steps = 0

    while not done and steps < max_steps:
        probs = policy_forward(theta, state)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward, done = env_step(state, action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        steps += 1

    return states, actions, rewards


def discounted_returns(rewards, gamma=0.99):
    """Compute G_t for each step t in the episode."""
    T = len(rewards)
    returns = np.zeros(T)
    # TODO: G_T = rewards[-1]; G_t = rewards[t] + gamma * G_{t+1}
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def reinforce_update(theta, states, actions, returns, alpha=0.01):
    """
    REINFORCE gradient update: θ += α * G_t * ∇log π(a_t|s_t; θ)

    Gradient of log π w.r.t. θ[s_t] for softmax policy:
    d/dθ[s][a_taken] += 1 - π(a_taken|s)
    d/dθ[s][other]   -= π(other|s)

    Returns updated theta.
    """
    theta = theta.copy()
    for t, (s, a, G) in enumerate(zip(states, actions, returns)):
        # TODO: compute gradient of log π(a|s; θ) w.r.t. θ[s]
        probs = policy_forward(theta, s)
        grad = -probs.copy()
        grad[a] += 1   # d/dθ[s][a] = 1 - π(a|s) for chosen action

        # TODO: theta[s] += alpha * G * grad
        pass
    return theta


# Simple 4-state chain MDP for testing
# States: 0, 1, 2, 3 (terminal=3). Actions: 0=left, 1=right
# Transitions: right from state s -> s+1 (reward=0), right from 2 -> 3 (reward=+1)
#              left from state s -> max(0, s-1) (reward=-1)
N_STATES = 4
N_ACTIONS = 2

def env_reset():
    return 0

def env_step(state, action):
    if state == 3:
        return state, 0, True
    if action == 1:   # right
        next_s = state + 1
        reward = 1 if next_s == 3 else 0
        done = next_s == 3
        return next_s, reward, done
    else:   # left
        return max(0, state - 1), -1, False


if __name__ == "__main__":
    np.random.seed(42)
    theta = np.zeros((N_STATES, N_ACTIONS))   # initialise policy weights
    episode_returns = []

    for ep in range(500):
        states, actions, rewards = sample_episode(theta, env_step, env_reset)
        returns = discounted_returns(rewards, gamma=0.99)
        theta = reinforce_update(theta, states, actions, returns, alpha=0.05)
        episode_returns.append(sum(rewards))

    # Plot learning curve
    window = 20
    smoothed = [np.mean(episode_returns[max(0,i-window):i+1]) for i in range(len(episode_returns))]
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('REINFORCE on Chain MDP')
    plt.tight_layout()
    plt.show()

    print("Final policy (action probabilities):")
    for s in range(N_STATES):
        probs = policy_forward(theta, s)
        print(f"  State {s}: left={probs[0]:.3f}, right={probs[1]:.3f}")
