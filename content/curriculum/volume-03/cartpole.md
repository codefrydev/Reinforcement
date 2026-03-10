---
title: "CartPole"
description: "The CartPole (Inverted Pendulum) environment: state, actions, and solving it with value-based or policy-based methods."
date: 2026-03-10T00:00:00Z
weight: 22
draft: false
tags: ["CartPole", "Gym", "reinforcement learning", "curriculum"]
keywords: ["CartPole", "inverted pendulum", "OpenAI Gym", "control"]
---

**Learning objectives**

- Understand the CartPole environment: state (cart position, velocity, pole angle, pole angular velocity), actions (left/right), and reward (+1 per step until termination).
- Implement a solution using linear function approximation (e.g. tile coding or simple features) and semi-gradient SARSA or Q-learning.
- Optionally solve with a small neural network (e.g. DQN-style) as in later chapters.

## What is CartPole?

**CartPole** (also called Inverted Pendulum) is a classic control task in OpenAI Gym / Gymnasium. A pole is attached to a cart that moves on a track. The **state** is continuous: cart position \\(x\\), cart velocity \\(\dot{x}\\), pole angle \\(\theta\\), pole angular velocity \\(\dot{\theta}\\). **Actions** are discrete: 0 = push left, 1 = push right. **Reward:** +1 for every step until the episode ends. The episode ends when the pole angle goes outside a range (e.g. \\(\pm 12°\\)) or the cart leaves the track (if bounded), or after a max step count (e.g. 500). So the goal is to **keep the pole upright as long as possible** (maximize total reward = number of steps).

## Why use CartPole?

- **Simple but non-trivial:** Small state space (4D), 2 actions. Good for testing function approximation (linear or small neural net) and for debugging RL code.
- **Standard benchmark:** Widely used in tutorials and papers; easy to compare with others.
- **Continuous state:** You cannot enumerate states; you need function approximation (tile coding, linear features, or neural net).

## CartPole in code

**Environment:** Use `gym.make("CartPole-v1")` (or Gymnasium equivalent). `env.reset()` returns the initial state (4-dim array). `env.step(action)` returns `obs, reward, terminated, truncated, info`. Reward is 1.0 per step; episode ends when `terminated` or `truncated` is True.

**Solving with linear FA:** Discretize or featurize the state (e.g. tile coding over the 4 dimensions, or hand-crafted features like \\([\theta, \dot{\theta}, x, \dot{x}]\\) with scaling). Use semi-gradient SARSA or Q-learning with epsilon-greedy. Train for many episodes; plot total reward per episode. A good solution can reach 500 steps (or the environment max) consistently.

**Solving with a neural network:** Use a small MLP that takes the state and outputs Q(s,a) for each action (or one output per action). Train with DQN-style updates (experience replay, target network) as in [Chapter 23: DQN](chapter-03/). CartPole is small enough that even a simple 2-layer net can solve it quickly.

## CartPole code (minimal sketch)

```python
import gym
env = gym.make("CartPole-v1")
s, _ = env.reset()
# s is (x, x_dot, theta, theta_dot)
# Choose action 0 or 1 (e.g. epsilon-greedy from Q(s))
# s_next, r, term, trunc, _ = env.step(a)
# Update Q or policy; repeat until term or trunc
```

See [Chapter 21: Linear Function Approximation](chapter-01/) for tile coding and semi-gradient methods, [Feature Engineering](feature-engineering/) for designing features, and [Chapter 23: DQN](chapter-03/) for a neural net approach.
