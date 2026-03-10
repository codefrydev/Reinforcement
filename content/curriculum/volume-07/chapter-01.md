---
title: "Chapter 61: The Hard Exploration Problem"
description: "DQN with ε-greedy on Montezuma's Revenge; sparse rewards."
date: 2026-03-10T00:00:00Z
weight: 61
draft: false
---

**Learning objectives**

- Run **DQN with ε-greedy** on a **sparse-reward** environment (e.g. Montezuma's Revenge if available, or a simple maze).
- **Observe** that the agent rarely discovers the first key (or goal) when rewards are sparse.
- **Explain** why sparse rewards cause failure: no learning signal until the goal is reached; random exploration is unlikely to reach it.

**Concept and real-world RL**

**Hard exploration** occurs when the reward is **sparse** (e.g. only at the goal): the agent gets no signal until it accidentally reaches the goal, which may require a long, specific sequence of actions. In **game AI** (Montezuma's Revenge, Pitfall), ε-greedy DQN fails because random exploration almost never finds the key. In **robot navigation** and **recommendation**, sparse rewards (e.g. "user clicked" or "reached goal") similarly make learning slow. This motivates **intrinsic motivation**, **curiosity**, and **hierarchical** methods.

**Where you see this in practice:** Atari hard-exploration games; robotics with sparse success; recommenders with delayed feedback.

**Exercise:** In the environment "Montezuma's Revenge" (if available), try a standard DQN with \\(\epsilon\\)-greedy. Observe that it rarely gets the first key. Explain why sparse rewards cause failure.

**Professor's hints**

- If Montezuma's Revenge is not available, use a **simple maze** (e.g. 10×10 grid, goal in corner, reward +1 only at goal, 0 elsewhere). DQN with ε-greedy will struggle to ever reach the goal in reasonable time.
- Log "episodes until first goal" or "max distance to goal"; you will see little progress for many episodes.
- Explanation: the gradient only flows when the agent gets a non-zero reward; without that, Q-values do not propagate. Random exploration in a large state space has exponentially small probability of hitting the goal.

**Common pitfalls**

- **Blaming the algorithm:** DQN is fine for dense rewards; the issue is the reward structure, not DQN per se. Same for other model-free methods.
- **Short runs:** Run for at least 1M steps (or 10k episodes) to see that the agent rarely succeeds.

**Extra practice**

1. **Warm-up:** Why does ε-greedy with ε=0.1 rarely find a goal that is 20 steps away in a maze?
2. **Coding:** Implement a 10×10 maze with goal at (9,9), reward +1 at goal. Run DQN for 5000 episodes. Plot "best return so far" and "episodes until first +1 reward". How many episodes until first success?
3. **Challenge:** Add a **dense shaping** reward (e.g. -0.01 per step, or +0.1 for moving toward goal). Does DQN now learn to reach the goal? Discuss the trade-off.
