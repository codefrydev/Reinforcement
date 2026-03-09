---
title: "Chapter 1: The Reinforcement Learning Framework"
description: "Gridworld discounted return from a sequence of actions."
date: 2026-03-10T00:00:00Z
weight: 1
draft: false
---

**Learning objectives**

- Identify the main components of an RL system: agent, environment, state, action, reward.
- Compute the discounted return for a sequence of rewards.
- Relate the gridworld to real tasks (e.g. navigation, games) where an agent gets delayed reward.

**Concept and real-world RL**

In reinforcement learning, an **agent** interacts with an **environment**: at each step the agent is in a **state**, chooses an **action**, and receives a **reward** and a new state. The **return** is the sum of (discounted) rewards along a trajectory; the agent’s goal is to maximize this return. A **gridworld** is a simple environment where states are cells and actions move the agent; it models navigation (e.g. a robot moving to a goal, or a game on a grid). Discounting (\\(\gamma < 1\\)) makes future rewards worth less than immediate ones and keeps the return finite in long or infinite horizons.

**Exercise:** In a 3×3 gridworld, the agent starts at (0,0) and aims to reach a goal at (2,2) with a reward of +1. Every other step gives 0 reward, and hitting a wall (outside grid) gives -1 and stays in place. Write a Python function that takes a sequence of actions (up, down, left, right) and returns the total discounted return (\\(\gamma = 0.9\\)).

**Professor's hints**

- Encode actions as you like (e.g. 0=up, 1=down, 2=left, 3=right) and update (row, col) accordingly. Moving "up" usually decreases the row index.
- Maintain current (row, col); for each action, compute the next cell. If the next cell is outside [0,2]×[0,2], stay in place and add reward -1; if it is (2,2), add +1 and you can stop (or continue; define whether the episode terminates at the goal).
- The return is \\(G = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\cdots\\). Use a loop: at each step add \\(\\gamma^t \\cdot r_t\\) to the total.
- Test with a short sequence that reaches (2,2) in a few steps and verify the return by hand.

**Common pitfalls**

- **Off-by-one in grid indices:** (0,0) is top-left if row 0 is "up"; check whether "up" means row-1 or row+1 and be consistent.
- **Forgetting to discount:** Each reward must be multiplied by \\(\gamma^t\\) where \\(t\\) is the step index (0, 1, 2, …). Do not sum raw rewards unless \\(\gamma=1\\).
- **Wall semantics:** Clarify whether "hitting a wall" gives -1 and the agent stays in the same cell, or whether the action is simply not applied; implement one convention consistently.

**Extra practice**

1. **Warm-up:** For rewards \\([0, 0, 1]\\) and \\(\gamma = 0.9\\), compute \\(G_0\\) by hand. Then write a one-line loop that computes it in Python.
2. **Challenge:** Extend your function to support a **list of (state, reward)** pairs (e.g. from a saved trajectory) and compute the return from the first state. No environment logic—just the math.
