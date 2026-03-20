---
title: "Choosing Rewards"
description: "How to design reward signals for MDPs and gridworld—shaping, terminal rewards, and step penalties."
date: 2026-03-10T00:00:00Z
weight: 5
draft: false
difficulty: 6
tags: ["rewards", "reward design", "MDP", "curriculum"]
keywords: ["reward design", "reward shaping", "MDP", "reinforcement learning"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Choosing Rewards"
---

**Learning objectives**

- Understand how reward choice affects optimal behavior (what the agent will try to maximize).
- Use step penalties and terminal rewards in gridworld to encourage short paths or goal reaching.
- Avoid common pitfalls: reward hacking and unintended incentives.

## Why rewards matter

The agent’s goal in an MDP is to maximize cumulative (often discounted) reward. So **the reward function defines the task**. Changing rewards changes what is “optimal.” Design rewards so that the behavior you want is exactly what maximizes total reward.

## Common patterns

**Step penalty (e.g. -1 per step):** Encourages the agent to reach a goal in as few steps as possible. Used in the classic 4×4 gridworld: every step costs 1, so the optimal policy is the shortest path to a terminal state.

**Terminal rewards:** A positive reward at the goal (e.g. +10) and/or negative at a pit (e.g. -10). The agent learns to reach the goal and avoid the pit. Discount factor \\(\gamma < 1\\) makes nearby rewards more attractive than distant ones.

**Sparse vs. dense:** Sparse = reward only at goal (or failure). Dense = reward (or cost) at every step or based on progress. Dense rewards can make learning easier but can also lead to “reward shaping” that changes the true objective if done carelessly.

## Choosing rewards for gridworld

- **Minimize steps:** Use reward \\(-1\\) per step and 0 at terminal states. Then total return = \\(-\\)(number of steps).
- **Goal and pit:** Use \\(r = +10\\) at goal, \\(r = -10\\) at pit, and e.g. \\(r = -1\\) per step so the agent has a reason to finish quickly.
- **Stochastic transitions:** Same idea; the Bellman equation averages over next states, so the value function reflects expected cumulative reward.

## Pitfalls

- **Reward hacking:** If the reward signal has a loophole (e.g. the agent can get small positive reward by doing something useless repeatedly), the agent may exploit it. Design rewards so that maximizing them matches the intended task.
- **Inconsistent scale:** Keep rewards on a consistent scale so that discounting and step size behave predictably.

See [Gridworld](gridworld/) for the environment and [Chapter 4: The Reward Hypothesis](chapter-04/) for the link between rewards and goals.
