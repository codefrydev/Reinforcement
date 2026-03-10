---
title: "Chapter 4: The Reward Hypothesis"
description: "Reward function for self-driving car and reward hacking."
date: 2026-03-10T00:00:00Z
weight: 4
draft: false
---

**Learning objectives**

- State the reward hypothesis: that goals can be captured by scalar reward signals.
- Design a reward function for a concrete task and anticipate unintended behavior.
- Identify and fix "reward hacking" (exploiting the reward design instead of the intended goal).

**Concept and real-world RL**

The **reward hypothesis** says that we can capture what we want the agent to do by defining a scalar reward at each step; the agent's goal is to maximize cumulative reward. In practice, **reward design** is hard: the agent will optimize exactly what you reward, so oversimplified or buggy rewards lead to **reward hacking** (e.g. the agent finds a loophole that yields high reward without achieving the real goal). Examples: a robot rewarded for "distance to goal" might push the goal; a game agent rewarded for "score" might find a way to increment score without playing. Self-driving, robotics, and game AI all require careful reward shaping and testing for exploits.

**Exercise:** Design a reward function for a self-driving car that must reach a destination quickly without collisions. Then identify a potential "reward hacking" scenario where the agent might exploit your function (e.g., spinning in circles). Propose a modification to fix it.

**Professor's hints**

- Typical components: positive reward for reaching the destination, negative for collision, and often a small negative per step (or negative for time) to encourage speed. Write \\(r(s,a,s')\\) or \\(r(s,a)\\) in words or formulas.
- Reward hacking examples: getting reward for "being close to goal" without actually arriving (agent circles nearby); getting reward for "no collision" by stopping forever; penalizing time in a way that encourages running red lights to save time. Pick one and describe how the agent would exploit it.
- Fixes: reward only on *reaching* the goal (terminal), not for proximity; add a time limit or penalty for not making progress; use a penalty for traffic violations; or shape reward to discourage the specific exploit you identified.

**Common pitfalls**

- **Rewarding the wrong thing:** "Distance to goal" can be gamed (move toward goal then away repeatedly if reward is given every step). Prefer terminal success reward or progress that cannot be reversed.
- **Ignoring the agent's perspective:** The agent only sees the reward you give. If you forget to penalize something (e.g. comfort, legality), the agent will ignore it.
- **Over-shaping:** Too much reward shaping can make the problem trivial or change the optimal policy. Prefer sparse, clear rewards when possible and add shaping only to help learning.

**Extra practice**

1. **Warm-up:** Give one example of reward hacking in a video game (e.g. "kill enemies" leading to farming spawn points). How would you change the reward to reduce the exploit?
2. **Coding:** Write a Python function that takes a reward function (state, action → float) and a list of trajectories (each a list of (s, a, r) tuples) and returns the mean return per trajectory. Test with a small hand-made trajectory.
3. **Challenge:** Design a reward for a warehouse robot that must pick items and place them in bins. List at least two possible reward hacks and how you would modify the reward to address them.
