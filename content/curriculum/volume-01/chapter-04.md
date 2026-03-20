---
title: "Chapter 4: The Reward Hypothesis"
description: "Reward function for self-driving car and reward hacking."
date: 2026-03-10T00:00:00Z
weight: 4
draft: false
difficulty: 6
tags: ["reward hypothesis", "reward function", "reward hacking", "curriculum"]
keywords: ["reward hypothesis", "reward design", "reward hacking", "self-driving car"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Ch 4"
---

**Learning objectives**

- State the reward hypothesis: that goals can be captured by scalar reward signals.
- Design a reward function for a concrete task and anticipate unintended behavior.
- Identify and fix "reward hacking" (exploiting the reward design instead of the intended goal).

**Concept and real-world RL**

The **reward hypothesis** says that we can capture what we want the agent to do by defining a scalar reward at each step; the agent's goal is to maximize cumulative reward. In practice, **reward design** is hard: the agent will optimize exactly what you reward, so oversimplified or buggy rewards lead to **reward hacking** (e.g. the agent finds a loophole that yields high reward without achieving the real goal). Examples: a robot rewarded for "distance to goal" might push the goal; a game agent rewarded for "score" might find a way to increment score without playing. Self-driving, robotics, and game AI all require careful reward shaping and testing for exploits.

**Illustration (reward components):** A typical self-driving reward might combine a large positive for reaching the goal, a large negative for collision, and a small negative per step to encourage speed. The chart below shows example magnitudes for each component.

{{< chart type="bar" title="Example reward components (self-driving)" labels="Reach goal, Collision, Per step" data="10, -5, -0.1" >}}

**Exercise:** Design a reward function for a self-driving car that must reach a destination quickly without collisions. Then identify a potential "reward hacking" scenario where the agent might exploit your function (e.g., spinning in circles). Propose a modification to fix it.

**Professor's hints**

- Typical components: positive reward for reaching the destination, negative for collision, and often a small negative per step (or negative for time) to encourage speed. Write \\(r(s,a,s')\\) or \\(r(s,a)\\) in words or formulas.
- Reward hacking examples: getting reward for "being close to goal" without actually arriving (agent circles nearby); getting reward for "no collision" by stopping forever; penalizing time in a way that encourages running red lights to save time. Pick one and describe how the agent would exploit it.
- Fixes: reward only on *reaching* the goal (terminal), not for proximity; add a time limit or penalty for not making progress; use a penalty for traffic violations; or shape reward to discourage the specific exploit you identified.

**Common pitfalls**

- **Rewarding the wrong thing:** "Distance to goal" can be gamed (move toward goal then away repeatedly if reward is given every step). Prefer terminal success reward or progress that cannot be reversed.
- **Ignoring the agent's perspective:** The agent only sees the reward you give. If you forget to penalize something (e.g. comfort, legality), the agent will ignore it.
- **Over-shaping:** Too much reward shaping can make the problem trivial or change the optimal policy. Prefer sparse, clear rewards when possible and add shaping only to help learning.

{{< collapse summary="Worked solution (warm-up: reward hacking in a game)" >}}
**Warm-up:** Give one example of reward hacking in a video game (e.g. "kill enemies" leading to farming spawn points). How would you change the reward to reduce the exploit?

**Example:** Reward = +1 per enemy killed. The agent discovers that standing near a spawn point yields a steady stream of respawning enemies and thus very high reward without completing the level or taking risk.

**Fix:** (1) Reward only for *level completion* or *objectives* (e.g. reach exit, defeat boss), not per kill. (2) Or cap reward per time window / per spawn region so that farming one spot has diminishing returns. (3) Or add a time penalty so that delaying (farming) reduces total return. The key is to tie reward to the *intended* goal (finish the level) rather than to a proxy (raw kill count) that can be gamed. In RL we always optimize the reward we define—so the design must match the real goal.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Give one example of reward hacking in a video game (e.g. "kill enemies" leading to farming spawn points). How would you change the reward to reduce the exploit?
2. **Coding:** Write a Python function that takes a reward function (state, action → float) and a list of trajectories (each a list of (s, a, r) tuples) and returns the mean return per trajectory. Test with a small hand-made trajectory.
3. **Challenge:** Design a reward for a warehouse robot that must pick items and place them in bins. List at least two possible reward hacks and how you would modify the reward to address them.
4. **Variant:** Modify your self-driving reward to add a comfort term (e.g. penalize large steering angle changes). Does this introduce any new reward hacks? How would you prevent them?
5. **Debug:** The function below always returns 0 because of a scoping bug. Find and fix it.

{{< pyrepl code="def mean_return(trajectories, gamma=0.9):\n    total = 0\n    for traj in trajectories:\n        episode_return = 0\n        for t, (s, a, r) in enumerate(traj):\n            episode_return += gamma**t * r\n    # BUG: total is never updated inside the loop\n    return total / len(trajectories) if trajectories else 0\n\ntraj1 = [('s0','a0',0), ('s1','a1',0), ('s2','a2',1)]\nprint(mean_return([traj1]))  # expected ~0.81" height="220" >}}

6. **Conceptual:** Can every goal be captured by a scalar reward? Give one real-world example where reward design is especially difficult (e.g. multi-objective tasks, human preferences).
7. **Recall:** State the reward hypothesis in one sentence from memory.
