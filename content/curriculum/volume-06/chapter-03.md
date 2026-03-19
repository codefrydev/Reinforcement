---
title: "Chapter 53: Planning with Known Models"
description: "BFS planner for gridworld; compare with DP."
date: 2026-03-10T00:00:00Z
weight: 53
draft: false
tags: ["planning", "BFS", "gridworld", "dynamic programming", "curriculum"]
keywords: ["planning with known models", "BFS planner", "gridworld", "DP"]
---

**Learning objectives**

- Implement a **planner** using **breadth-first search (BFS)** for a gridworld with **known deterministic** dynamics.
- Recover the **optimal policy** (path to goal) and compare with **dynamic programming** (value iteration) in terms of computation and result.
- Relate BFS to shortest-path planning in **robot navigation**.

**Concept and real-world RL**

When the **model is known** and deterministic, we can **plan** without learning: BFS finds the shortest path from start to goal; value iteration computes optimal values for all states. In **robot navigation** (grid or graph), BFS is used for pathfinding; DP is used when we need values everywhere (e.g. for reward shaping). Both assume the model is correct; in RL we often learn the model or the value function from data.

**Where you see this in practice:** Game AI pathfinding (A*, BFS); industrial control with known dynamics.

**Illustration (BFS vs DP):** Planning with BFS finds the shortest path; DP (value iteration) yields the same optimal policy. The chart below shows the number of steps to converge for a small grid (BFS: one run; DP: sweeps).

{{< chart type="bar" palette="comparison" title="Steps to solution (4×4 grid)" labels="BFS, Value iteration" data="16, 12" yLabel="Steps" >}}

**Exercise:** For a simple gridworld with known deterministic dynamics, implement a planner using breadth-first search to find the optimal policy. Compare with dynamic programming.

**Professor's hints**

- BFS: from start, expand neighbors (up/down/left/right); track visited; stop when goal is reached. Backtrack to get the action sequence.
- DP: run value iteration until convergence; then extract policy \\(\\pi(s) = \\arg\\max_a \\sum_{s'} P(s'|s,a)[r + \\gamma V(s')]\\).
- Compare: BFS gives one path from start; DP gives policy for every state. For a single start–goal pair, BFS is often cheaper.

**Common pitfalls**

- **Ties:** If multiple actions are optimal, BFS may return one path; DP policy may break ties arbitrarily. Be consistent.
- **Discount in BFS:** BFS finds shortest path (minimum steps); if rewards are -1 per step, that matches. For general rewards, use a cost-sensitive search or DP.

{{< collapse summary="Worked solution (warm-up: planning with a model)" >}}
**Key idea:** With a known or learned model we can plan: e.g. value iteration, policy iteration, or search (BFS/DFS) from the current state. BFS with -1 per step finds the shortest path to the goal. The model gives \\(P(s',r|s,a)\\); we use it to compute values or to simulate rollouts without taking real env steps.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** When is BFS preferable to value iteration for finding an optimal policy from a single start state?
2. **Coding:** Implement BFS and value iteration on a 5×5 grid with goal at (4,4). Compare the policy from BFS (from (0,0)) with the policy from DP at (0,0). Do they match?
3. **Challenge:** Add a small probability (e.g. 0.1) of a random transition. Use **value iteration** only (BFS no longer exact). Compare the optimal policy with the deterministic case.
4. **Variant:** Scale the grid to 10×10 and add walls blocking some cells. How does the number of value iteration sweeps to convergence grow? Does BFS remain computationally cheaper for a single start state?
5. **Debug:** A student implements value iteration but forgets to use the Bellman *max* and instead uses the Bellman *expectation* (averaging over actions equally). What policy does this converge to, and why is it suboptimal?
6. **Conceptual:** Value iteration requires a known transition model P(s'|s,a). If the model is only approximate, how should you expect the resulting policy to differ from the true optimal? Is it better to have an over-optimistic or over-pessimistic model, and why?
