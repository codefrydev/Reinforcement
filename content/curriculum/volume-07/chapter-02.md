---
title: "Chapter 62: Intrinsic Motivation"
description: "State visitation count bonus; exploration in gridworld."
date: 2026-03-10T00:00:00Z
weight: 62
draft: false
tags: ["intrinsic motivation", "state visitation", "exploration", "gridworld", "curriculum"]
keywords: ["intrinsic motivation", "state visitation count", "exploration bonus", "gridworld"]
---

**Learning objectives**

- Design an **intrinsic reward** based on **state visitation count**: bonus = \\(1/\\sqrt{\\text{count}}\\) (or similar) so rarely visited states are more attractive.
- Implement an agent that uses **total reward = extrinsic + intrinsic** and compare **exploration behavior** (e.g. coverage of the state space) with an agent that uses only extrinsic reward.
- Relate to **curiosity** and **exploration** in **game AI** and **robot navigation**.

**Concept and real-world RL**

**Intrinsic motivation** gives the agent a bonus for visiting **novel** or **surprising** states, so it explores even when extrinsic reward is sparse. **Count-based** bonus \\(1/\\sqrt{N(s)}\\) (inverse square root of visit count) encourages visiting states that have been seen fewer times. In **game AI** and **robot navigation**, this can help discover the goal; in **recommendation**, novelty bonuses encourage diversity. The combination extrinsic + intrinsic balances exploitation (reward) and exploration (novelty).

**Where you see this in practice:** Count-based exploration in MDPs; pseudo-counts in Atari (e.g. Bellemare et al.); curiosity modules.

**Illustration (intrinsic bonus):** The bonus \\(1/\\sqrt{\\text{count}}\\) is high for rarely visited states. The chart below shows intrinsic reward for states with different visit counts.

{{< chart type="bar" title="Intrinsic reward vs visit count" labels="1, 4, 9, 16, 25" data="1, 0.5, 0.33, 0.25, 0.2" >}}

**Exercise:** Design an intrinsic reward based on state visitation count. For a simple gridworld, count how many times each state is visited and give a bonus = \\(1/\\sqrt{\\text{count}}\\). Implement an agent that uses total reward = extrinsic + intrinsic. Compare exploration behavior.

**Professor's hints**

- Maintain a dict or array N[s] = visit count. After each step, intrinsic_r = 1/sqrt(N[s]) (or 1/(1+N[s])), then N[s] += 1. Total reward = extrinsic_r + beta * intrinsic_r; tune beta.
- Compare: run an agent with only extrinsic reward (e.g. +1 at goal) and one with extrinsic + intrinsic. Plot "number of unique states visited" vs steps; the intrinsic agent should cover more of the grid.
- Gridworld: small (e.g. 5×5) so you can visualize which cells get visited; goal in one corner.

**Common pitfalls**

- **Intrinsic reward too large:** If beta is huge, the agent ignores the goal and only seeks novelty. Scale intrinsic so it is on the same order as extrinsic (or tune).
- **Counts in continuous state:** For continuous states, use hashing or a density model (pseudo-count); this exercise uses discrete grid.

{{< collapse summary="Worked solution (warm-up: count-based exploration)" >}}
**Key idea:** Count-based bonus: \\(r^+_t = 1/\\sqrt{N(s_t)}\\) (or \\(1/(1+N(s_t))\\)) so the agent is rewarded for visiting states it has rarely seen. In discrete tabular settings we store \\(N(s)\\); in continuous we use hashing or a density model to estimate "novelty." This encourages the agent to cover the state space and can help in sparse-reward tasks.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does \\(1/\\sqrt{N(s)}\\) encourage visiting rarely visited states more than \\(1/N(s)\\)?
2. **Coding:** Implement count-based intrinsic reward in a 5×5 gridworld. Plot coverage (% cells visited at least once) vs steps for beta=0, 0.1, 1.0. Which explores fastest?
3. **Challenge:** Use **pseudo-counts** from a density model (e.g. fit a kernel density estimator on visited states). Use bonus = 1/sqrt(pseudo_count). Does it work in a continuous state space (e.g. 2D position)?
