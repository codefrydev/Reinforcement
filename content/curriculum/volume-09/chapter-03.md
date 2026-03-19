---
title: "Chapter 83: Independent Q-Learning (IQL)"
description: "IQL in cooperative meet-up game; non-stationarity."
date: 2026-03-10T00:00:00Z
weight: 83
draft: false
tags: ["IQL", "independent Q-learning", "multi-agent", "non-stationarity", "curriculum"]
keywords: ["IQL", "independent Q-learning", "cooperative", "non-stationarity"]
---

**Learning objectives**

- **Implement** **independent Q-learning (IQL)** in a simple cooperative game (e.g. two agents must "meet" in the same cell or coordinate to achieve a joint goal).
- **Observe** the **non-stationarity** problem: as one agent's policy changes, the transition and reward from the other agent's perspective change, so the environment appears non-stationary.
- **Explain** why IQL can still work in some cooperative settings despite non-stationarity, and when it fails or converges slowly.
- **Compare** IQL with a baseline (e.g. random or hand-coded coordination) on the meet-up or similar task.
- **Relate** IQL and non-stationarity to **game AI** (teammates) and **dialogue** (multiple agents).

**Concept and real-world RL**

**Independent Q-learning (IQL)** means each agent runs its own Q-learning (or DQN) without observing the other agents' actions or policies. Each agent treats the others as part of the "environment," so from one agent's perspective, the environment is **non-stationary**: the transition and effective reward depend on the other agents' policies, which are changing over training. In **cooperative** settings (e.g. two agents must meet in the same cell), IQL can still learn if the task is simple or if exploration eventually finds good joint behavior, but convergence is not guaranteed and can be slow. In **game AI** and **dialogue**, IQL is a simple baseline for multi-agent coordination.

**Where you see this in practice:** IQL as baseline in MARL; non-stationarity in multi-agent learning; cooperative MARL benchmarks.

**Illustration (non-stationarity):** When both agents learn, the environment each sees is non-stationary. Return can oscillate. The chart below shows mean return over training (two agents, IQL, meet-up game).

{{< chart type="line" palette="return" title="Mean return (IQL, two agents)" labels="0, 2k, 4k, 6k, 8k, 10k" data="20, 60, 40, 80, 55, 95" xLabel="Step" yLabel="Mean return" >}}

**Exercise:** Implement independent Q-learning in a simple cooperative game (e.g., two agents need to meet). Show the non-stationarity problem: as one agent's policy changes, the other's environment changes, causing instability.

**Professor's hints**

- **Meet-up game:** Small grid (e.g. 5×5). Two agents; state for each = (own position, other's position) or (own position only if partial obs). Actions: move N/S/E/W. Reward: +1 when both are in the same cell (or when they "meet" within a step), 0 otherwise. Episodes run until meet or timeout.
- **IQL:** Each agent has its own Q(s_i, a_i) or Q(o_i, a_i). Update with standard Q-learning; the "environment" for agent i includes the other agent's action, but agent i does not condition on it (or does not see it). So from agent i's view, the transition is stochastic/non-stationary because the other agent's policy changes.
- **Showing non-stationarity:** Log variance of returns or Q-values over training; or plot learning curves (return vs episodes) and note instability (oscillations, slow convergence). Compare with a setting where the other agent is fixed (stationary)—learning should be more stable.
- Use **shared** or **separate** state representation; if each agent only sees its own position, the problem is harder (partial observability).

**Common pitfalls**

- **Credit assignment:** In cooperative tasks, both agents get the same reward; each might not know if the success was due to its action or the other's. IQL does not address this explicitly; the hope is that enough exploration finds coordinated behavior.
- **Exploration:** With two agents exploring independently, they may rarely "meet" early on; use sufficient exploration (e.g. ε-greedy with high ε initially) or a simple reward shaping (e.g. reward for getting closer).
- **Partial observability:** If agents do not see each other's position, the state is partially observable and the task is harder; you can start with full observability (each sees both positions) to isolate the non-stationarity effect.

{{< collapse summary="Worked solution (warm-up: multi-agent non-stationarity)" >}}
**Key idea:** When multiple agents learn simultaneously, each agent’s best response depends on the others’ policies. So the transition and reward from one agent’s perspective change over time (non-stationary). Independent Q-learning may not converge; we can use centralized critics (each agent’s Q depends on all states/actions) during training, then deploy with decentralized execution (each agent uses only its own observation).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does the transition probability P(s' | s, a_1, a_2) make the environment "non-stationary" for agent 1 when agent 2's policy is changing?
2. **Coding:** Implement the meet-up game and IQL for two agents. Plot mean return (over 100 eval episodes) every 500 training episodes. Run for 10k episodes. Do you see oscillations or slow convergence? Compare with one agent fixed (e.g. random) and the other learning.
3. **Challenge:** Add a **third agent** to the meet-up (all three must meet in one cell). Does IQL become more unstable? Plot return variance over training.
4. **Variant:** Change IQL to use opponent modeling: agent 1 maintains an estimate of agent 2's action distribution and conditions its Q-function on it. Does explicit opponent modeling stabilize training compared to ignoring the opponent's changing policy?
5. **Debug:** An IQL implementation on the meet-up task never converges — agents circle each other indefinitely. Logging shows both agents update their Q-tables on every step, using each other's updated Q as part of the environment transition. Explain why simultaneous Q-table updates without a fixed-point guarantee lead to instability and how alternating updates (one agent at a time) helps.
6. **Conceptual:** IQL treats other agents as part of the environment. Describe two scenarios: one where this approximation works fine in practice, and one where the non-stationarity is severe enough to prevent convergence. What property of the task determines which case applies?
