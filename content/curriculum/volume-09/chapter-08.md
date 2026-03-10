---
title: "Chapter 88: Multi-Agent PPO (MAPPO)"
description: "MAPPO with parameter sharing; centralized value; compare with IPPO."
date: 2026-03-10T00:00:00Z
weight: 88
draft: false
tags: ["MAPPO", "multi-agent PPO", "parameter sharing", "curriculum"]
keywords: ["MAPPO", "multi-agent PPO", "parameter sharing", "centralized value", "IPPO"]
---

**Learning objectives**

- **Adapt** a PPO implementation to the **multi-agent** setting with **parameter sharing**: all agents use the same policy network π(a_i | o_i) (and optionally the same value function), with agent identity or observation distinguishing them.
- **Use** a **centralized value function** V(s_global) or V(s_global, a_1,...,a_n) to reduce variance and improve credit assignment; the policy remains decentralized π_i(a_i | o_i).
- **Train** on a **collaborative** task (e.g. particle env or simple grid) and compare with **IPPO** (Independent PPO: each agent runs PPO with its own parameters and no centralized value).
- **Explain** the benefits of parameter sharing (sample efficiency, symmetry) and centralized value (better baseline, stability).
- **Relate** MAPPO to **game AI** (team games) and **robot navigation** (homogeneous multi-robot).

**Concept and real-world RL**

**Multi-Agent PPO (MAPPO)** applies PPO in multi-agent settings with two common design choices: **parameter sharing** (one policy network for all agents, with observation or id as input) and **centralized value** (critic uses global state, or global state and all actions, to compute a baseline). Parameter sharing improves sample efficiency when agents are homogeneous and increases symmetry; the centralized value provides a better baseline and can help with credit assignment. **IPPO** is the baseline where each agent runs PPO independently (no sharing, no central value). In **game AI** and **robot navigation**, MAPPO is used for cooperative control with many agents.

**Where you see this in practice:** MAPPO in SMAC and other MARL benchmarks; parameter sharing and central value in cooperative MARL.

**Exercise:** Adapt your PPO implementation to the multi-agent setting with parameter sharing among agents. Use a centralized value function (critic). Train on a collaborative task and compare with independent PPO.

**Professor's hints**

- **Parameter sharing:** One policy π(a | o, id) or π(a | o) where o is the observation (same format for each agent). For each agent i, pass o_i (and optionally id_i) and get π(·| o_i). All agents share the same parameters.
- **Centralized value:** V_ψ(s_global) where s_global = (o_1,...,o_n) or full state. For the advantage, use A_t = R_t - V_ψ(s_t). The return R_t can be the team return (same for all agents) or per-agent. Use team return for cooperative tasks.
- **PPO update:** Collect trajectories (all agents step); compute advantages with the centralized V; update the shared policy with PPO clip loss and the centralized V with MSE. Same clip and entropy as single-agent PPO.
- **IPPO:** Each agent has its own π_i and V_i (or no value sharing). Each gets the same team reward (or individual reward). Update each with PPO independently. Same env and horizon for fair comparison.

**Common pitfalls**

- **Observation format:** With parameter sharing, ensure each agent's observation has the same size/shape (e.g. pad or use relative coordinates). Include agent id or a one-hot if the role can differ.
- **Advantage computation:** Use the same advantage for all agents in a step (team reward) when the task is fully cooperative; otherwise use per-agent reward for advantage.
- **Central value at execution:** The centralized value is only for training; at test time we only need π(a_i | o_i) for each agent.

{{< collapse summary="Worked solution (warm-up: COMA)" >}}
**Key idea:** COMA uses a centralized critic \\(V(s)\\) or \\(Q(s, u)\\) and counterfactual advantage: for agent \\(i\\), \\(A_i = Q(s, u) - \\sum_{u_i'} \\pi_i(u_i'|o_i) Q(s, (u_{-i}, u_i'))\\). So we compare the joint action to the expected value if \\(i\\) had acted differently (marginalizing over \\(i\\)’s policy). This gives a credit assignment signal for \\(i\\) that accounts for others’ actions. Execution is decentralized (\\(\\pi_i\\) only).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might parameter sharing speed up learning in a cooperative task with homogeneous agents?
2. **Coding:** Implement MAPPO (parameter sharing + centralized V) and IPPO on "simple_spread" (or a 3-agent cooperative grid). Train both for 100k steps. Plot mean team return. Which reaches higher return faster?
3. **Challenge:** Add **value factorization** (e.g. V_tot = f(V_1(o_1),...,V_n(o_n), s)) and compare with a single centralized V. Does factorization help or hurt in your task?
