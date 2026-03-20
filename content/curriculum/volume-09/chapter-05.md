---
title: "Chapter 85: Multi-Agent DDPG (MADDPG)"
description: "MADDPG on simple spread; centralized critics, decentralized actors."
date: 2026-03-10T00:00:00Z
weight: 85
draft: false
difficulty: 8
tags: ["MADDPG", "multi-agent", "centralized critic", "curriculum"]
keywords: ["MADDPG", "multi-agent DDPG", "centralized critics", "decentralized actors"]
roadmap_color: "blue"
roadmap_icon: "network"
roadmap_phase_label: "Vol 9 · Ch 5"
---

**Learning objectives**

- **Implement** **MADDPG** for the Multi-Agent Particle Environment (e.g. "simple spread"): each agent has a **decentralized actor** (policy π_i(o_i) or π_i(s_i)) and a **centralized critic** Q_i(s, a_1,...,a_n) that takes the full state and all actions.
- **Train** the critics with TD targets using (s, a_1,...,a_n) and the actors with the gradient of Q_i w.r.t. agent i's action (DDPG-style).
- **Explain** why centralized critics help: each Q_i can use the full state and joint action, so the critic sees a stationary environment; the actor for agent i is updated to maximize Q_i(s, a_1,...,a_i,...,a_n) by changing a_i (with a_i = π_i(o_i) at execution).
- **Run** on "simple spread" (or similar) and report coordination behavior and return.
- **Relate** MADDPG to **robot navigation** (multi-robot) and **game AI** (cooperative or competitive).

**Concept and real-world RL**

**MADDPG** extends DDPG to multi-agent settings using **CTDE**: each agent i has a **centralized critic** Q_i(s, a_1,...,a_n) that takes the **global state** s and **all agents' actions** (a_1,...,a_n). The **actor** for agent i outputs a_i from o_i (or s_i) only. During training, we have s and all a_j, so we can compute Q_i and its gradient w.r.t. a_i; the actor is updated to maximize Q_i(s, π_1(o_1),...,π_i(o_i),...,π_n(o_n)) by backprop through a_i = π_i(o_i). At execution, each agent uses only π_i(o_i). In **robot navigation** and **game AI**, MADDPG is used for continuous control with multiple agents (e.g. cooperative spread, predator-prey).

**Where you see this in practice:** MADDPG in multi-agent particle envs; CTDE for continuous MARL.

**Illustration (MADDPG):** Centralized critics use all agents' states and actions; actors use only local observations. The chart below shows mean return on "simple spread" over training.

{{< chart type="line" palette="return" title="Mean return (MADDPG simple spread)" labels="0, 50k, 100k, 150k, 200k" data="-150, -80, -20, 20, 50" xLabel="Step" yLabel="Mean return" >}}

**Exercise:** Implement MADDPG for the Multi-Agent Particle Environment (e.g., "simple spread"). Use centralized critics that have access to all agents' states and actions, but decentralized actors.

**Professor's hints**

- **Simple spread:** N agents and N landmarks; agents must cover the landmarks (each agent to one landmark). Reward can be negative distance to nearest landmark (or shared reward). State = positions of all agents and landmarks; observation for agent i = e.g. relative positions to others and landmarks (or full state if you prefer).
- **Critic:** Q_i(s, a_1,...,a_n). Input: concatenate s (or [o_1,...,o_n]) and (a_1,...,a_n). Output: scalar. Train with TD: target = r_i + γ Q_i'(s', π_1'(o_1'),...,π_n'(o_n')) (use target networks for π' and Q').
- **Actor:** π_i(o_i) → a_i. Loss for agent i: -Q_i(s, a_1,...,π_i(o_i),...,a_n) where a_j for j≠i are from the replay buffer (or current policies). So we only gradient through π_i.
- **Execution:** Each agent runs a_i = π_i(o_i); no need for other agents' actions or global state.

**Common pitfalls**

- **Replay buffer:** Store (s, a_1,...,a_n, r_1,...,r_n, s'). When sampling, you have all actions; use them for the critic target and for the actor update (other agents' actions from the buffer).
- **Continuous actions:** MADDPG is for continuous action spaces; ensure your env and policy output continuous actions (e.g. mean of a Gaussian, or bounded with tanh).
- **Instability:** Use target networks for both actors and critics; soft or periodic updates. Tune learning rates (often critics need smaller LR than in single-agent DDPG).

{{< collapse summary="Worked solution (warm-up: MADDPG)" >}}
**Key idea:** MADDPG: each agent has an actor and a critic. The critic for agent \\(i\\) takes *all* agents’ observations and actions (centralized); the actor for \\(i\\) takes only \\(i\\)’s observation (decentralized). So we have CTDE: the critic handles multi-agent credit assignment and non-stationarity; the actor is executed with local info only. We train with DDPG-style updates and target networks.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In MADDPG, why does each critic Q_i take (s, a_1,...,a_n) instead of only (o_i, a_i)?
2. **Coding:** Implement MADDPG for "simple_spread" (or a 2-agent continuous coordination task). Train for 50k steps. Plot mean return (sum of agent rewards) every 1k steps. Compare with IQL (each agent has Q_i(o_i, a_i) only) on the same env.
3. **Challenge:** Add **parameter sharing**: all agents share the same actor and critic (with agent index as input). Does it speed up learning or improve coordination in "simple spread"?
4. **Variant:** Change the number of agents from 2 to 4 in a spread task. How does the joint action space grow, and how does MADDPG scale compared to a centralized single-agent controller that takes all observations?
5. **Debug:** MADDPG trains but agent rewards diverge (one agent gets very high reward, the other near zero). The replay buffer stores transitions using the policy at collection time, but during training the critics use the *current* policy to compute a_j. This causes a mismatch in the centralized critic update. Explain how to correctly compute target actions for the centralized Bellman target.
6. **Conceptual:** MADDPG assumes all agents' policies are known during critic training. In a competitive setting, opponents may not share their policies. Describe the practical implications: when is the centralized critic assumption violated, and how do CTDE methods adapt for competitive vs cooperative settings?
