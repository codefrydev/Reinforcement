---
title: "Chapter 84: Centralized Training, Decentralized Execution (CTDE)"
description: "Explain CTDE with example; why it helps non-stationarity."
date: 2026-03-10T00:00:00Z
weight: 84
draft: false
difficulty: 8
tags: ["CTDE", "centralized training", "decentralized execution", "MARL", "curriculum"]
keywords: ["CTDE", "centralized training decentralized execution", "MARL", "non-stationarity"]
roadmap_color: "blue"
roadmap_icon: "network"
roadmap_phase_label: "Vol 9 · Ch 4"
---

**Learning objectives**

- **Explain** the **CTDE** paradigm: during **training**, algorithms can use **centralized** information (e.g. global state, all agents' actions) to learn better value functions or gradients; during **execution**, each agent uses only its **local** observation and policy (decentralized).
- **Give** a concrete example (e.g. QMIX, MADDPG, or a simple cooperative task) where the critic or value function uses global state and the actor uses only local observation.
- **Explain** why CTDE helps with **non-stationarity**: during training, the centralized critic sees the full state and other agents' actions, so the environment from the critic's perspective is "stationary" (we know the joint action); each agent's policy can then be trained with this stable learning signal.
- **Identify** why decentralized execution is important for scalability and deployment (no need to communicate all observations at test time).
- **Relate** CTDE to **game AI** (team coordination) and **robot navigation** (multi-robot systems).

**Concept and real-world RL**

**Centralized training, decentralized execution (CTDE)** is a design pattern in multi-agent RL: at **training time**, we can use **centralized** information (global state s, all actions a_1,...,a_n) to compute a joint Q(s, a_1,...,a_n) or a centralized critic, which provides a stable learning signal and mitigates non-stationarity; at **execution time**, each agent i uses only its **local** observation o_i and its own policy π_i(a_i | o_i)—no communication of other agents' states or actions. This gives the best of both worlds: learning benefits from global information, deployment stays scalable. In **game AI** and **robot navigation**, CTDE is used in QMIX, MADDPG, and similar algorithms for cooperative or mixed settings.

**Where you see this in practice:** QMIX, MADDPG, COMA; CTDE in StarCraft and multi-robot control.

**Illustration (CTDE):** Centralized training uses global state for the critic; decentralized execution uses only local observations. The chart below compares return (centralized critic vs independent) on a cooperative task.

{{< chart type="bar" palette="comparison" title="Final return (cooperative task)" labels="Independent, CTDE" data="70, 120" yLabel="Return" >}}

**Exercise:** Explain the CTDE paradigm with an example. Why does it help with non-stationarity during training while keeping execution scalable?

**Professor's hints**

- **Example:** In QMIX, each agent has a local Q_i(o_i, a_i); a **mixing network** takes (Q_1,...,Q_n) and global state s and outputs joint Q_tot. Training: we have access to s and all (o_i, a_i), so we can compute Q_tot and train with TD. Execution: each agent only needs o_i and its own Q_i (or policy); we do not need to compute Q_tot or know other agents' actions.
- **Non-stationarity:** In IQL, agent i's Q(s_i, a_i) depends on the others' behavior, which changes. In CTDE, the **centralized** Q(s, a_1,...,a_n) is learned with full information, so the "environment" for this Q is the true MDP (fixed transition). The individual policies are then updated using this stable Q (e.g. by gradient or by factorizing Q_tot). So the learning signal is more stable.
- **Scalability:** At test time we do not need to gather all observations or broadcast actions; each agent runs independently. This is important for distributed systems and low-latency deployment.
- Write 1–2 short paragraphs with one concrete algorithm (e.g. QMIX or MADDPG) and a small task (e.g. cooperative navigation).

**Common pitfalls**

- **Confusing training and execution:** Be explicit: "during training we use X; during execution we use Y." CTDE specifically means centralized only at train time.
- **Assuming full observability at execution:** In CTDE, execution is decentralized—each agent may have only partial observation. The centralized part (e.g. Q_tot) is not used at execution.
- **Overclaiming:** CTDE helps with non-stationarity and credit assignment but does not solve all MARL challenges (e.g. exploration, coordination).

{{< collapse summary="Worked solution (warm-up: CTDE)" >}}
**Key idea:** CTDE: during training we have access to global state (or all agents’ observations) so we can learn a centralized critic \\(Q(s, a_1, \\ldots, a_n)\\) or value \\(V(s)\\). This addresses non-stationarity (we condition on everyone’s action) and credit assignment (we see the joint effect). At execution we only need \\(\\pi_i(a_i | o_i)\\) for each agent (decentralized). QMIX, VDN, and COMA use this idea.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why can we use global state during training but not during execution? What would break if we required global state at execution?
2. **Coding:** Implement a minimal CTDE setup: 2 agents on a small grid, joint reward. Centralized critic: Q(s_global, a_1, a_2). Each agent has policy π_i(o_i). Train the critic with TD using (s, a_1, a_2); train each π_i with policy gradient using the critic (e.g. gradient of Q w.r.t. a_i). At test time, run only π_1(o_1) and π_2(o_2). Report train vs test return.
3. **Challenge:** In the same setup, compare **IQL** (each agent has Q_i(o_i, a_i), no central critic) with **CTDE** (centralized Q, decentralized π). Use the same env and same number of steps. Which converges faster and to a higher return?
4. **Variant:** Scale the number of agents from 2 to 5. How does the size of the centralized critic's input space grow? At what number of agents does the centralized critic become a bottleneck, and what factored architectures (e.g. QMIX, MAPPO) address this?
5. **Debug:** A CTDE implementation trains successfully but evaluation return is much lower than training return. The centralized critic uses global state during training, but the policies were accidentally also conditioned on global state (not just local observations) during training. Explain why this causes the train-test gap and how to verify that policies only use local observations.
6. **Conceptual:** CTDE allows a centralized critic during training to reduce variance in policy gradient estimates. Explain intuitively why a centralized critic has lower variance than a decentralized one. What information does the centralized critic have that individual critics miss?
