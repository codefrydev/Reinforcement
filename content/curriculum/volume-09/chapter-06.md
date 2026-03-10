---
title: "Chapter 86: Value Decomposition Networks (VDN)"
description: "VDN: sum individual Q to joint Q; compare with IQL."
date: 2026-03-10T00:00:00Z
weight: 86
draft: false
tags: ["VDN", "value decomposition", "multi-agent", "curriculum"]
keywords: ["VDN", "value decomposition", "multi-agent", "cooperative MARL"]
---

**Learning objectives**

- **Implement** **VDN**: for a cooperative game, define **joint Q** as the sum of **individual Q-values**: Q_tot(s, a_1,...,a_n) = Q_1(o_1, a_1) + ... + Q_n(o_n, a_n).
- **Train** with a **joint reward** (e.g. team reward): use TD on Q_tot so that the sum of individual Qs approximates the joint return; backprop distributes the gradient to each Q_i.
- **Compare** VDN with **IQL** (each agent trains Q_i on local reward or team reward without factorization) in terms of learning speed and final return.
- **Explain** the limitation of VDN: additivity may not hold for all tasks (e.g. when there are strong synergies or redundancies between agents).
- **Relate** VDN to **game AI** (team games) and **robot navigation** (multi-robot coordination).

**Concept and real-world RL**

**Value Decomposition Networks (VDN)** learn a **joint Q-value** for cooperative multi-agent settings by **summing** individual Q-values: Q_tot = Σ_i Q_i(o_i, a_i). The joint reward (e.g. team reward) is used to train Q_tot with standard TD; the gradient flows to each Q_i. This provides a **credit assignment** mechanism (each Q_i gets a share of the joint signal) while keeping execution **decentralized** (each agent chooses a_i = argmax Q_i(o_i, a_i)). VDN assumes the joint Q can be additively decomposed; when that holds, it often learns faster than IQL. In **game AI** and **robot navigation**, VDN is a simple baseline for cooperative MARL.

**Where you see this in practice:** VDN and QMIX (which generalizes VDN); value decomposition in StarCraft and similar benchmarks.

**Illustration (VDN vs IQL):** VDN sums individual Q-values for the joint Q; with joint reward it often converges faster than IQL. The chart below compares mean return over episodes.

{{< chart type="line" title="Mean return (VDN vs IQL, cooperative)" labels="0, 1k, 2k, 3k, 4k, 5k" data="10, 40, 75, 110, 140, 165" >}}

**Exercise:** Implement VDN for a cooperative game: sum individual Q-values to form a joint Q-value. Train with a joint reward. Compare with IQL.

**Professor's hints**

- **Individual Q_i:** Each agent i has a network Q_i(o_i, a_i). Input: observation (and maybe agent id). Output: scalar for each action (or one scalar for continuous action).
- **Joint Q_tot:** Q_tot = Q_1(o_1, a_1) + ... + Q_n(o_n, a_n). For TD: target = r_joint + γ max_{a'} Q_tot(s', a'_1,...,a'_n). The max over joint action is tractable if each agent's max is independent: max_{a'} Q_tot = Σ_i max_{a'_i} Q_i(o'_i, a'_i). So each agent can do argmax locally.
- **Training:** Sample batch (s, a_1,...,a_n, r, s'). Compute Q_tot = Σ_i Q_i(o_i, a_i). Target = r + γ Σ_i max_{a'_i} Q_i'(o'_i, a'_i). Loss = (Q_tot - target)^2. Backprop updates all Q_i.
- **IQL baseline:** Each agent trains Q_i with the same joint reward (each gets r_joint). No factorization; each does TD on Q_i(o_i, a_i) with target r_joint + γ max Q_i(o'_i, a'_i). Compare learning curves.

**Common pitfalls**

- **Greedy action selection:** For the TD target, use max over actions. With VDN, the max of the sum is the sum of maxes (per agent), so each agent can compute max_{a_i} Q_i(o'_i, a_i) independently. Do not use a joint max over (a_1,...,a_n) with a single network.
- **Exploration:** Both VDN and IQL need exploration (e.g. ε-greedy or noise on actions). Use the same exploration schedule when comparing.
- **Same reward:** Give the same joint reward to both methods so the comparison is fair.

{{< collapse summary="Worked solution (warm-up: QMIX)" >}}
**Key idea:** QMIX learns \\(Q_{tot}(s, u)\\) as a monotonic combination of per-agent \\(Q_i(o_i, u_i)\\): \\(Q_{tot} = f(Q_1, \\ldots, Q_n)\\) with \\(\\partial Q_{tot}/\\partial Q_i \\geq 0\\). So a greedy joint action is computed by each agent choosing greedy w.r.t. its \\(Q_i\\) (decentralized). The mixing network is trained so that \\(Q_{tot}\\) fits the true joint return; monotonicity ensures consistency between individual greedy and joint greedy.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does summing Q_i to get Q_tot allow each agent to choose its action independently at execution time (argmax a_i Q_i(o_i, a_i))?
2. **Coding:** Implement VDN and IQL on a 2-agent cooperative task (e.g. meet-up or a simple grid with joint goal). Train both for 5k episodes. Plot mean return vs episodes for both. Does VDN converge faster?
3. **Challenge:** Design a small cooperative task where the **optimal** joint Q is **not** additive (e.g. strong synergy: reward only when both agents take a specific joint action). Implement VDN anyway and see how much it underperforms; then try QMIX (next chapter) and compare.
