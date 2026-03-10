---
title: "Chapter 72: Conservative Q-Learning (CQL)"
description: "CQL loss penalizing Q for OOD actions; compare with naive SAC."
date: 2026-03-10T00:00:00Z
weight: 72
draft: false
tags: ["CQL", "Conservative Q-Learning", "OOD", "offline RL", "curriculum"]
keywords: ["CQL", "Conservative Q-Learning", "OOD actions", "offline RL", "SAC"]
---

**Learning objectives**

- **Implement** the CQL loss: add a term that **penalizes** Q-values for actions drawn from the **current policy** (or a uniform distribution) so that Q is lower for out-of-distribution actions.
- **Apply** CQL to the offline dataset from Chapter 71 and train an offline SAC (or similar) with the CQL regularizer.
- **Compare** the learned policy's evaluation return and Q-values with naive SAC on the same dataset.
- **Explain** why penalizing Q for OOD actions helps avoid overestimation and improves offline performance.
- **Relate** CQL to **recommendation** and **healthcare** where we must learn from fixed logs without overestimating unseen actions.

**Concept and real-world RL**

**Conservative Q-Learning (CQL)** modifies the Q-learning objective so that Q-values for **out-of-distribution (OOD)** actions are **penalized** (pushed down). The typical formulation adds a term that increases the Q-loss when Q(s, a) is large for actions a sampled from the current policy (or a uniform distribution), while keeping Q(s, a) accurate for (s, a) in the dataset. This reduces overestimation for actions the agent would take but that are under-represented in the data. In **recommendation** and **healthcare**, we need to learn from historical data without recommending or prescribing actions that the data does not support; CQL-style conservatism is one approach.

**Where you see this in practice:** CQL and related offline RL algorithms (e.g. BRAC, TD3+BC); safe policy learning from logs.

**Exercise:** Implement the CQL loss by adding a term that penalizes Q-values for out-of-distribution actions. Apply it to the offline dataset from Chapter 71 and compare with naive SAC.

**Professor's hints**

- **CQL term:** E_s[ log sum_a exp(Q(s,a)) - E_a~π [ Q(s,a) ] ] or similar: encourage Q to be lower for policy actions relative to a log-sum-exp over actions. Often implemented as: add to critic loss α * (E[Q(s,a_π)] - E[Q(s,a_data)]), so we penalize Q when it is high for policy samples and reward (reduce loss) when Q is high for data actions. Check the CQL paper for the exact form; a simple version is to add α * mean(Q(s, a_random)) where a_random is from the current policy.
- **α (regularization weight):** Tune α; too large and Q is too conservative (underestimate), too small and overestimation remains. Start with α=0.1–1.0 and sweep.
- Use the **same** offline dataset as in Chapter 71 (random policy on Hopper) so you can directly compare CQL vs naive SAC evaluation return.
- Keep the rest of SAC (actor loss, target network, etc.) unchanged; only modify the critic loss.

**Common pitfalls**

- **Wrong sign of the penalty:** CQL should **lower** Q for OOD actions; double-check that your added term increases the loss when Q(s, π(s)) is large, so that the gradient step reduces Q for those actions.
- **Over-regularization:** If α is too large, Q becomes too small everywhere and the policy may become too conservative (e.g. only choose actions that appear very often in the data). Monitor both Q and evaluation return.
- **Sampling OOD actions:** You need to sample actions from the current policy (or a uniform distribution over actions) for the penalty; use the actor to generate a for each s in the batch.

{{< collapse summary="Worked solution (warm-up: CQL)" >}}
**Key idea:** CQL (Conservative Q-Learning) adds a regularizer that *lower*s Q-values for actions not in the dataset (e.g. sample \\(a\\) from the current policy and minimize \\(Q(s,a)\\)). So the learned Q is conservative: it is high only for (s,a) pairs that appear in the data. The policy then prefers in-distribution actions and avoids overestimated OOD actions. This stabilizes offline RL.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does penalizing Q(s, π(s)) during training help when the data was collected by a different policy?
2. **Coding:** Implement CQL with a simple penalty: loss += α * mean(Q(s, a_π)) where a_π ~ π(·|s). Train on the Chapter 71 offline Hopper dataset. Plot evaluation return vs α ∈ {0, 0.01, 0.1, 1.0}. Which α works best?
3. **Challenge:** Implement the full CQL loss from the paper (with log-sum-exp and data expectation). Compare sample efficiency and final return with your simplified penalty.
