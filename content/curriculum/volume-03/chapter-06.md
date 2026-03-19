---
title: "Chapter 26: Double DQN (DDQN)"
description: "Double DQN: online selects, target evaluates; compare with DQN."
date: 2026-03-10T00:00:00Z
weight: 26
draft: false
tags: ["Double DQN", "DDQN", "DQN", "overestimation", "curriculum"]
keywords: ["Double DQN", "DDQN", "overestimation", "target network"]
---

**Learning objectives**

- Implement **Double DQN**: use the online network to choose \\(a^* = \\arg\\max_a Q_{online}(s',a)\\), then use \\(Q_{target}(s', a^*)\\) as the TD target (instead of \\(\\max_a Q_{target}(s',a)\\)).
- Understand why this reduces overestimation of Q-values (max of estimates is biased high).
- Compare average Q-values and reward curves with standard DQN on CartPole.

**Concept and real-world RL**

Standard DQN uses \\(y = r + \\gamma \\max_{a'} Q_{target}(s',a')\\). The max over noisy estimates is biased upward (overestimation), which can hurt learning. **Double DQN** decouples action selection from evaluation: the *online* network selects \\(a^*\\), the *target* network evaluates \\(Q_{target}(s', a^*)\\). This reduces overestimation and often improves stability and final performance. It is a small code change and is commonly used in modern DQN variants (e.g. Rainbow).

**Illustration (overestimation):** Standard DQN's max over Q-values tends to overestimate; Double DQN often yields lower, more accurate Q-values. The chart below compares mean Q(s,a) after training (CartPole).

{{< chart type="bar" palette="comparison" title="Mean Q(s,a) after training (CartPole)" labels="DQN, Double DQN" data="28, 22" yLabel="Mean Q(s,a)" >}}

**Exercise:** Modify your DQN to use Double DQN: use the online network to select actions and the target network to evaluate them. Compare the average Q-values and performance with standard DQN on CartPole.

**Professor's hints**

- For each transition in the batch: compute \\(a^* = \\arg\\max_a Q_{online}(s', a)\\) (no grad; use `.detach()` or no_grad). Then \\(y = r + \\gamma (1 - \\text{done}) Q_{target}(s', a^*)\\). Use gather or indexing to get \\(Q_{target}(s', a^*)\\) for each sample in the batch.
- Logging: track the mean of \\(Q(s,a)\\) over the batch (or over a fixed set of states) for both DQN and DDQN. DDQN often has lower (less overestimated) Q-values and sometimes learns faster or more stably.
- Same hyperparameters (replay size, target update, \\(\\epsilon\\)) for both; only the target computation changes.

**Common pitfalls**

- **Using target for both select and eval:** That would be standard DQN. DDQN must use *online* for argmax and *target* for the value.
- **Gradient through a*:** When you compute \\(a^*\\) from the online network, do not backprop through that when computing the loss. The target \\(Q_{target}(s', a^*)\\) is a constant. Use `.detach()` on the target tensor.
- **Batch dimension:** For a batch of (s, a, r, s', done), you need \\(a^*\\) and \\(Q_{target}(s', a^*)\\) per sample. Use `torch.gather` or index: `Q_target(s').gather(1, a_star.unsqueeze(1)).squeeze(1)`.

{{< collapse summary="Worked solution (warm-up: why max Q overestimates)" >}}
**Warm-up:** In one sentence, why does \\(\\max_a Q(s,a)\\) overestimate the true value when \\(Q\\) is noisy? **Answer:** The max of noisy unbiased estimates is biased upward: even if each \\(Q(s,a)\\) is unbiased, the one that happens to be largest in a batch is likely to have positive noise, so \\(\\max_a Q(s,a) \\geq \\) true \\(Q^*\\) on average. Double DQN reduces this by using the online network to select the action and the target network to evaluate it, decorrelating the selection from the evaluation.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does \\(\\max_a Q(s,a)\\) overestimate the true value when \\(Q\\) is noisy?
2. **Coding:** Modify your DQN to Double DQN: use online network for argmax action, target network for Q(s', a*). Compare Q-values (mean over batch) after 10k steps with standard DQN on CartPole.
3. **Challenge:** Log the *max* Q-value per batch (before and after training) for DQN vs DDQN. Does DDQN's max Q stay more moderate? Relate to overestimation.
4. **Variant:** Apply Double DQN to LunarLander (a more challenging env). Is overestimation more pronounced there than on CartPole? Measure mean Q-values for both.
5. **Debug:** The code below uses the target network for both action selection and evaluation (standard DQN, not Double DQN). Fix it to use the online network for selection.

{{< pyrepl code="import torch\n\ndef compute_target_dqn(online_net, target_net, s_next, r, done, gamma=0.9):\n    with torch.no_grad():\n        # BUG: uses target_net for action selection (standard DQN)\n        q_next = target_net(s_next)  # should be online_net for DDQN\n        a_star = q_next.argmax(dim=1)  # action selection from target (wrong)\n        q_target = target_net(s_next).gather(1, a_star.unsqueeze(1)).squeeze(1)\n    return r + gamma * (1 - done) * q_target\n\n# Fix: use online_net for a_star, target_net only for evaluation\ndef compute_target_ddqn(online_net, target_net, s_next, r, done, gamma=0.9):\n    with torch.no_grad():\n        a_star = online_net(s_next).argmax(dim=1)  # FIXED: online selects\n        q_target = target_net(s_next).gather(1, a_star.unsqueeze(1)).squeeze(1)\n    return r + gamma * (1 - done) * q_target\n\nprint('DDQN: online selects action, target evaluates Q(s\\', a*)')" height="260" >}}

6. **Conceptual:** Why does using the *same* network for both action selection and evaluation in the TD target lead to overestimation bias?
7. **Recall:** State the Double DQN target formula \\(y = r + \\gamma Q_{target}(s', \\arg\\max_a Q_{online}(s',a))\\) from memory.
