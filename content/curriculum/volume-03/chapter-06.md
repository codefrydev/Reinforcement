---
title: "Chapter 26: Double DQN (DDQN)"
description: "Double DQN: online selects, target evaluates; compare with DQN."
date: 2026-03-10T00:00:00Z
weight: 26
draft: false
---

**Learning objectives**

- Implement **Double DQN**: use the online network to choose \\(a^* = \\arg\\max_a Q_{online}(s',a)\\), then use \\(Q_{target}(s', a^*)\\) as the TD target (instead of \\(\\max_a Q_{target}(s',a)\\)).
- Understand why this reduces overestimation of Q-values (max of estimates is biased high).
- Compare average Q-values and reward curves with standard DQN on CartPole.

**Concept and real-world RL**

Standard DQN uses \\(y = r + \\gamma \\max_{a'} Q_{target}(s',a')\\). The max over noisy estimates is biased upward (overestimation), which can hurt learning. **Double DQN** decouples action selection from evaluation: the *online* network selects \\(a^*\\), the *target* network evaluates \\(Q_{target}(s', a^*)\\). This reduces overestimation and often improves stability and final performance. It is a small code change and is commonly used in modern DQN variants (e.g. Rainbow).

**Exercise:** Modify your DQN to use Double DQN: use the online network to select actions and the target network to evaluate them. Compare the average Q-values and performance with standard DQN on CartPole.

**Professor's hints**

- For each transition in the batch: compute \\(a^* = \\arg\\max_a Q_{online}(s', a)\\) (no grad; use `.detach()` or no_grad). Then \\(y = r + \\gamma (1 - \\text{done}) Q_{target}(s', a^*)\\). Use gather or indexing to get \\(Q_{target}(s', a^*)\\) for each sample in the batch.
- Logging: track the mean of \\(Q(s,a)\\) over the batch (or over a fixed set of states) for both DQN and DDQN. DDQN often has lower (less overestimated) Q-values and sometimes learns faster or more stably.
- Same hyperparameters (replay size, target update, \\(\\epsilon\\)) for both; only the target computation changes.

**Common pitfalls**

- **Using target for both select and eval:** That would be standard DQN. DDQN must use *online* for argmax and *target* for the value.
- **Gradient through a*:** When you compute \\(a^*\\) from the online network, do not backprop through that when computing the loss. The target \\(Q_{target}(s', a^*)\\) is a constant. Use `.detach()` on the target tensor.
- **Batch dimension:** For a batch of (s, a, r, s', done), you need \\(a^*\\) and \\(Q_{target}(s', a^*)\\) per sample. Use `torch.gather` or index: `Q_target(s').gather(1, a_star.unsqueeze(1)).squeeze(1)`.

**Extra practice**

1. **Warm-up:** In one sentence, why does \\(\\max_a Q(s,a)\\) overestimate the true value when \\(Q\\) is noisy?
2. **Challenge:** Log the *max* Q-value per batch (before and after training) for DQN vs DDQN. Does DDQN's max Q stay more moderate? Relate to overestimation.
