---
title: "Chapter 28: Prioritized Experience Replay (PER)"
description: "Sum-tree prioritized buffer with TD error; importance-sampling weights."
date: 2026-03-10T00:00:00Z
weight: 28
draft: false
---

**Learning objectives**

- Implement **prioritized replay**: assign each transition a priority (e.g. TD error \\(|\\delta|\\)) and sample with probability proportional to \\(p_i^\\alpha\\).
- Use a **sum tree** (or a simpler alternative) for efficient sampling and priority updates.
- Apply **importance-sampling weights** \\(w_i = (N \\cdot P(i))^{-\\beta} / \\max_j w_j\\) to correct the bias introduced by non-uniform sampling.

**Concept and real-world RL**

**Prioritized Experience Replay (PER)** samples transitions with probability proportional to their "priority"—often the TD error—so that surprising or informative transitions are replayed more often. This can speed up learning but introduces bias (the update distribution is not the same as the uniform replay distribution). **Importance-sampling weights** correct for this by weighting the gradient update so that in expectation we recover the uniform case. A **sum tree** allows O(log N) sampling and priority update. PER is used in Rainbow and other sample-efficient DQN variants.

**Exercise:** Implement a prioritized replay buffer using a sum tree. Use TD error as priority. Sample according to \\(P(i) = p_i^\\alpha / \\sum p_j^\\alpha\\) and apply importance-sampling weights to correct bias. Integrate with DQN and test on a simple environment.

**Professor's hints**

- Sum tree: a binary tree where each leaf is a transition's priority and each node stores the sum of its children. To sample, draw \\(u \\sim \\text{Uniform}(0, \\text{total})\\) and traverse from root to leaf to find the leaf (transition) containing \\(u\\). When you update a priority, update the leaf and propagate the change up the tree.
- Priority: when a transition is first stored, use \\(|\\delta|\\) or \\(|\\delta| + \\epsilon\\) (to avoid zero priority). After a transition is replayed, recompute its TD error and update its priority in the tree.
- IS weights: \\(P(i) = p_i^\\alpha / \\sum_k p_k^\\alpha\\). Weight \\(w_i = (N \\cdot P(i))^{-\\beta}\\), then normalize by \\(w_i \\leftarrow w_i / \\max_k w_k\\). Multiply the TD error (or the loss) by \\(w_i\\) in the update. \\(\\beta\\) is annealed from a small value to 1 over training.

**Common pitfalls**

- **Forgetting IS weights:** Without them, the expected update is biased. Always multiply the loss (or gradient) by the IS weight for each sample in the batch.
- **Numerical stability:** Priorities must be positive. Use \\(|\\delta| + \\epsilon\\) (e.g. \\(10^{-6}\\)) and clamp very large priorities if needed.
- **Sum tree implementation:** If the full sum tree is complex, you can start with a simpler version: store priorities in an array and sample by proportional sampling (e.g. cumulative sum + binary search, or numpy.random.choice with p = priorities/sum). It is O(N) but correct for small buffers.

**Extra practice**

1. **Warm-up:** Why does sampling by TD error introduce bias? (We sample more often transitions we already think are "wrong"; the gradient is then biased toward those.)
2. **Challenge:** Skip the sum tree and implement PER with a list of (transition, priority) and proportional sampling (normalize priorities to get probabilities). Compare sample efficiency with uniform replay on CartPole. Add IS weights and compare again.
