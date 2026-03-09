---
title: "Chapter 21: Linear Function Approximation"
description: "Linear FA with tile coding for MountainCar; semi-gradient SARSA."
date: 2026-03-10T00:00:00Z
weight: 21
draft: false
---

**Learning objectives**

- Represent the action-value function as \\(Q(s,a;w) = w^T \\phi(s,a)\\) with a feature vector \\(\\phi\\).
- Use **tile coding** (overlapping grid tilings) to produce binary features for continuous state (e.g. MountainCar).
- Implement **semi-gradient SARSA**: update \\(w\\) using the TD target with current \\(Q\\) for the next state.

**Concept and real-world RL**

**Linear function approximation** approximates \\(Q(s,a) \\approx w^T \\phi(s,a)\\). The weights \\(w\\) are learned from data; \\(\\phi(s,a)\\) is a fixed or hand-designed feature. **Tile coding** partitions the state space into overlapping tilings; each tiling is a grid, and the feature vector has a 1 for each tile that contains the state (and the action), so we get a sparse binary vector. This allows generalization across similar states. **Semi-gradient** methods use the TD target but treat the next-state value as a constant when taking the gradient (no backprop through the target). Linear FA is the simplest form of value approximation and appears in legacy RL and as a baseline.

**Exercise:** Implement linear function approximation for the MountainCar environment. Use tile coding (e.g., from Sutton's code) to create binary features. Train a semi-gradient SARSA to learn a policy. Plot the learning curve.

**Professor's hints**

- MountainCar state is (position, velocity). Use 2D tile coding: several tilings (e.g. 8), each with a grid over position and velocity. For each (s,a), form \\(\\phi(s,a)\\) by stacking one-hot (or binary) tile indices for each tiling; for action, either separate tilings per action or append action index to the state. Result: a long binary vector, one weight per feature.
- Semi-gradient SARSA: \\(\\delta = r + \\gamma Q(s',a';w) - Q(s,a;w)\\), then \\(w \\leftarrow w + \\alpha \\delta \\nabla_w Q(s,a;w)\\). For linear \\(Q = w^T \\phi\\), \\(\\nabla_w Q = \\phi(s,a)\\). So \\(w \\leftarrow w + \\alpha \\delta \\phi(s,a)\\).
- Learning curve: plot total reward per episode vs episode. MountainCar is hard; you may need many episodes and careful step size.

**Common pitfalls**

- **Gradient through target:** Do not backprop through \\(Q(s',a';w)\\) when computing the gradient. The target \\(r + \\gamma Q(s',a')\\) is treated as a constant; only \\(Q(s,a)\\) is differentiated.
- **Feature scale:** Tile coding gives binary features; if you use raw state as features, normalize (e.g. by state bounds) so that step size \\(\\alpha\\) is meaningful.
- **Exploration:** MountainCar needs exploration (e.g. \\(\\epsilon\\)-greedy). Without it, the agent may never reach the goal and never get the positive reward.

**Extra practice**

1. **Warm-up:** For a 1D state in [0,1] with one tiling of 4 tiles, what is \\(\\phi(0.25)\\)? (One tile is "on"; the rest 0. So a 4-dim vector with one 1.)
2. **Challenge:** Replace tile coding with **radial basis functions** (RBFs): \\(\\phi_i(s) = \\exp(-\\|s - c_i\\|^2 / (2\\sigma^2))\\) for a grid of centers \\(c_i\\). Compare learning speed with tile coding on MountainCar.
