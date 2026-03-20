---
title: "Chapter 21: Linear Function Approximation"
description: "Linear FA with tile coding for MountainCar; semi-gradient SARSA."
date: 2026-03-10T00:00:00Z
weight: 21
draft: false
difficulty: 7
tags: ["function approximation", "linear FA", "tile coding", "MountainCar", "curriculum"]
keywords: ["linear function approximation", "tile coding", "semi-gradient SARSA", "MountainCar"]
roadmap_color: "green"
roadmap_icon: "chart"
roadmap_phase_label: "Vol 3 · Ch 1"
---

**Learning objectives**

- Represent the action-value function as \\(Q(s,a;w) = w^T \\phi(s,a)\\) with a feature vector \\(\\phi\\).
- Use **tile coding** (overlapping grid tilings) to produce binary features for continuous state (e.g. MountainCar).
- Implement **semi-gradient SARSA**: update \\(w\\) using the TD target with current \\(Q\\) for the next state.

**Concept and real-world RL**

**Linear function approximation** approximates \\(Q(s,a) \\approx w^T \\phi(s,a)\\). The weights \\(w\\) are learned from data; \\(\\phi(s,a)\\) is a fixed or hand-designed feature. **Tile coding** partitions the state space into overlapping tilings; each tiling is a grid, and the feature vector has a 1 for each tile that contains the state (and the action), so we get a sparse binary vector. This allows generalization across similar states. **Semi-gradient** methods use the TD target but treat the next-state value as a constant when taking the gradient (no backprop through the target). Linear FA is the simplest form of value approximation and appears in legacy RL and as a baseline.

**Illustration (learning curve):** With tile coding and semi-gradient SARSA on MountainCar, total reward per episode typically improves over many episodes. The chart below shows a typical trend (reward per episode, smoothed).

{{< chart type="line" palette="return" title="Episode return (MountainCar, semi-gradient SARSA)" labels="0, 200, 400, 600, 800, 1000" data="-500, -350, -200, -120, -80, -65" xLabel="Episode" yLabel="Return" >}}

**Exercise:** Implement linear function approximation for the MountainCar environment. Use tile coding (e.g., from Sutton's code) to create binary features. Train a semi-gradient SARSA to learn a policy. Plot the learning curve.

**Professor's hints**

- MountainCar state is (position, velocity). Use 2D tile coding: several tilings (e.g. 8), each with a grid over position and velocity. For each (s,a), form \\(\\phi(s,a)\\) by stacking one-hot (or binary) tile indices for each tiling; for action, either separate tilings per action or append action index to the state. Result: a long binary vector, one weight per feature.
- Semi-gradient SARSA: \\(\\delta = r + \\gamma Q(s',a';w) - Q(s,a;w)\\), then \\(w \\leftarrow w + \\alpha \\delta \\nabla_w Q(s,a;w)\\). For linear \\(Q = w^T \\phi\\), \\(\\nabla_w Q = \\phi(s,a)\\). So \\(w \\leftarrow w + \\alpha \\delta \\phi(s,a)\\).
- Learning curve: plot total reward per episode vs episode. MountainCar is hard; you may need many episodes and careful step size.

**Common pitfalls**

- **Gradient through target:** Do not backprop through \\(Q(s',a';w)\\) when computing the gradient. The target \\(r + \\gamma Q(s',a')\\) is treated as a constant; only \\(Q(s,a)\\) is differentiated.
- **Feature scale:** Tile coding gives binary features; if you use raw state as features, normalize (e.g. by state bounds) so that step size \\(\\alpha\\) is meaningful.
- **Exploration:** MountainCar needs exploration (e.g. \\(\\epsilon\\)-greedy). Without it, the agent may never reach the goal and never get the positive reward.

{{< collapse summary="Worked solution (warm-up: tile coding φ(0.25))" >}}
**Warm-up:** For 1D state in [0,1] with one tiling of 4 tiles, what is \\(\\phi(0.25)\\)? **Answer:** With 4 tiles over [0,1], each tile covers an interval of length 0.25. So tile 0 covers [0, 0.25), tile 1 [0.25, 0.5), etc. The state 0.25 lies in tile 1 (or the boundary). So \\(\\phi(0.25)\\) is a 4-dim vector with a 1 in the position corresponding to that tile and 0 elsewhere, e.g. \\([0, 1, 0, 0]^T\\). This binary feature is used in linear \\(V(s) = w^T \\phi(s)\\) for value approximation.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a 1D state in [0,1] with one tiling of 4 tiles, what is \\(\\phi(0.25)\\)? (One tile is "on"; the rest 0. So a 4-dim vector with one 1.)
2. **Coding:** Implement tile coding for a 2D state (e.g. position in [0,1]×[0,1]) with 4 tilings and 8 tiles per dimension. Return the feature vector for (0.5, 0.5). What is the total number of features?
3. **Challenge:** Replace tile coding with **radial basis functions** (RBFs): \\(\\phi_i(s) = \\exp(-\\|s - c_i\\|^2 / (2\\sigma^2))\\) for a grid of centers \\(c_i\\). Compare learning speed with tile coding on MountainCar.
4. **Variant:** Try 4 tilings and 16 tilings on MountainCar. Does more tilings improve performance? What is the trade-off in feature vector size and memory?
5. **Debug:** The code below computes the gradient through the target \\(Q(s',a';w)\\) when updating \\(w\\), violating semi-gradient. Fix it by treating the target as a constant.

{{< pyrepl code="import numpy as np\n\ndef semi_gradient_sarsa_update(w, phi_s, phi_s_next, r, alpha=0.01, gamma=0.9):\n    Q_sa = w @ phi_s\n    Q_s_next = w @ phi_s_next\n    # BUG: delta uses w for both terms -> gradient flows through target\n    delta = r + gamma * Q_s_next - Q_sa\n    # Should NOT update w toward Q_s_next gradient\n    # Fix: treat Q_s_next as a constant (already correct in delta,\n    # but gradient only applies to phi_s)\n    w += alpha * delta * phi_s  # correct: gradient is phi_s only\n    return w\n\nphi = np.array([1.0, 0.0, 0.0, 0.0])\nw = np.zeros(4)\nw = semi_gradient_sarsa_update(w, phi, phi, 1.0)\nprint('Updated w:', w[:2])" height="240" >}}

6. **Conceptual:** What is the "deadly triad" in reinforcement learning with function approximation? Name the three components.
7. **Recall:** Write the semi-gradient SARSA weight update \\(w \\leftarrow w + \\alpha \\delta \\nabla_w Q(s,a;w)\\) and state what \\(\\nabla_w Q\\) equals for linear approximation.
