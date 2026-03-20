---
title: "Feature Engineering for Reinforcement Learning"
description: "Designing state and state-action features for linear value approximation."
date: 2026-03-10T00:00:00Z
weight: 21
draft: false
difficulty: 7
tags: ["feature engineering", "function approximation", "curriculum"]
keywords: ["features", "tile coding", "linear approximation", "RL"]
roadmap_color: "green"
roadmap_icon: "chart"
roadmap_phase_label: "Vol 3 · Feature Engineering"
---

**Learning objectives**

- Choose or design feature vectors \\(\phi(s)\\) or \\(\phi(s,a)\\) for linear \\(V(s) = w^T \phi(s)\\) or \\(Q(s,a) = w^T \phi(s,a)\\).
- Use tile coding, polynomial features, and normalization appropriately.
- Understand how feature choice affects generalization and learning speed.

## Why features matter

In **linear function approximation**, we approximate \\(V(s) \approx w^T \phi(s)\\) or \\(Q(s,a) \approx w^T \phi(s,a)\\). The **feature vector** \\(\phi\\) determines what the function can represent. Good features capture the right structure (e.g. similar states get similar values) and keep the dimension manageable so that learning is stable and sample-efficient.

## Common feature schemes

**Tile coding (multiple overlapping tilings):** Partition the state space into overlapping grids (tilings). Each tiling divides the space into cells; the feature vector has a 1 for each cell that contains the current state (and optionally the action). Multiple tilings with offsets provide overlap and smooth generalization. Binary, sparse; works well for bounded continuous state (e.g. position, velocity).

**Polynomial features:** For state \\(s = (s_1, \ldots, s_d)\\), use monomials like \\(1, s_1, s_2, s_1^2, s_1 s_2, \ldots\\). Simple but dimension grows quickly; normalization (e.g. scale \\(s\\) to \\([-1,1]\\)) helps.

**Radial basis functions (RBFs):** Place centers \\(c_i\\) in state space; \\(\phi_i(s) = \exp(-\|s - c_i\|^2 / (2\sigma^2))\\). Smooth generalization; choice of centers and \\(\sigma\\) matters.

**Raw state with normalization:** \\(\phi(s) = (s - \mu) / \sigma\\) or scale to a fixed range. Minimal engineering; works when the state is already meaningful and low-dimensional (e.g. CartPole’s 4D state). Combine with polynomial terms if needed.

## State-action features

For \\(Q(s,a)\\), either:

- **Separate features per action:** \\(\phi(s,a)\\) is zero everywhere except for a block of features corresponding to action \\(a\\); that block is \\(\phi(s)\\). So \\(Q(s,a) = w_a^T \phi(s)\\) with shared or separate \\(w_a\\).
- **Append action:** \\(\phi(s,a) = [\phi(s); \mathbf{1}_a]\\) where \\(\mathbf{1}_a\\) is one-hot for action \\(a\\). Then one weight vector \\(w\\): \\(Q(s,a) = w^T \phi(s,a)\\).

## Practical tips

- **Normalize:** Scale inputs so that feature magnitudes are comparable (e.g. 0–1 or zero mean, unit variance). Avoid huge or tiny values that make step size \\(\alpha\\) hard to tune.
- **Sparsity:** Tile coding and RBFs give sparse or local features; updates only touch a few weights. This can speed learning and improve generalization.
- **Domain knowledge:** Use problem structure (e.g. for CartPole, angle and angular velocity are critical; position may matter less for stability).

See [Chapter 21: Linear Function Approximation](chapter-01/) for tile coding and semi-gradient methods, and [CartPole](cartpole/) for an environment that benefits from good features.
