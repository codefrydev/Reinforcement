---
title: "Checkpoint: Volume 3, Midpoint (After Chapter 25)"
description: "5 quick questions after Chapters 21–25 of Volume 3. Check you're ready to continue."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["checkpoint", "volume 3", "assessment", "function approximation", "DQN", "neural networks"]
weight: 11
roadmap_icon: "book"
roadmap_color: "indigo"
roadmap_phase_label: "Vol 3 Mid-Point"
---

Take this checkpoint after completing Chapters 21–25 (linear function approximation, neural networks for RL, DQN basics). All 5 should feel manageable — if any are unclear, re-read the relevant chapter before continuing.

---

**Q1.** In linear function approximation, V(s) = **w · φ(s)**. What are **w** and **φ(s)**?

{{< collapse summary="Answer" >}}
- **φ(s)** (phi of s) is the **feature vector** — a fixed representation of state s as a vector of numbers. For example, φ(s) might be tile-coded features, polynomial features, or hand-crafted state descriptors. It converts a raw state into a fixed-length vector.
- **w** is the **weight vector** — the learnable parameters. It has the same dimension as φ(s). The dot product w · φ(s) = Σ_i w_i φ_i(s) gives the estimated value.

The key insight: instead of storing one number per state (tabular), we learn d weights that generalize across all states via their features.
{{< /collapse >}}

---

**Q2.** Write the semi-gradient TD update rule for linear function approximation.

{{< collapse summary="Answer" >}}
**w ← w + α δ_t ∇̂ V(S_t, w)**

Where:
- **δ_t** = R_{t+1} + γ V(S_{t+1}, w) − V(S_t, w) is the TD error,
- **∇̂ V(S_t, w)** is the gradient of the value estimate with respect to w (for linear FA, this equals φ(S_t)),
- The update is "semi-gradient" because the TD target treats V(S_{t+1}, w) as a fixed constant — the gradient is only taken through V(S_t, w), not through the target.

For linear FA specifically: **w ← w + α δ_t φ(S_t)**.
{{< /collapse >}}

---

**Q3.** What is the purpose of a replay buffer in DQN?

{{< collapse summary="Answer" >}}
The replay buffer (experience replay) stores past transitions (s, a, r, s') and allows the agent to **train on random mini-batches** drawn from this buffer rather than using only the most recent transition.

This serves two purposes:
1. **Breaks temporal correlations**: consecutive transitions are highly correlated; training on them in sequence causes unstable learning. Random sampling decorrelates the training data.
2. **Data efficiency**: each experience can be replayed multiple times, making better use of costly environment interactions.

Without a replay buffer, the neural network would overfit to recent experiences and forget earlier ones.
{{< /collapse >}}

---

**Q4.** What is the purpose of a target network in DQN?

{{< collapse summary="Answer" >}}
The target network is a **periodically-frozen copy of the Q-network** used to compute the TD targets during training.

The TD target is: r + γ max_{a'} Q̂(s', a'; w⁻), where w⁻ are the frozen target network weights.

Without a target network, the TD target changes every update step because it depends on the same network being trained. This creates a moving target problem — chasing a constantly-shifting goal causes instability and divergence. By freezing the target network for many steps (e.g. every 10,000 steps), the targets are stable enough for learning to converge.
{{< /collapse >}}

---

**Q5.** Why can't we just use a lookup table for Atari games instead of a neural network?

{{< collapse summary="Answer" >}}
Atari game frames are 84×84 pixel images with 256 grayscale values. Even with frame stacking (4 frames), the state space has roughly **256^(84×84×4) ≈ 10^170,000** possible states — a number incomprehensibly larger than the number of atoms in the observable universe.

A lookup table requires one entry per state. This is:
- **Computationally impossible** to store or iterate over.
- **Never generalizes**: states seen once are not connected to visually similar states.

A neural network, by contrast, **generalizes** across similar pixel inputs — nearby states in pixel space share features and similar learned values. It compresses the value function into millions of weights that interpolate sensibly across the vast state space.
{{< /collapse >}}

---

All 5 correct? Continue to Chapter 26 (DQN extensions). Stuck on 2 or more? Re-read Chapters 22–24.
