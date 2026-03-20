---
title: "Volume 2 Review & Bridge to Volume 3"
description: "Review Volume 2 tabular methods and preview Volume 3. From Q-tables to neural network function approximation."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 6
weight: 100
tags: ["review", "bridge", "Volume 2", "Volume 3", "function approximation", "DQN", "tabular"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Review"
---

You have finished Volume 2. Before starting Volume 3, take this 10-minute review.

---

## Volume 2 Recap Quiz

{{< collapse summary="Q1. What is the TD error and why is it useful?" >}}
The **TD error** is δ_t = R_{t+1} + γ V(S_{t+1}) − V(S_t). It measures how much the current estimate V(S_t) differs from a one-step bootstrapped target. It is the signal used to update value estimates. When δ_t = 0 everywhere, the values are self-consistent (a fixed point of the Bellman operator).
{{< /collapse >}}

{{< collapse summary="Q2. What makes Q-learning off-policy?" >}}
Q-learning updates Q(s,a) toward the **maximum** next-state Q-value — the target policy is greedy. But the agent may be following a different behavior policy (e.g. ε-greedy) during experience collection. Because target ≠ behavior policy, Q-learning is off-policy. This lets it learn the optimal Q* even while exploring.
{{< /collapse >}}

{{< collapse summary="Q3. What is the key advantage of TD methods over Monte Carlo?" >}}
TD methods can **update online after every step** — they do not need to wait until the end of an episode. This makes them applicable to continuing tasks (no episode boundary) and much faster to learn in long-episode environments. Monte Carlo must wait for the full return, which can be very high variance in long episodes.
{{< /collapse >}}

{{< collapse summary="Q4. What is the tabular Q-table, and why does it break down for CartPole?" >}}
A tabular Q-table stores one Q(s,a) value per (state, action) pair in a dictionary or array. For a discrete gridworld with 9 states and 4 actions, the table has 36 entries — manageable.

For **CartPole**, the state is [cart position, cart velocity, pole angle, pole angular velocity] — four continuous values. Even coarsely discretizing each into 10 bins gives 10^4 = 10,000 states × 2 actions = 20,000 entries. More realistic discretizations (100 bins each) give 10^8 entries. The table explodes exponentially with state dimension (the **curse of dimensionality**).
{{< /collapse >}}

{{< collapse summary="Q5. What does it mean to 'generalize' in RL, and why can't tabular methods do it?" >}}
**Generalization**: learning that similar states have similar values, so experience with one state informs estimates for nearby states. Tabular methods treat every state independently — seeing state (1.05, ...) tells you nothing about (1.06, ...). Neural networks can generalize: shared weights mean that gradient updates to one input region affect similar inputs.
{{< /collapse >}}

---

## What Changes in Volume 3

| | Volume 2 (Tabular) | Volume 3 (Function Approximation) |
|---|---|---|
| **State representation** | Discrete index into table | Feature vector / raw pixels |
| **Value storage** | Q-table (one entry per state-action) | Neural network weights |
| **State space** | Small, discrete | Large, continuous, or image-based |
| **Generalization** | None — each state independent | Yes — similar inputs → similar outputs |
| **Key algorithms** | SARSA, Q-learning, n-step | Linear FA, DQN, Double DQN, Dueling DQN |
| **Key challenge** | Curse of dimensionality | Training stability (deadly triad) |

**The key insight:** Replace the Q-table Q(s,a) with a parametric function Q(s,a; θ) — a neural network. The weights θ are shared across all states, enabling generalization. The update rule becomes a gradient descent step instead of a table lookup.

---

## Bridge Exercise: From Q-table to Q-network

First, see how the Q-table explodes for continuous states:

{{< pyrepl code="import math\n\n# CartPole state: [x, v, theta, omega]\n# Each continuous dimension, discretized into n_bins bins\n\ndef table_size(n_dims, n_bins, n_actions):\n    return n_bins**n_dims * n_actions\n\nprint('CartPole Q-table size (4 dims):')\nfor bins in [5, 10, 20, 50, 100]:\n    size = table_size(4, bins, 2)\n    print(f'  {bins:3d} bins/dim -> {size:>12,} entries ({size*8/1e6:.1f} MB for float64)')\n\nprint()\nprint('Atari (84x84 grayscale, 4-frame stack):')\natari_states = 256**(84*84*4)  # astronomical\nprint(f'  States: ~10^{math.log10(float(256**(84*84*4))):.0f} (impossible to tabulate)')" height="240" >}}

Now see the neural network alternative:

{{< pyrepl code="# A tiny Q-network (linear for simplicity)\n# Instead of a table, we use: Q(s, a) = w_a . s + b_a\n\ndef linear_q_network(state, weights, bias):\n    \"\"\"weights: shape (n_actions, n_state_dims), bias: (n_actions,)\"\"\"\n    return [sum(w*s for w,s in zip(weights[a], state)) + bias[a]\n            for a in range(len(weights))]\n\n# Initialize tiny network for CartPole (4 inputs, 2 actions)\nimport random\nrandom.seed(42)\nn_state = 4\nn_actions = 2\nweights = [[random.gauss(0, 0.1) for _ in range(n_state)] for _ in range(n_actions)]\nbias = [0.0] * n_actions\n\n# Evaluate Q for a sample CartPole state\nstate = [0.05, 0.02, -0.01, 0.03]\nQ_values = linear_q_network(state, weights, bias)\nprint(f'State:    {state}')\nprint(f'Q-values: {[round(q, 4) for q in Q_values]}')\nprint(f'Greedy:   action {Q_values.index(max(Q_values))}')\nprint(f'\\nNetwork params: {n_state * n_actions + n_actions} (vs {10**4 * 2:,} table entries)')" height="280" >}}

{{< collapse summary="What changed" >}}
The Q-table has been replaced by a weight matrix. Instead of a lookup, we compute a dot product. The number of parameters is fixed (8 weights + 2 biases = 10), regardless of how many unique states the agent visits. Volume 3 extends this to deep networks (DQN) and adds techniques (replay buffer, target network) to make training stable.
{{< /collapse >}}

---

## Ready for Volume 3?

Before continuing, confirm:

- [ ] I can write the Q-learning and SARSA update rules from memory and explain the difference.
- [ ] I understand why the Q-table fails for CartPole (dimensionality argument).
- [ ] I understand the bridge exercise: fixed parameters instead of per-state entries.
- [ ] I know what "bootstrapping" means (using current estimates as targets).

**Next:** [Volume 3: Function Approximation & DQN](../volume-03/)
