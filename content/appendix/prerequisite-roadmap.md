---
title: "Machine Learning and AI Prerequisite Roadmap (pt 1–2)"
description: "What to learn before or alongside reinforcement learning—math, programming, and ML basics."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["FAQ", "prerequisites", "roadmap", "machine learning"]
keywords: ["prerequisite", "roadmap", "ML", "AI", "before RL"]
---

**Learning objectives**

- See the recommended order of topics before (or alongside) RL: math, programming, optional supervised learning.
- Know what this curriculum assumes and where to fill gaps.

## Prerequisite roadmap (overview)

**Pt 1 — Foundations**

1. **Programming:** Variables, types, conditionals, loops, functions, basic data structures (lists, dicts). Language: **Python**. If you have no programming, start with the [Learning path](../learning-path/) Phase 0 and [Prerequisites: Python](../prerequisites/python/).
2. **Probability and statistics:** Sample mean, variance, expectation, law of large numbers. Used in bandits, Monte Carlo, and value functions. See [Math for RL: Probability](../math-for-rl/probability/).
3. **Linear algebra:** Vectors, dot product, matrices, matrix-vector product. Used in value approximation \\(V(s) = w^T \phi(s)\\) and gradients. See [Math for RL: Linear algebra](../math-for-rl/linear-algebra/).
4. **Calculus:** Derivatives, chain rule, partial derivatives. Used in policy gradients and loss minimization. See [Math for RL: Calculus](../math-for-rl/calculus/).
5. **NumPy (and optionally Pandas, Matplotlib):** Arrays, indexing, random numbers, plotting. See [Prerequisites: NumPy](../prerequisites/numpy/), [Matplotlib](../prerequisites/matplotlib/), [Pandas](../prerequisites/pandas/).

**Pt 2 — Toward deep RL**

6. **PyTorch or TensorFlow:** Tensors, autograd, simple neural networks (forward pass, backward, optimizer). Needed for Volume 3+ (DQN, policy gradients). See [Prerequisites: PyTorch](../prerequisites/pytorch/) or [TensorFlow](../prerequisites/tensorflow/).
7. **Gym / Gymnasium:** Environments, `reset()`, `step()`, observation and reward. See [Prerequisites: Gym](../prerequisites/gym/).
8. **Optional—supervised learning:** Basic idea of loss, gradient descent, and overfitting. Helpful for understanding function approximation and DQN; not strictly required to start RL if you are comfortable with gradients and loss.

## Order of study

- **No programming / no math:** Phase 0 → Python prerequisite → Math for RL (probability, linear algebra, calculus) → Prerequisites (NumPy, etc.) → [Preliminary assessment](../preliminary/) → Volume 1.
- **Programming but weak math:** Math for RL → Prerequisites → Preliminary → Volume 1.
- **Math and programming, no RL:** Preliminary (to see what you know) → Volume 1 → Volume 2 → Volume 3+.
- **Some ML, want RL:** Volume 1 (quick if you know MDPs) → Volume 2 → Volume 3+.

Use the [Course outline](../../course-outline/) and [Learning path](../learning-path/) as the main map; this roadmap shows what to shore up before or in parallel.
