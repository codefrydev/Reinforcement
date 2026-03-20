---
title: "Math for reinforcement learning — probability, linear algebra, calculus"
description: "Probability, linear algebra, and calculus with reinforcement learning motivation. Practice questions and Professor's hints included."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["math for RL", "probability", "linear algebra", "calculus", "prerequisites"]
keywords: ["math for reinforcement learning", "probability linear algebra calculus", "RL prerequisites", "practice questions"]
roadmap_icon: "calculator"
roadmap_color: "green"
roadmap_phase_label: "Math for RL"
---

This track covers the **math you need to read and do reinforcement learning**: probability & statistics, linear algebra, and calculus. Each topic is tied to how it appears in RL (bandits, value functions, gradients). **Each topic page includes practice questions with full step-by-step solutions** (collapsible “Answer and explanation”) so you can check your work and see the derivation. Work through the pages in order, or use them to fill gaps after the [Preliminary assessment](../preliminary/).

**Recommended order:** [Probability & statistics](probability/) → [Linear algebra](linear-algebra/) → [Calculus](calculus/).

---

## Why this math matters in RL

- **Probability:** Rewards are often random; value functions are *expected* returns. Bandits, Monte Carlo methods, and policy evaluation all use expectations and sample averages.
- **Linear algebra:** States and observations are vectors; value functions are sometimes linear in a weight vector; neural networks are built from matrix-vector products and gradients.
- **Calculus:** Policy gradients and loss-based updates use derivatives and the chain rule. You do not need to derive everything by hand, but you need to understand what a gradient is and how it is used.

---

## Quick links

| Topic | Content | RL use |
|-------|---------|--------|
| [Probability & statistics](probability/) | Expectations, variance, sample mean, distributions, law of large numbers | Bandit rewards, MC returns, policy evaluation |
| [Linear algebra](linear-algebra/) | Vectors, dot product, matrices, gradients | State vectors, value parameterization, gradient updates |
| [Calculus](calculus/) | Derivatives, chain rule, partial derivatives | Policy gradient, loss gradients, backprop |

After finishing this track, take the **[Phase 1 self-check](../assessment/phase-1-math/)** (10 questions). If you pass, you are ready for Phase 2 and [Volume 1](../curriculum/volume-01/).
