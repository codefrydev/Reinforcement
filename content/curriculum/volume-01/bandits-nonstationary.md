---
title: "Bandits: Nonstationary"
description: "When reward distributions change over time—exponential recency-weighted average and constant step size."
date: 2026-03-10T00:00:00Z
weight: 6
draft: false
tags: ["bandits", "nonstationary", "step size", "recency", "curriculum"]
keywords: ["nonstationary bandits", "running average", "step size", "recency weighting"]
---

**Learning objectives**

- Understand why a plain sample mean is bad when reward distributions change over time.
- Use exponential recency-weighted average (constant step size) for nonstationary bandits.
- Implement and compare fixed step size vs. sample mean on a drifting testbed.

## Theory

In **nonstationary** bandits, the expected reward of each arm can change over time. The **sample mean** update \\(\bar{Q}_{n+1} = \bar{Q}_n + \frac{1}{n+1}(r - \bar{Q}_n)\\) gives equal weight to all past rewards, so old data can dominate and the agent is slow to adapt.

**Exponential recency-weighted average:** Use a constant step size \\(\alpha \in (0, 1]\\):

\[
Q_{n+1}(a) = Q_n(a) + \alpha \bigl( r - Q_n(a) \bigr)
\]

This gives more weight to recent rewards. Equivalently, \\(Q_n(a)\\) is a weighted average of past rewards with weights that decay exponentially backward in time. So the agent adapts to changes in the environment.

**Choosing \\(\alpha\\):** Larger \\(\alpha\\) adapts faster but is noisier; smaller \\(\alpha\\) is smoother but slower to react. For truly nonstationary problems, a constant \\(\alpha\\) (e.g. 0.1) is often better than \\(1/n\\) step sizes.

## Bandit summary, real data, and online learning

- **Stationary:** Sample mean or small constant \\(\alpha\\) both work.
- **Nonstationary:** Use constant \\(\alpha\\) (or other recency weighting).
- **Real data:** Often nonstationary (user preferences drift, ad click rates change). Online learning with step sizes is standard.
- **Summary:** Epsilon-greedy, UCB1, and Thompson Sampling can all be combined with constant step size when the environment is nonstationary.

## Beginner's exercise prompt

Implement a **nonstationary** 10-armed testbed: at each step, add a small random drift to each arm’s mean (e.g. \\(\mu_a \leftarrow \mu_a + \mathcal{N}(0, 0.01)\\)). Run an epsilon-greedy agent with (1) sample mean update \\(1/n\\), and (2) constant step size \\(\alpha = 0.1\\). Plot average reward over time. You should see that constant \\(\alpha\\) tracks the changing best arm better.

## Code sketch

- For constant step size: \\(Q(a) \leftarrow Q(a) + \alpha (r - Q(a))\\) after each pull of arm \\(a\\).
- For the nonstationary environment: after each step, add Gaussian noise to each arm’s true mean (and optionally clip to a range).

See [Chapter 2: Multi-Armed Bandits](chapter-02/) for the stationary testbed and [Bandits: Why not use a library?](bandits-why-not-library/) for when to code from scratch vs. use a library.
