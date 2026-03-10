---
title: "Bandits: UCB1"
description: "Upper Confidence Bound (UCB1) algorithm for multi-armed bandits—balance exploration and exploitation using uncertainty."
date: 2026-03-10T00:00:00Z
weight: 4
draft: false
tags: ["bandits", "UCB1", "upper confidence bound", "exploration", "curriculum"]
keywords: ["UCB1", "UCB", "multi-armed bandits", "exploration exploitation"]
---

**Learning objectives**

- Understand the UCB1 action-selection rule and why it explores uncertain arms.
- Implement UCB1 on the 10-armed testbed and compare with epsilon-greedy.
- Interpret the exploration bonus \\(c \sqrt{\ln t / N(a)}\\).

## Theory

**UCB1** (Upper Confidence Bound) chooses the action that maximizes an *upper bound* on the expected reward:

\[
a_t = \arg\max_a \left[ Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]
\]

- \\(Q(a)\\) is the sample mean reward for arm \\(a\\).
- \\(N(a)\\) is how many times arm \\(a\\) has been pulled.
- \\(t\\) is the total number of pulls so far.
- \\(c\\) is a constant (e.g. 2) that controls exploration.

The term \\(c \sqrt{\ln t / N(a)}\\) is an **exploration bonus**: arms that have been pulled less often (small \\(N(a)\\)) get a higher bonus, so they are tried more. As \\(N(a)\\) grows, the bonus shrinks. So UCB1 explores **systematically** rather than randomly (unlike epsilon-greedy).

UCB1 has nice theoretical guarantees for finite-horizon regret under certain assumptions. In practice, it often performs well and is a good alternative to epsilon-greedy when you want deterministic, confidence-based exploration.

## Beginner's exercise prompt

Implement UCB1 for the 10-armed Gaussian testbed. Use \\(c = 2\\). Run for 1000 steps, average over many runs, and plot average reward over time. Compare with epsilon-greedy (\\(\epsilon = 0.1\\)) and greedy.

## Code sketch

- Maintain \\(Q(a)\\) (sample mean) and \\(N(a)\\) (pull count) for each arm.
- At step \\(t\\), choose \\(a = \arg\max_a \left[ Q(a) + c \sqrt{\ln(t+1) / (N(a) + 1)} \right]\\) (add 1 to avoid division by zero before any pull).
- After receiving reward \\(r\\): \\(N(a) \leftarrow N(a) + 1\\), \\(Q(a) \leftarrow Q(a) + \frac{1}{N(a)}(r - Q(a))\\).

**Common pitfall:** Ensure every arm is pulled at least once at the start (e.g. first 10 steps pull arms 0–9 in order), or use the \\(N(a)+1\\) in the denominator so untried arms get a very large bonus.

See [Chapter 2: Multi-Armed Bandits](chapter-02/) for the testbed and epsilon-greedy, and [Bandits: Thompson Sampling](bandits-thompson-sampling/) for a Bayesian alternative.
