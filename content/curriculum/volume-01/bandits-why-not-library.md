---
title: "Bandits: Why don't we just use a library?"
description: "When to implement bandits from scratch vs. use existing libraries—learning goals and control."
date: 2026-03-10T00:00:00Z
weight: 7
draft: false
difficulty: 6
tags: ["bandits", "pedagogy", "libraries", "curriculum"]
keywords: ["bandits", "library", "implement from scratch", "learning"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Bandits Why Not Library"
---

**Learning objectives**

- Understand why this curriculum has you implement bandits (and other algorithms) from scratch.
- Know when it is appropriate to switch to a library in practice.

## Why implement from scratch?

1. **Understanding:** Writing the update equations and selection rules yourself forces you to understand how they work. If you only call `library.solve()`, you may not know what step size, prior, or exploration rule is being used—or how to debug when things go wrong.

2. **Control:** In research or production, you often need to modify the algorithm (e.g. custom priors, nonstandard rewards, or integration with existing systems). Code you wrote is code you can change.

3. **Foundation:** Bandits are the simplest RL setting. Mastering them (epsilon-greedy, UCB1, Thompson Sampling, nonstationary) gives you intuition for exploration–exploitation and for incremental updates that appear everywhere in RL (Q-learning, policy gradients, etc.).

4. **Debugging:** When a library gives surprising results, you need to know the algorithm well enough to check assumptions, step sizes, and data flow. Having implemented it once makes that much easier.

## When to use a library

- **Production A/B tests or recommenders:** Once you understand the algorithms, using a well-tested library (e.g. Vowpal Wabbit, BanditLib, or framework-specific tools) saves time and reduces bugs.
- **Complex variants:** If you need contextual bandits, adversarial bandits, or heavy-duty Bayesian inference, a library can handle scaling and edge cases.
- **Speed and scale:** Libraries are often optimized and parallelized; use them when performance matters and you already know what the algorithm is doing.

**Summary:** Learn by implementing; then use libraries when they fit and you understand what they do. This course emphasizes the “learn by implementing” part so that later you can use libraries wisely.

See [Chapter 2: Multi-Armed Bandits](chapter-02/) and the other [Bandits](volume-01/) pages for the implementations you build first.
