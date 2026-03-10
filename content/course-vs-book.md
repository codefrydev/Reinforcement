---
title: "This Course vs. RL Book: What's the Difference?"
description: "How this curriculum relates to Sutton and Barto's Reinforcement Learning: An Introduction and other books."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["course", "book", "Sutton and Barto", "curriculum"]
keywords: ["RL book", "Sutton and Barto", "course comparison", "reinforcement learning"]
---

**Learning objectives**

- Understand how this curriculum aligns with (and extends beyond) the classic *Reinforcement Learning: An Introduction* (Sutton & Barto).
- Know when to use the course vs. the book for depth and exercises.

## This course vs. the RL book

**Sutton & Barto** (*Reinforcement Learning: An Introduction*, 2nd ed.) is the standard textbook for RL. It builds from bandits and MDPs through dynamic programming, Monte Carlo, temporal difference, and function approximation, with clear math and many examples (gridworld, blackjack, etc.). This **curriculum** follows a similar progression for the core topics (bandits → MDPs → DP → MC → TD → approximation) so that if you read the book alongside the course, the order matches.

**Differences:**

- **Exercises:** This curriculum provides one exercise per chapter with **worked solutions** (collapsible), so you can check your reasoning and code. The book has end-of-chapter exercises without full solutions in the text.
- **Scope:** The curriculum goes beyond the book’s first 10–11 chapters: it includes 100 chapters across 10 volumes (policy gradients, deep RL, model-based, exploration, offline RL, multi-agent, real-world, safety, LLMs). The book focuses on the foundational material; the course adds modern and advanced topics.
- **Code and environment:** The course assumes you implement algorithms and run them (e.g. Gym/Gymnasium, PyTorch). The book describes algorithms and sometimes gives pseudocode; the course ties theory to runnable code and specific environments (gridworld, blackjack, CartPole, etc.).
- **Pacing:** You can do the course at your own pace. The learning path (phases 0–5) and prerequisites help you decide what to do before diving into the curriculum.

## When to use which

- **For foundations (bandits, MDPs, DP, MC, TD, linear FA):** Use both. Read the book for the math and intuition; use the course for exercises, solutions, and code.
- **For deeper theory:** The book has more formal proofs and derivations. Use it when you want to understand *why* an algorithm converges or how the Bellman equations are derived.
- **For implementation and extensions:** Use the course: it points you to code walkthroughs, environments, and later volumes (DQN, policy gradients, etc.) that the book does not cover in the same detail.
- **For a project (e.g. stock trading):** The course provides a structured project (see [Stock Trading Project](../stock-trading/)) that applies Q-learning and approximation; the book gives the underlying theory.

**Summary:** The course and the book are complementary. Follow the [Course Outline](../course-outline/) for order; use the book for deeper reading on the same topics; use the course for exercises, solutions, code, and advanced volumes.
