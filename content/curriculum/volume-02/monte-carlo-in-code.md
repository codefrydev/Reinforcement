---
title: "Monte Carlo in Code"
description: "Code walkthrough for Monte Carlo policy evaluation and Monte Carlo control, with and without exploring starts."
date: 2026-03-10T00:00:00Z
weight: 12
draft: false
difficulty: 6
tags: ["Monte Carlo", "code", "policy evaluation", "control", "curriculum"]
keywords: ["Monte Carlo", "first-visit", "MC control", "exploring starts", "code"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Monte Carlo In Code"
---

**Learning objectives**

- Implement first-visit Monte Carlo policy evaluation in code (returns, averaging).
- Implement Monte Carlo control (estimate Q, improve policy greedily).
- Implement MC control without exploring starts (e.g. epsilon-greedy behavior).

## Monte Carlo policy evaluation in code

**Setup:** You have an episodic environment (e.g. blackjack, gridworld) and a fixed policy \\(\pi\\). Goal: estimate \\(V^\\pi(s)\\).

**Algorithm:**

1. Run an episode: follow \\(\pi\\), collect \\((s_0, r_1, s_1, r_2, \ldots, r_T, s_T)\\).
2. For each step \\(t\\), compute the **return** from \\(t\\): \\(G_t = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{T-t-1} r_T\\) (or loop backward from the end).
3. **First-visit:** For each state \\(s\\) that appears in the episode, find the *first* time \\(t\\) with \\(s_t = s\\). Add \\(G_t\\) to a list (or running sum) for state \\(s\\); increment the count for \\(s\\).
4. After many episodes: \\(V(s) = \\) (sum of returns from first visits to \\(s\\)) / (count of first visits to \\(s\\)).

**Code sketch:** Use a dict `returns[s] = []` or `(total, count)`. In each episode, track which states have been seen; on first visit to \\(s\\) at step \\(t\\), append \\(G_t\\) (or add to total and increment count). At the end of all episodes, \\(V(s) = \\) mean of returns for \\(s\\).

## Monte Carlo control in code

**Goal:** Learn \\(Q^*(s,a)\\) and an optimal policy.

**MC control with exploring starts:**

1. Initialize \\(Q(s,a)\\) arbitrarily; policy \\(\pi\\) e.g. greedy w.r.t. \\(Q\\).
2. Loop: pick a *random* (s, a) pair to start an episode (exploring start). Run episode from that (s,a), then follow \\(\pi\\).
3. For each (s,a) visited, compute return from that (s,a) to end of episode. **First-visit:** for each (s,a) pair, use only the return from its first occurrence in the episode.
4. Update \\(Q(s,a) = \\) average of those returns (over episodes).
5. Improve policy: \\(\pi(s) = \arg\max_a Q(s,a)\\).
6. Repeat until convergence.

**Code:** Store returns per (s,a): `returns[(s,a)] = (sum, count)`. After each episode, for first-visit (s,a) add the return from that step; then set \\(Q(s,a) = \\) sum/count. Set \\(\pi(s) = \arg\max_a Q(s,a)\\).

## Monte Carlo control without exploring starts in code

Exploring starts are often unrealistic (we cannot choose the first state in many tasks). **Alternative:** use an **epsilon-greedy** policy: with probability \\(\epsilon\\) choose a random action, with probability \\(1-\epsilon\\) choose \\(\arg\max_a Q(s,a)\\). Run episodes by starting from a *normal* start state (e.g. environment reset). Use first-visit MC to update \\(Q(s,a)\\) from the returns. Then set \\(\pi\\) to epsilon-greedy w.r.t. \\(Q\\) (or decay \\(\epsilon\\) over time). The policy is on-policy: we evaluate and improve the same epsilon-greedy policy.

**Code:** Same as above but no random (s,a) start—start from `env.reset()`. Use epsilon-greedy to select actions during the episode. Update \\(Q\\) from first-visit (s,a) returns; keep \\(\epsilon\\) fixed or decay it.

See [Chapter 11: Monte Carlo Methods](chapter-01/) for theory and the blackjack exercise, and [TD, SARSA, Q-Learning in Code](td-sarsa-q-in-code/) for temporal-difference algorithms.
