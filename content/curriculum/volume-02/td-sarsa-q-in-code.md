---
title: "TD, SARSA, and Q-Learning in Code"
description: "Code walkthrough for TD(0) prediction, SARSA, and Q-learning (tabular)."
date: 2026-03-10T00:00:00Z
weight: 13
draft: false
difficulty: 6
tags: ["temporal difference", "SARSA", "Q-learning", "code", "curriculum"]
keywords: ["TD(0)", "SARSA", "Q-learning", "tabular", "code"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Td Sarsa Q In Code"
---

**Learning objectives**

- Implement TD(0) prediction in code: update \\(V(s)\\) after each transition.
- Implement SARSA (on-policy TD control): update \\(Q(s,a)\\) using the next action from the behavior policy.
- Implement Q-learning (off-policy TD control): update \\(Q(s,a)\\) using the max over next actions.

## TD(0) prediction in code

**Goal:** Estimate \\(V^\\pi\\) for a fixed policy \\(\pi\\).

**Update:** After each transition \\((s, r, s')\\):
\[
V(s) \leftarrow V(s) + \alpha \bigl[ r + \gamma V(s') - V(s) \bigr]
\]
Use \\(V(s') = 0\\) if \\(s'\\) is terminal.

**Code:** Initialize \\(V\\) (e.g. dict or array, all zeros). For each episode: get \\(s\\) from env.reset(). Loop: choose \\(a \sim \pi(s)\\), step env to get \\(r, s', \mathrm{done}\\). Update `V[s] += alpha * (r + gamma * (0 if done else V[s']) - V[s])`. Set \\(s \leftarrow s'\\); repeat until done.

## SARSA in code

**Goal:** Learn \\(Q^\\pi\\) and improve an on-policy policy (e.g. epsilon-greedy).

**Update:** After each transition \\((s, a, r, s')\\), we have the *next* action \\(a' \sim \pi(s')\\) (e.g. epsilon-greedy). Then:
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma Q(s', a') - Q(s,a) \bigr]
\]

**Code:** Initialize \\(Q(s,a)\\). Each episode: \\(s = \mathrm{env.reset()}\\), \\(a = \\) epsilon-greedy from \\(Q(s,\cdot)\\). Loop: step with \\(a\\) to get \\(r, s', \mathrm{done}\\). Choose \\(a' = \\) epsilon-greedy from \\(Q(s', \cdot)\\). Update `Q[s,a] += alpha * (r + gamma * (0 if done else Q[s',a']) - Q[s,a])`. Set \\(s,a \leftarrow s', a'\\); repeat until done.

## Q-learning in code

**Goal:** Learn \\(Q^*\\) (off-policy): we take actions from an exploratory policy (e.g. epsilon-greedy) but update using the *best* next action.

**Update:** After \\((s, a, r, s')\\):
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \bigr]
\]

**Code:** Same as SARSA but: when computing the target, use \\(\max_{a'} Q(s', a')\\) instead of \\(Q(s', a')\\). So after stepping, do `target = r + gamma * (0 if done else max(Q[s'].values())` or `Q[s'].max()`, then `Q[s,a] += alpha * (target - Q[s,a])`. Action selection for the *next* step is still epsilon-greedy (or greedy for evaluation).

See [Chapter 12: TD Learning](chapter-02/), [Chapter 13: SARSA](chapter-03/), and [Chapter 14: Q-Learning](chapter-04/) for theory.
