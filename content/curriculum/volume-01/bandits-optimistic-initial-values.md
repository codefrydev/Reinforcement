---
title: "Bandits: Optimistic Initial Values"
description: "Using optimistic initial Q-values to encourage early exploration in multi-armed bandits."
date: 2026-03-10T00:00:00Z
weight: 3
draft: false
difficulty: 6
tags: ["bandits", "optimistic initial values", "exploration", "curriculum"]
keywords: ["optimistic initial values", "multi-armed bandits", "exploration", "Q initialization"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Bandits Optimistic Initial Values"
---

**Learning objectives**

- Understand why initializing action values optimistically can encourage exploration.
- Implement optimistic initial values and compare with epsilon-greedy on the 10-armed testbed.
- Recognize when optimistic initialization helps (stationary, deterministic-ish) and when it does not (nonstationary).

## Theory

**Optimistic initial values** mean we set \\(Q(a)\\) to a value *higher* than the typical reward at the start (e.g. \\(Q(a) = 5\\) when rewards are usually in \\([-2, 2]\\)). The agent then chooses the arm with the highest \\(Q(a)\\). After a pull, the running mean update \\(\bar{Q}_{n+1} = \bar{Q}_n + \frac{1}{n+1}(r - \bar{Q}_n)\\) brings \\(Q(a)\\) down toward the true mean. So every arm looks “good” at first; as an arm is pulled, its \\(Q\\) drops toward reality. The agent is naturally encouraged to try all arms before settling, which is a form of **exploration without epsilon**.

This works well in **stationary** environments. In **nonstationary** environments, being optimistic once at the start is not enough—we need ongoing exploration (e.g. constant step size or epsilon-greedy).

## Beginner's exercise prompt

Implement the 10-armed testbed again. This time, use a **greedy** agent (no random exploration) but set \\(Q(a) = +5\\) for all arms at the start (optimistic). Run for 1000 steps and plot average reward over time. Compare with greedy with \\(Q(a) = 0\\) and with epsilon-greedy \\(\epsilon = 0.1\\).

## Code sketch

- Initialize `Q = np.ones(10) * 5` (or another optimistic value).
- Use **greedy** selection: \\(a = \arg\max_a Q(a)\\).
- Update \\(Q(a) \leftarrow Q(a) + \frac{1}{N(a)}(r - Q(a))\\).
- Run many independent runs and average the reward curve.

**Professor's hint:** Optimistic greedy often does better than epsilon-greedy early on because exploration is directed (every arm is tried). For very long runs, epsilon-greedy may still explore occasionally and can help if the environment is slightly nonstationary.

See also [Chapter 2: Multi-Armed Bandits](chapter-02/) for epsilon-greedy and the 10-armed testbed, and [Bandits: Nonstationary](bandits-nonstationary/) when rewards change over time.
