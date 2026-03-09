---
title: "Chapter 58: Model-Based Policy Optimization (MBPO)"
description: "MBPO: ensemble dynamics, short rollouts, add to SAC buffer."
date: 2026-03-10T00:00:00Z
weight: 58
draft: false
---

**Exercise:** Implement MBPO for a continuous task: learn an ensemble of dynamics models, use them to generate short rollouts from real states, and add these to the replay buffer for SAC training. Compare with SAC alone.
