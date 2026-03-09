---
title: "Chapter 36: Advantage Actor-Critic (A2C)"
description: "A2C for CartPole with TD error as advantage; sync multi-env."
date: 2026-03-10T00:00:00Z
weight: 36
draft: false
---

**Exercise:** Implement A2C for CartPole. Use the TD error \\(r + \\gamma V(s') - V(s)\\) as the advantage. Use a shared feature extractor or separate networks. Synchronously run multiple environments to collect data.
