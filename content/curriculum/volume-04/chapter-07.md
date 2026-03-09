---
title: "Chapter 37: Asynchronous Advantage Actor-Critic (A3C)"
description: "A3C with multiprocessing workers; compare speed with A2C."
date: 2026-03-10T00:00:00Z
weight: 37
draft: false
---

**Exercise:** Implement A3C using Python's multiprocessing. Create several worker processes that each interact with its own environment and update a global network asynchronously. Test on CartPole and compare training speed with A2C.
