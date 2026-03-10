---
title: "Chapter 57: Dreamer and Latent Imagination"
description: "Simplified Dreamer: RSSM, imagination phase, actor-critic."
date: 2026-03-10T00:00:00Z
weight: 57
draft: false
---

**Learning objectives**

- Implement a **simplified Dreamer**-style algorithm: train an **RSSM-like** model on collected trajectories, then **roll out in latent space** to train an actor-critic.
- Understand the **imagination** phase: no real env steps; only latent rollouts for policy updates.
- Relate to **robot control** and **sample-efficient** RL.

**Concept and real-world RL**

**Dreamer** learns a **recurrent state-space model (RSSM)** in latent space: encode observation to latent, predict next latent given action, predict reward and continue. The **actor-critic** is trained on **imagined** rollouts (latent only), so many gradient steps use no real env interaction. In **robot navigation** and **game AI**, this yields high sample efficiency. The key is training the model and the policy on the same data so the latent space is useful for control.

**Where you see this in practice:** Dreamer v1/v2/v3; used in benchmarks and robotics research.

**Exercise:** Implement a simplified version of the Dreamer algorithm's "imagination" phase: train an RSSM-like model on collected trajectories, then roll out trajectories in the latent space to train an actor-critic.

**Professor's hints**

- RSSM: encoder o_t → z_t; recurrent state h_t; transition (h_t, z_t, a_t) → h_{t+1}, z_{t+1}; reward and continue heads from (h,z). Train on (o,a,r) sequences with reconstruction and prediction losses.
- Imagination: from a latent (h,z) sampled from the buffer, rollout K steps using the model (no env). Compute lambda-return or GAE in latent space; update actor-critic on these imagined trajectories.
- Simplified: you can use a simpler dynamics model (e.g. MLP next latent) instead of full RSSM; the idea is the same.

**Common pitfalls**

- **Distribution shift:** Imagined states may diverge from real latent distribution. Dreamer uses short rollouts and reconditions on real data periodically.
- **Value in latent space:** The critic must be trained on imagined returns; use TD or GAE in the latent rollout.

**Extra practice**

1. **Warm-up:** Why is training the actor on imagined rollouts sample-efficient?
2. **Coding:** Train a simple latent model (encoder + next-latent predictor) on CartPole. Roll out 10 steps in latent space from real encodings. Compare imagined rewards with actual rewards from the env for the same actions.
3. **Challenge:** Implement a full imagination phase: 50 latent steps, actor-critic update. Compare sample efficiency (return vs env steps) with PPO on CartPole.
