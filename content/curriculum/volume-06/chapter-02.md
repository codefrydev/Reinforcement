---
title: "Chapter 52: Learning World Models"
description: "Train NN to predict next state from CartPole; compounding error."
date: 2026-03-10T00:00:00Z
weight: 52
draft: false
---

**Learning objectives**

- Collect **random trajectories** from CartPole and train a **neural network** to predict the next state given (state, action).
- Evaluate **prediction accuracy** over 1 step, 5 steps, and 10 steps; observe **compounding error** as the horizon grows.
- Relate model error to the limitations of long-horizon model-based rollouts.

**Concept and real-world RL**

A **world model** (or dynamics model) predicts \\(s_{t+1}\\) from \\(s_t, a_t\\). We can train it on collected data (e.g. MSE loss). Errors **compound** over multi-step rollouts: a small 1-step error becomes large after many steps. In **robot navigation**, learned models are used for short-horizon planning; in **game AI** (e.g. Dreamer), models are used in latent space to reduce dimensionality and control rollouts. Understanding compounding error is key to designing model-based algorithms.

**Where you see this in practice:** World models appear in Dreamer, MBPO, and PILCO; robotics often uses short model rollouts.

**Exercise:** Collect random trajectories from the CartPole environment. Train a neural network to predict the next state given current state and action. Evaluate its prediction accuracy over multiple steps and note the compounding error.

**Professor's hints**

- Data: run random policy for 10k steps, store (s, a, s'). Train: input (s, a), target s'; use MSE. Split train/val.
- Evaluation: from a held-out s0, rollout the model for 1, 5, 10 steps (feeding predicted s as input). Compare predicted s_t to true s_t from the env (same actions). Plot MSE vs step; it typically grows.
- Compounding: small errors at each step accumulate; the model never saw its own predictions during training (distribution shift).

**Common pitfalls**

- **Training on same distribution as evaluation:** When you rollout the model, you feed it *predicted* states, not real ones. So long-horizon error reflects distribution shift (model sees its own errors).
- **Normalization:** Normalize state and action for training; denormalize for env comparison.

**Extra practice**

1. **Warm-up:** Why does 10-step prediction error usually exceed 1-step error?
2. **Coding:** Train the model; plot 1-step, 5-step, 10-step MSE on a validation set. Fit a curve (e.g. exponential) to error vs step.
3. **Challenge:** Use an **ensemble** of 5 models and take the mean prediction. Does ensemble reduce compounding error over 10 steps?
